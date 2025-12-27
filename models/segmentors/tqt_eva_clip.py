"""
TQT-EVA-CLIP 分割模型

基于 EVA-CLIP 的可通行区域语义分割模型，支持消融实验配置:

消融实验配置说明:
================
| 配置项              | 说明                                          | 默认值        |
|---------------------|----------------------------------------------|---------------|
| use_sne             | 是否使用 Surface Normal 双模态融合            | False         |
| sne_fusion_stage    | SNE融合阶段: 'backbone'(segmentor中) / 'pixel'(head中) | 'backbone' |
| sne_fusion_mode     | SNE融合方式: 'proj'/'add'/'concat'/'cross_attn'/'ot' | 'proj'     |
| use_ot_align        | 是否使用最优传输进行 img/sne 与 text 的对齐    | False         |
| prompt_cls          | 是否使用动态场景感知提示 (天气/光照/路面)       | False         |
| prompt_cls_type     | 场景分类方式: 'text_encoder'/'linear'/'linear_text' | 'text_encoder' |
| use_context_decoder | 是否使用 context decoder 增强 text embedding   | True          |
| use_learnable_prompt| 是否使用可学习的 prompt prefix (CoOp)          | True          |

融合阶段说明:
=============
- backbone: 在 segmentor 中完成 RGB-SNE 融合，融合后的特征传给 head (sne_feature=None)
- pixel: 在 head 中使用独立 pixel_decoder 并行处理 RGB 和 SNE 后融合

prompt_cls_type 说明:
====================
- text_encoder: 使用 CLIP text encoder 编码属性文本进行分类 + 生成场景原型 (原始方案)
- linear: 使用 3 组可学习类别原型 + 24 个可学习场景向量 (消融: 验证 text encoder 必要性)
- linear_text: 使用 3 组可学习类别原型 + text encoder 生成场景原型 (混合方案)
- learnable_only: 仅使用 2 个可学习掩码向量，不使用任何文本编码 (基线: 验证文本模态必要性)

注：linear/linear_text 模式的属性分类使用余弦相似度 (与 text_encoder 一致)，保证消融实验公平性
注：learnable_only 模式自动禁用 context_decoder 和 text/visual 正则化

消融实验组合示例:
================
- Baseline (tqdm):             use_sne=False
- +SNE (backbone proj):        use_sne=True, sne_fusion_stage='backbone', sne_fusion_mode='proj'
- +SNE (backbone cross_attn):  use_sne=True, sne_fusion_stage='backbone', sne_fusion_mode='cross_attn'
- +SNE (backbone OT):          use_sne=True, sne_fusion_stage='backbone', sne_fusion_mode='ot'
- +SNE (pixel proj):           use_sne=True, sne_fusion_stage='pixel', sne_fusion_mode='proj'
- +SNE (pixel cross_attn):     use_sne=True, sne_fusion_stage='pixel', sne_fusion_mode='cross_attn'
- +PromptCls:                  prompt_cls=True
- +PromptCls (linear):         prompt_cls=True, prompt_cls_type='linear'
- +PromptCls (linear_text):    prompt_cls=True, prompt_cls_type='linear_text'
- LearnableOnly (no text):     prompt_cls=True, prompt_cls_type='learnable_only'
- -ContextDecoder:             use_context_decoder=False
- -LearnablePrompt:            use_learnable_prompt=False
"""

import copy
import json
import itertools
import os
from pathlib import Path

import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models.builder import SEGMENTORS
from models.segmentors import tqdm_EVA_CLIP, OTFeatureAligner

from ..backbones.utils import tokenize


def maybe_pad(input_feature, height=None, width=None, data_format="channels_last"):
    """Pad spatial dims to even sizes for pixel shuffle/unshuffle paths."""
    if data_format == "channels_first":
        input_feature = input_feature.permute(0, 2, 3, 1).contiguous()

    if height is None or width is None:
        height, width = input_feature.shape[1:-1]

    if (height % 2 == 1) or (width % 2 == 1):
        pad_values = (0, 0, 0, width % 2, 0, height % 2)
        input_feature = F.pad(input_feature, pad_values)

    if data_format == "channels_first":
        input_feature = input_feature.permute(0, 3, 1, 2).contiguous()

    return input_feature


def pixel_shuffle(x, scale_factor=0.5, data_format="channels_last"):
    """Bidirectional pixel shuffle for up/down sampling in NHWC/NCHW."""
    if data_format == "channels_first":
        x = x.permute(0, 2, 3, 1).contiguous()

    b, h, w, c = x.size()
    scale = float(scale_factor)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # intermediate channel for first reshape; safe for both up/down paths
    mid_c = int(round(c / scale))
    out_c = int(round(c / (scale * scale)))

    x = x.reshape(b, h, new_w, mid_c)
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(b, new_w, new_h, out_c)
    x = x.permute(0, 2, 1, 3).contiguous()

    if data_format == "channels_first":
        x = x.permute(0, 3, 1, 2).contiguous()

    return x


class PixelSamplingBlock(nn.Module):
    """PixelShuffle/Unshuffle based up/down sampling block."""

    def __init__(self, dim, scale_factor, norm_layer=nn.SyncBatchNorm, act_layer=nn.GELU):
        super().__init__()
        self.scale_factor = scale_factor
        self.is_upsample = scale_factor > 1

        if self.is_upsample:
            expand_ratio = int(scale_factor ** 2)
            self.proj = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1)
        else:
            down_ratio = int((1 / scale_factor) ** 2)
            self.proj = nn.Conv2d(dim * down_ratio, dim, kernel_size=1)

        self.norm = norm_layer(dim)
        self.act = act_layer()

    def forward(self, x):
        x = maybe_pad(x, data_format="channels_first")

        if self.is_upsample:
            x = self.proj(x)
            x = pixel_shuffle(x, scale_factor=self.scale_factor, data_format="channels_first")
        else:
            x = pixel_shuffle(x, scale_factor=self.scale_factor, data_format="channels_first")
            x = self.proj(x)

        x = self.norm(x)
        x = self.act(x)
        return x


class BidirectionalCrossAttention(nn.Module):
    """双向交叉注意力模块，用于两个特征序列的相互增强。"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn_1to2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_2to1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        """双向交叉注意力前向传播。"""
        # x1 用 x2 增强
        attn_out1, attn_weights_1to2 = self.cross_attn_1to2(query=x1, key=x2, value=x2)
        x1 = self.norm1(x1 + attn_out1)

        # x2 用 x1 增强
        attn_out2, attn_weights_2to1 = self.cross_attn_2to1(query=x2, key=x1, value=x1)
        x2 = self.norm2(x2 + attn_out2)

        return x1, x2, attn_weights_1to2, attn_weights_2to1


@SEGMENTORS.register_module()
class tqt_EVA_CLIP(tqdm_EVA_CLIP):
    """TQT-EVA-CLIP: 支持消融实验的可通行区域分割模型。

    消融实验配置:
        - use_sne (bool): 启用 Surface Normal 双模态融合，默认 False
        - sne_fusion_stage (str): 融合阶段，'backbone'(segmentor中)/'pixel'(head中)，默认 'backbone'
        - sne_fusion_mode (str): 融合方式，'proj'/'add'/'cross_attn'/'ot'，默认 'proj'
        - use_ot_align (bool): 启用最优传输对齐 (img/sne -> text)，默认 False
        - prompt_cls (bool): 启用动态场景感知提示，默认 False
        - prompt_cls_type (str): 场景分类方式，默认 'text_encoder'
            - 'text_encoder': CLIP text encoder 分类 + 生成原型 (原始方案)
            - 'linear': 可学习类别原型 + 可学习场景向量 (消融: 验证 text encoder 必要性)
            - 'linear_text': 可学习类别原型 + text encoder 生成场景原型 (混合方案)
            - 'learnable_only': 仅可学习掩码向量，无文本编码 (基线: 验证文本模态)
            注：所有模式均使用余弦相似度进行分类，保证消融实验公平性
        - use_context_decoder (bool): 启用 context decoder 增强 text，默认 True
        - use_learnable_prompt (bool): 启用可学习 prompt prefix (CoOp)，默认 True

    消融实验组合示例:
        1. Baseline (与 tqdm 一致):
           use_sne=False, prompt_cls=False, use_context_decoder=True, use_learnable_prompt=True

        2. +SNE (简单融合):
           use_sne=True, sne_fusion_mode='simple'

        3. +SNE (双向交叉注意力):
           use_sne=True, sne_fusion_mode='cross_attn'

        4. +SNE (最优传输融合):
           use_sne=True, sne_fusion_mode='ot'

        5. +SNE + 全局OT对齐:
           use_sne=True, sne_fusion_mode='ot', use_ot_align=True

        6. +动态场景提示 (text_encoder):
           prompt_cls=True, prompt_cls_type='text_encoder'

        7. +动态场景提示 (linear, 消融text encoder):
           prompt_cls=True, prompt_cls_type='linear'

        8. +动态场景提示 (linear_text, 混合方案):
           prompt_cls=True, prompt_cls_type='linear_text'

        9. -Context Decoder:
           use_context_decoder=False

        10. -可学习Prompt:
            use_learnable_prompt=False
    """

    # 场景属性类别定义
    WEATHER_CLASSES = ['sunny', 'snowy', 'foggy', 'rainy']
    ROAD_CLASSES = ['paved road', 'unpaved road']
    LIGHT_CLASSES = ['daytime', 'dusk', 'nighttime']
    MATERIAL_CLASSES = [
        'rural', 'forest', 'farmland', 'grassland', 'sandstone',
        'field', 'concrete', 'asphalt', 'riverside', 'dirt'
    ]

    def __init__(
        self,
        eva_clip,
        decode_head,
        class_names,
        context_length,
        context_decoder=None,
        token_embed_dim=512,
        text_dim=512,
        visual_dim=768,                   # 视觉特征维度 (由 model_name 决定)
        neck=None,
        identity_head=None,
        visual_reg=True,
        textual_reg=True,
        force_reg_e0_eval=True,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        patch_fpn=False,                 # 额外开关：使用补丁式 FPN 重建（patch expand/merge）
        patch_fpn_xsam=False,            # 额外开关：使用 PixelSamplingBlock 版补丁式 FPN
        supervise_ot_pi=False,           # 额外开关：对 OT 传输计划 pi 进行分割掩码深监督
        # ====== 消融实验配置 ======
        use_sne=False,                    # 1. 是否使用 SNE 双模态
        sne_fusion_stage='backbone',      # 2. 融合阶段: 'backbone' / 'pixel'
        sne_fusion_mode='proj',           # 3. 融合方式: 'proj' / 'add' / 'cross_attn' / 'ot'
        use_ot_align=False,               # 4. 是否使用 OT 对齐 (img/sne -> text)
        ot_use_score_prior=True,          # 5. OT 是否使用预测图来分配文本分布权重
        ot_score_prior_mode='argmax',     # 5.1 OT 文本先验模式: 'argmax' / 'prob'
        ot_score_prior_temperature=1.0,   # 5.2 OT prob 模式的温度系数
        ot_cost_type='cos',               # 6. OT 成本矩阵计算方式: 'cos' 或 'l2'
        ot_fuse_output=True,              # 7. OTFeatureAligner 输出是否做残差融合
        ot_fuse_mode='proj',              # 8. OT 融合方式: 'proj' / 'mean' / 'max' / 'config'
        ot_softunion=False,               # 9. 是否在 score_map 上做 soft union 融合
        prompt_cls=False,                 # 6. 是否使用动态场景感知提示
        prompt_cls_topk_train=None,       # 6.4 训练时 joint_probs TopK (None=全部, 10=Top10)
        prompt_cls_topk_test=None,        # 6.5 测试时 joint_probs TopK (None=全部, 3=Top3)
        prompt_cls_mode='hard',           # 6.1 动态提示模式: 'hard' (argmax) / 'soft' (weighted sum)
        prompt_cls_temperature_mode='ot_prior', # 6.2 动态提示温度模式: 'ot_prior' (默认) / 'tau'
        prompt_cls_type='text_encoder',   # 6.3 场景分类方式: 'text_encoder' / 'linear' / 'linear_text'
        use_material_cls=False,           # 6.6 是否使用材质分类 (增加场景组合数)
        use_context_decoder=True,         # 6. 是否使用 context decoder
        use_learnable_prompt=True,        # 7. 是否使用可学习 prompt prefix
        use_score_map_reg=True,           # 8. 是否使用 Vision-Language score_map 正则化
        **args
    ):
        # 调用父类初始化
        super().__init__(
            eva_clip=eva_clip,
            decode_head=decode_head,
            class_names=class_names,
            context_length=context_length,
            context_decoder=context_decoder if use_context_decoder else None,
            token_embed_dim=token_embed_dim,
            text_dim=text_dim,
            neck=neck,
            identity_head=identity_head,
            visual_reg=visual_reg,
            textual_reg=textual_reg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **args
        )

        # ====== 保存消融实验配置 ======
        self.use_sne = use_sne
        self.sne_fusion_stage = sne_fusion_stage
        self.sne_fusion_mode = sne_fusion_mode
        self.use_ot_align = use_ot_align
        self.ot_use_score_prior = ot_use_score_prior
        self.ot_score_prior_mode = ot_score_prior_mode
        
        # 如果传入 None，则使用可学习的温度系数 (初始值 1.0)
        if ot_score_prior_temperature is None:
            self.ot_score_prior_temperature = nn.Parameter(torch.tensor(0.1))
        else:
            self.ot_score_prior_temperature = ot_score_prior_temperature
            
        self.ot_cost_type = ot_cost_type
        self.ot_fuse_output = ot_fuse_output
        self.ot_fuse_mode = ot_fuse_mode
        self.ot_softunion = ot_softunion
        self.prompt_cls = prompt_cls
        self.prompt_cls_mode = prompt_cls_mode
        self.prompt_cls_temperature_mode = prompt_cls_temperature_mode
        self.prompt_cls_type = prompt_cls_type
        self.prompt_cls_topk_train = prompt_cls_topk_train
        self.prompt_cls_topk_test = prompt_cls_topk_test
        self.use_material_cls = use_material_cls
        self.use_context_decoder = use_context_decoder
        self.use_learnable_prompt = use_learnable_prompt
        self.use_score_map_reg = use_score_map_reg
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.patch_fpn_xsam = patch_fpn_xsam
        self.patch_fpn = patch_fpn or patch_fpn_xsam
        self.supervise_ot_pi = supervise_ot_pi
        self.force_reg_e0_eval = force_reg_e0_eval
        self._reg_e0_eval_logged = False
        self.is_deploy = False

        # 训练阶段可选的可视化/调试开关
        train_debug_cfg = train_cfg.get('debug_vis', {}) if isinstance(train_cfg, dict) else {}
        self.train_debug_max = int(train_debug_cfg.get('max_samples', 0))
        self.train_debug_every = max(1, int(train_debug_cfg.get('interval', 1)))
        self.train_debug_save_debug = bool(train_debug_cfg.get('save_debug', False))
        self.train_debug_out_dir = train_debug_cfg.get('output_dir', None)
        self._train_debug_seen = 0

        assert sne_fusion_stage in ['backbone', 'pixel'], \
            "sne_fusion_stage must be 'backbone' or 'pixel'"
        assert sne_fusion_mode in ['proj', 'add', 'concat', 'cross_attn', 'ot'], \
            "sne_fusion_mode must be 'proj', 'add', 'concat', 'cross_attn' or 'ot'"
        assert ot_fuse_mode in ['proj', 'mean', 'max', 'config'], "ot_fuse_mode must be 'proj', 'mean', 'max' or 'config'"
        assert ot_score_prior_mode in ['argmax', 'prob'], "ot_score_prior_mode must be 'argmax' or 'prob'"
        assert prompt_cls_type in ['text_encoder', 'linear', 'linear_text', 'learnable_only'], \
            "prompt_cls_type must be 'text_encoder', 'linear', 'linear_text' or 'learnable_only'"

        # learnable_only 模式：自动禁用 context_decoder 和正则化
        if prompt_cls_type == 'learnable_only':
            self.use_context_decoder = False
            self.context_decoder = None
            # 注意：visual_reg 和 textual_reg 在父类中已设置，
            # 我们在 forward_train 中根据 prompt_cls_type 跳过正则化损失

        # 打印消融实验配置
        self._print_ablation_config()

        # 加载场景信息字典 (用于 prompt_cls，learnable_only 模式不需要)
        # if self.prompt_cls and self.prompt_cls_type != 'learnable_only':
        scene_dict_path = '/root/tqdm/dataset/ORFD/english_scene_dict.json'
        with open(scene_dict_path, 'r') as f:
            self.scene2info = json.load(f)

        # ====== 初始化计数器变量 (与 SNE 解耦) ======
        self.save_img_sne_sum = 0
        self.save_img_sne_sum_val = 0

        # ====== 根据配置初始化模块 ======
        if self.use_sne:
            self._init_sne_modules(visual_dim, text_dim)
        if self.prompt_cls:
            self._init_prompt_modules(token_embed_dim, context_length)

        # 可选：基于中尺度特征用 patch expand/merge 重建金字塔，保持与默认 FPN 兼容
        if self.patch_fpn:
            self.patch_expand1 = nn.Sequential(
                nn.ConvTranspose2d(visual_dim, visual_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(visual_dim),
                nn.GELU(),
            )
            self.patch_expand2 = nn.Sequential(
                nn.ConvTranspose2d(visual_dim, visual_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(visual_dim),
                nn.GELU(),
            )
            self.patch_merge1 = nn.Sequential(
                nn.Conv2d(visual_dim, visual_dim, kernel_size=3, stride=2, padding=1),
                nn.SyncBatchNorm(visual_dim),
                nn.GELU(),
            )
        elif self.patch_fpn_xsam:
            self.patch_expand1 = PixelSamplingBlock(visual_dim, scale_factor=2)
            self.patch_expand2 = PixelSamplingBlock(visual_dim, scale_factor=2)
            self.patch_merge1 = PixelSamplingBlock(visual_dim, scale_factor=0.5)

        # 如果不使用 context_decoder，置为 None
        if not self.use_context_decoder:
            self.context_decoder = None

    def _print_ablation_config(self):
        """打印消融实验配置信息。"""
        config_str = (
            f"\n{'='*60}\n"
            f"[TQT Ablation Config]\n"
            f"{'='*60}\n"
            f"  use_sne:              {self.use_sne}\n"
            f"  sne_fusion_stage:     {self.sne_fusion_stage}\n"
            f"  sne_fusion_mode:      {self.sne_fusion_mode}\n"
            f"  use_ot_align:         {self.use_ot_align}\n"
            f"  ot_cost_type:         {self.ot_cost_type}\n"
            f"  ot_fuse_output:       {self.ot_fuse_output}\n"
            f"  ot_score_prior_mode:  {self.ot_score_prior_mode}\n"
            f"  ot_score_prior_temp:  {self.ot_score_prior_temperature}\n"
            f"  ot_softunion:         {self.ot_softunion}\n"
            f"  prompt_cls:           {self.prompt_cls}\n"
            f"  prompt_cls_mode:      {self.prompt_cls_mode}\n"
            f"  prompt_cls_type:      {self.prompt_cls_type}\n"
            f"  use_material_cls:     {self.use_material_cls}\n"
            f"  use_context_decoder:  {self.use_context_decoder}\n"
            f"  use_learnable_prompt: {self.use_learnable_prompt}\n"
            f"  supervise_ot_pi:      {self.supervise_ot_pi}\n"
            f"  force_reg_e0_eval:    {self.force_reg_e0_eval}\n"
            f"{'='*60}"
        )
        print(config_str)

    def _init_sne_modules(self, visual_dim, text_dim):
        """初始化 SNE 相关模块。"""
        # SNE backbone (共享权重初始化)
        self.sne_backbone = copy.deepcopy(self.backbone)

        # SNE 分支的 context decoder (如果启用)
        if self.use_context_decoder and self.context_decoder is not None:
            self.context_decoder_sne = copy.deepcopy(self.context_decoder)
            self.gamma_sne = nn.Parameter(torch.ones(text_dim) * 1e-4)
            nn.init.trunc_normal_(self.gamma_sne)
        else:
            self.context_decoder_sne = None

        # ====== 只在 backbone 阶段融合时初始化融合模块 ======
        # pixel 阶段的融合模块在 tqtHead 中初始化
        if self.sne_fusion_stage == 'backbone':
            if self.sne_fusion_mode == 'cross_attn':
                # 双向交叉注意力
                self.cross_attn_modules = nn.ModuleList([
                    BidirectionalCrossAttention(visual_dim) for _ in range(4)
                ])
                # 融合投影层
                self.img_sne_proj = nn.ModuleList([
                    nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1) for _ in range(4)
                ])

            elif self.sne_fusion_mode == 'ot':
                # 最优传输对齐模块
                self.img_attn = nn.ModuleList([
                    OTFeatureAligner(
                        visual_dim,
                        use_score_prior=self.ot_use_score_prior,
                        score_prior_mode=self.ot_score_prior_mode,
                        # ot_score_prior_temperature=self.ot_score_prior_temperature, # Removed
                        fuse_output=self.ot_fuse_output,
                        cost_type=self.ot_cost_type,
                    )
                    for _ in range(4)
                ])
                self.sne_attn = nn.ModuleList([
                    OTFeatureAligner(
                        visual_dim,
                        use_score_prior=self.ot_use_score_prior,
                        score_prior_mode=self.ot_score_prior_mode,
                        # ot_score_prior_temperature=self.ot_score_prior_temperature, # Removed
                        fuse_output=self.ot_fuse_output,
                        cost_type=self.ot_cost_type,
                    )
                    for _ in range(4)
                ])
                self.text_proj = nn.Linear(text_dim, visual_dim)
                # 融合投影层
                self.img_sne_proj = nn.ModuleList([
                    nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1) for _ in range(4)
                ])

            elif self.sne_fusion_mode == 'proj':
                # 简单 proj 融合：投影层
                self.img_sne_proj = nn.ModuleList([
                    nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1) for _ in range(4)
                ])
            # 'add' 模式不需要额外模块

        # 全局 OT 对齐 (可选)
        if self.use_ot_align:
            self.opt_sne = OTFeatureAligner(
                dim=text_dim,
                use_score_prior=self.ot_use_score_prior,
                score_prior_mode=self.ot_score_prior_mode,
                # score_prior_temperature=self.ot_score_prior_temperature, # Removed
                fuse_output=self.ot_fuse_output,
                cost_type=self.ot_cost_type,
            )
            self.opt_img = OTFeatureAligner(
                dim=text_dim,
                use_score_prior=self.ot_use_score_prior,
                score_prior_mode=self.ot_score_prior_mode,
                # score_prior_temperature=self.ot_score_prior_temperature, # Removed
                fuse_output=self.ot_fuse_output,
                cost_type=self.ot_cost_type,
            )
            if not hasattr(self, 'text_proj'):
                self.text_proj = nn.Linear(text_dim, visual_dim)

    def _init_prompt_modules(self, token_embed_dim, context_length):
        """初始化 Prompt 相关模块。

        根据 prompt_cls_type 初始化不同的分类和场景原型模块:
        - 'text_encoder': 使用 CLIP text encoder 编码属性文本 + 生成场景原型
        - 'linear': 使用 3 个 linear 分类器 + 24 个可学习场景向量
        - 'linear_text': 使用 3 个 linear 分类器 + text encoder 生成场景原型
        - 'learnable_only': 仅使用 2 个可学习掩码向量 (无文本编码器)
        """
        # learnable_only 模式：仅初始化可学习掩码向量，跳过其他所有模块
        if self.prompt_cls_type == 'learnable_only':
            # 创建 2 个可学习的类别原型向量 (traversable / non-traversable)
            # 形状: [2, text_dim]，直接作为分类向量
            self.learnable_class_prototypes = nn.Parameter(
                torch.randn(2, self.text_dim)
            )
            nn.init.trunc_normal_(self.learnable_class_prototypes, std=0.02)

            # 不需要其他 prompt 模块
            self.contexts = None
            self.contexts_traversable = None
            self.contexts_notraversable = None
            return

        # 可学习 prompt (CoOp style) - 仅 text_encoder 和 linear_text 模式需要
        if self.use_learnable_prompt and self.prompt_cls_type != 'linear':
            # 父类已初始化 self.contexts，这里初始化额外的
            self.contexts_traversable = nn.Parameter(
                torch.randn(1, self.prompt_num, token_embed_dim)
            )
            self.contexts_notraversable = nn.Parameter(
                torch.randn(1, self.prompt_num, token_embed_dim)
            )
            nn.init.trunc_normal_(self.contexts_traversable)
            nn.init.trunc_normal_(self.contexts_notraversable)
        else:
            # 不使用可学习 prompt，使用固定 context (None)
            self.contexts = None
            self.contexts_traversable = None
            self.contexts_notraversable = None

        # 场景感知提示分类模块
        if self.prompt_cls:
            if self.prompt_cls_type == 'text_encoder':
                # 原始方案：使用 text encoder 编码属性文本进行分类
                self.weather_prompt = nn.Parameter(
                    torch.randn(1, self.prompt_num, token_embed_dim)
                )
                self.light_prompt = nn.Parameter(
                    torch.randn(1, self.prompt_num, token_embed_dim)
                )
                self.road_prompt = nn.Parameter(
                    torch.randn(1, self.prompt_num, token_embed_dim)
                )
                if self.use_material_cls:
                    self.material_prompt = nn.Parameter(
                        torch.randn(1, self.prompt_num, token_embed_dim)
                    )
                self._init_attribute_embeddings(context_length)

            elif self.prompt_cls_type in ['linear', 'linear_text']:
                # 消融方案：使用可学习的类别原型向量 (用余弦相似度分类，与 text_encoder 保持一致)
                # 形状: [num_classes, text_dim]，作为类别原型与图像特征计算余弦相似度
                self.weather_prototypes = nn.Parameter(
                    torch.randn(len(self.WEATHER_CLASSES), self.text_dim)
                )
                self.light_prototypes = nn.Parameter(
                    torch.randn(len(self.LIGHT_CLASSES), self.text_dim)
                )
                self.road_prototypes = nn.Parameter(
                    torch.randn(len(self.ROAD_CLASSES), self.text_dim)
                )
                if self.use_material_cls:
                    self.material_prototypes = nn.Parameter(
                        torch.randn(len(self.MATERIAL_CLASSES), self.text_dim)
                    )
                    nn.init.trunc_normal_(self.material_prototypes, std=0.02)

                # 初始化类别原型
                nn.init.trunc_normal_(self.weather_prototypes, std=0.02)
                nn.init.trunc_normal_(self.light_prototypes, std=0.02)
                nn.init.trunc_normal_(self.road_prototypes, std=0.02)

                if self.prompt_cls_type == 'linear':
                    # 纯 linear 方案：场景向量数量取决于是否使用材质分类
                    # 不使用材质: 4 天气 x 3 光照 x 2 道路 = 24
                    # 使用材质:   4 天气 x 3 光照 x 2 道路 x 10 材质 = 240
                    # 每个场景有 2 个类别 (traversable / non-traversable)
                    num_scenes = (
                        len(self.WEATHER_CLASSES) * len(self.LIGHT_CLASSES) * len(self.ROAD_CLASSES)
                    )
                    if self.use_material_cls:
                        num_scenes *= len(self.MATERIAL_CLASSES)
                    self.scene_prototypes = nn.Parameter(
                        torch.randn(num_scenes, 2, self.text_dim)
                    )
                    nn.init.trunc_normal_(self.scene_prototypes, std=0.02)
                else:
                    # linear_text 方案：仍使用 text encoder 生成场景原型
                    # 需要初始化属性文本 embedding (仅用于生成场景原型)
                    self._init_attribute_embeddings(context_length)

    def _init_attribute_embeddings(self, context_length):
        """初始化场景属性的文本嵌入。"""
        self.attr_texts = {}
        self.attr_t0 = {}
        self.attr_I = {}

        # 基础属性列表
        attr_list = [
            ('weather', self.WEATHER_CLASSES),
            ('road', self.ROAD_CLASSES),
            ('light', self.LIGHT_CLASSES),
        ]
        # 如果启用材质分类，添加材质属性
        if self.use_material_cls:
            attr_list.append(('material', self.MATERIAL_CLASSES))

        for name, classes in attr_list:
            self.attr_texts[name] = torch.cat([
                tokenize(c, context_length=context_length) for c in classes
            ]).to('cuda')
            self.attr_t0[name] = self._compute_normalized_text_features(classes)
            self.attr_I[name] = torch.arange(len(classes), device='cuda')

    def _compute_normalized_text_features(self, class_names):
        """计算归一化的文本特征用于正则化。"""
        tokens = torch.cat([
            tokenize(f"A photo of {c}", context_length=self.text_encoder.context_length)
            for c in class_names
        ]).to('cuda')
        features = self.text_encoder(tokens, context=None)
        return F.normalize(features, dim=-1, p=2)

    # =========================================================================
    # 特征提取方法
    # =========================================================================

    def extract_feat(self, img, sne=None):
        """提取图像特征和可选的 SNE 特征。"""
        x = self.backbone.extract_feats(img)
        if self.patch_fpn:
            x = self._apply_patch_fpn(x)

        if self.use_sne and sne is not None:
            sne_feature = self.sne_backbone.extract_feats(sne)
            if self.patch_fpn:
                sne_feature = self._apply_patch_fpn(sne_feature)
            return x, sne_feature

        return x, None

    def after_extract_feat(self, x, sne=None):
        """后处理特征提取结果。
        
        根据消融实验配置选择不同的处理路径。
        """
        if self.use_sne:
            assert sne is not None, "SNE feature is required when use_sne=True"

        # ======== 1. 分离多尺度特征和全局嵌入 ========
        x_orig = list(x[:-1])
        global_feat, visual_embeddings = x[-1]
        b_size = global_feat.shape[0]

        sne_orig = None
        sne_embeddings = None
        if self.use_sne and sne is not None:
            sne_orig = list(sne[:-1])
            _, sne_embeddings = sne[-1]

        # 构建返回的多尺度特征
        new_orig = (x_orig, sne_orig) if sne_orig is not None else x_orig

        # ======== 2. 构建 Context (用于 Context Decoder) ========
        visual_context = torch.cat(
            [global_feat, visual_embeddings.flatten(-2).permute(0, 2, 1)], dim=1
        )

        sne_context = None
        if self.use_sne and sne is not None and self.context_decoder_sne is not None:
            global_sne_feat, sne_emb = sne[-1]
            sne_context = torch.cat(
                [global_sne_feat, sne_emb.flatten(-2).permute(0, 2, 1)], dim=1
            )

        # ======== 3. 生成文本嵌入 ========
        if self.prompt_cls:
            if self.prompt_cls_type == 'learnable_only':
                # learnable_only 模式：直接使用可学习的类别原型向量
                # self.learnable_class_prototypes: [2, text_dim] -> [B, 2, text_dim]
                text_embeddings = self.learnable_class_prototypes.unsqueeze(0).expand(b_size, -1, -1)
                # global_feat 保持不变，不需要返回额外的分类信息
                global_feat = (global_feat, None, None)
            else:
                text_embeddings, global_feat = self._generate_dynamic_text_embeddings(
                    global_feat, b_size
                )
        else:
            # 根据是否使用可学习 prompt
            context = self.contexts if self.use_learnable_prompt else None
            text_embeddings = self.text_encoder(
                self.texts, context=context
            ).expand(b_size, -1, -1)

        # ======== 4. Context Decoder 融合 (可选) ========
        if self.use_context_decoder and self.context_decoder is not None:
            text_diff = self.context_decoder(text_embeddings, visual_context)
            if sne_context is not None and self.context_decoder_sne is not None:
                text_sne_diff = self.context_decoder_sne(text_embeddings, sne_context)
                text_embeddings = (
                    text_embeddings +
                    self.gamma * text_diff +
                    self.gamma_sne * text_sne_diff
                )
            else:
                text_embeddings = text_embeddings + self.gamma * text_diff

        ret_text_emb = text_embeddings

        # ======== 5. 计算相似度图 ========
        text_embeddings_norm = F.normalize(text_embeddings, dim=-1, p=2)
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)

        score_map = {
            'img': torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, text_embeddings_norm)
        }
        if sne_embeddings is not None:
            sne_embeddings_norm = F.normalize(sne_embeddings, dim=1, p=2)
            score_map['sne'] = torch.einsum(
                'bchw,bkc->bkhw', sne_embeddings_norm, text_embeddings_norm
            )

        # ======== 6. 构建原始嵌入返回值 ========
        orig_embeddings = (x[-1], sne[-1] if sne is not None else None)

        return new_orig, score_map, ret_text_emb, global_feat, orig_embeddings

    def _apply_patch_fpn(self, feats_all):
        """对 backbone 输出的多尺度特征进行补丁式金字塔重建，保持尾部全局特征不变。"""
        if not isinstance(feats_all, (list, tuple)) or len(feats_all) < 2:
            return feats_all
        pyramid = list(feats_all[:-1])
        tail = feats_all[-1]
        pyramid = self._rebuild_pyramid_from_mid(pyramid)
        return pyramid + [tail]

    def _rebuild_pyramid_from_mid(self, feats):
        """使用中尺度特征经 patch expand/merge 重建四层金字塔。"""
        if len(feats) < 3:
            return feats
        base = feats[2]
        p2 = self.patch_expand1(base)
        p1 = self.patch_expand2(p2)
        p3 = base
        p4 = self.patch_merge1(base)
        return [p1, p2, p3, p4]

    def switch_to_deploy(self):
        """
        切换到部署模式：预计算所有文本嵌入，并删除 Text Encoder 以节省资源。

        根据 prompt_cls_type 执行不同的缓存策略:
        - 'text_encoder': 缓存属性 embeddings + 场景原型，删除 text encoder
        - 'linear': 直接使用可学习的 scene_prototypes，删除 text encoder
        - 'linear_text': 缓存场景原型，删除 text encoder
        - 'learnable_only': 直接使用 learnable_class_prototypes，删除 text encoder
        """
        if self.is_deploy:
            return

        print(f"Switching to deployment mode (prompt_cls_type={self.prompt_cls_type})...")

        # 确保处于评估模式
        self.eval()

        with torch.no_grad():
            if self.prompt_cls_type == 'learnable_only':
                # learnable_only 方案：直接使用可学习的类别原型
                # 将 learnable_class_prototypes 注册为 buffer
                self.register_buffer("cached_class_prototypes", self.learnable_class_prototypes.data.clone())

            elif self.prompt_cls_type == 'text_encoder':
                # 原始方案：缓存属性 embeddings + 场景原型
                self.text_encoder.eval()

                # 1. 缓存属性锚点 (用于 CLIP 相似度分类)
                if not hasattr(self, 'cached_attr_embs'):
                    self.cached_attr_embs = nn.ParameterDict()

                for name in ['weather', 'light', 'road']:
                    tokens = self.attr_texts[name]
                    prompt = getattr(self, f"{name}_prompt")
                    embs = self.text_encoder(tokens, context=prompt)
                    self.register_buffer(f"cached_{name}_embs", embs)

                # 2. 缓存 24 个组合原型
                final_prototypes = self._cache_prototypes_with_text_encoder()
                self.register_buffer("cached_prototypes", final_prototypes)

            elif self.prompt_cls_type == 'linear':
                # Linear 方案：直接使用可学习的 scene_prototypes
                # 将 scene_prototypes 注册为 buffer (转为不可训练)
                self.register_buffer("cached_prototypes", self.scene_prototypes.data.clone())

            else:  # linear_text
                # 混合方案：使用 text encoder 生成场景原型
                self.text_encoder.eval()
                final_prototypes = self._cache_prototypes_with_text_encoder()
                self.register_buffer("cached_prototypes", final_prototypes)

        # 删除 Text Encoder (linear 模式也删除，虽然它本来就不用)
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None

        # 标记为部署模式
        self.is_deploy = True
        print("Model is ready for inference.")

    def _cache_prototypes_with_text_encoder(self):
        """使用 text encoder 缓存 24 个场景原型。"""
        if not hasattr(self, 'all_combinations'):
            self.all_combinations = list(itertools.product(
                self.WEATHER_CLASSES, self.LIGHT_CLASSES, self.ROAD_CLASSES
            ))

        prompts = [f"A {w} scene during {l} on a {r}, " for w, l, r in self.all_combinations]
        tokens = tokenize(prompts, context_length=self.context_length).to(next(self.parameters()).device)

        trav_embs = self.text_encoder(tokens, context=self.contexts_traversable)
        notrav_embs = self.text_encoder(tokens, context=self.contexts_notraversable)

        # 拼接: [24, 2, D]
        return torch.stack([trav_embs, notrav_embs], dim=1)

    def _generate_dynamic_text_embeddings(self, global_feat, b_size):
        """生成动态场景感知文本嵌入。

        根据图像特征预测天气/光照/路面属性，构建动态提示。
        支持三种 prompt_cls_type:
        - 'text_encoder': 使用 CLIP text encoder 编码属性文本进行分类 + 生成场景原型
        - 'linear': 使用 3 个 linear 分类器 + 24 个可学习场景向量
        - 'linear_text': 使用 3 个 linear 分类器 + text encoder 生成场景原型

        支持 'hard' (argmax) 和 'soft' (weighted sum) 两种模式。
        支持 deploy 模式加速。

        Args:
            global_feat: 全局图像特征 [B, 1, C]
            b_size: 批次大小

        Returns:
            tuple: (文本嵌入, 更新后的全局特征)
        """
        # ==================== 部署模式 (极速推理) ====================
        if self.is_deploy:
            return self._generate_dynamic_text_embeddings_deploy(global_feat, b_size)

        # ==================== 训练/普通推理模式 ====================
        # 根据 prompt_cls_type 选择不同的分类和原型生成路径
        if self.prompt_cls_type == 'text_encoder':
            return self._generate_text_encoder_embeddings(global_feat, b_size)
        elif self.prompt_cls_type == 'linear':
            return self._generate_linear_embeddings(global_feat, b_size)
        else:  # linear_text
            return self._generate_linear_text_embeddings(global_feat, b_size)

    def _generate_dynamic_text_embeddings_deploy(self, global_feat, b_size):
        """部署模式下的动态文本嵌入生成（使用缓存）。"""
        image_feature = F.normalize(global_feat.squeeze(1), dim=-1, p=2)
        attr_probs = {}
        attr_embeddings_dummy = {}
        attr_logits_dummy = {}

        # 构建属性名列表
        attr_names = ['weather', 'light', 'road']
        if self.use_material_cls:
            attr_names.append('material')

        if self.prompt_cls_type == 'text_encoder':
            # 使用缓存的 text encoder embeddings
            for name in attr_names:
                embs = getattr(self, f"cached_{name}_embs")
                attr_embeddings_dummy[name] = embs
                logits = torch.einsum('bc,nc->bn', image_feature, F.normalize(embs, dim=-1, p=2))
                attr_logits_dummy[name] = logits
                attr_probs[name] = F.softmax(logits / (self.tau if self.prompt_cls_temperature_mode == 'tau' else self.ot_score_prior_temperature), dim=1)
        else:
            # linear / linear_text: 使用可学习类别原型 (余弦相似度)
            prototypes = {
                'weather': self.weather_prototypes,
                'light': self.light_prototypes,
                'road': self.road_prototypes,
            }
            if self.use_material_cls:
                prototypes['material'] = self.material_prototypes
            for name, proto in prototypes.items():
                proto_norm = F.normalize(proto, dim=-1, p=2)
                logits = torch.matmul(image_feature, proto_norm.T)
                attr_logits_dummy[name] = logits
                attr_embeddings_dummy[name] = None  # linear 模式没有属性 embedding
                attr_probs[name] = F.softmax(logits / (self.tau if self.prompt_cls_temperature_mode == 'tau' else self.ot_score_prior_temperature), dim=1)

        # 计算联合概率
        if self.use_material_cls:
            joint_probs = torch.einsum(
                'bw,bl,br,bm->bwlrm',
                attr_probs['weather'], attr_probs['light'], attr_probs['road'], attr_probs['material']
            ).reshape(b_size, -1)
        else:
            joint_probs = torch.einsum(
                'bw,bl,br->bwlr',
                attr_probs['weather'], attr_probs['light'], attr_probs['road']
            ).reshape(b_size, -1)

        # 应用 TopK 过滤 (训练/测试使用不同的 K 值)
        joint_probs = self._apply_topk_to_joint_probs(joint_probs)

        # 使用缓存的场景原型
        text_embeddings = torch.einsum('bk,kcd->bcd', joint_probs, self.cached_prototypes)

        return text_embeddings, (global_feat, attr_embeddings_dummy, attr_logits_dummy)

    def _generate_text_encoder_embeddings(self, global_feat, b_size):
        """使用 text encoder 的原始方案。"""
        # 属性配置
        attr_configs = [
            ('weather', self.weather_prompt, self.WEATHER_CLASSES),
            ('light', self.light_prompt, self.LIGHT_CLASSES),
            ('road', self.road_prompt, self.ROAD_CLASSES),
        ]
        if self.use_material_cls:
            attr_configs.append(('material', self.material_prompt, self.MATERIAL_CLASSES))

        image_feature = F.normalize(global_feat.squeeze(1), dim=-1, p=2)
        attr_embeddings = {}
        attr_logits = {}
        attr_preds = {}
        attr_probs = {}

        # 1. 属性分类 (使用 CLIP 余弦相似度)
        for name, prompt, classes in attr_configs:
            emb = self.text_encoder(
                self.attr_texts[name], context=prompt
            ).expand(b_size, -1, -1)
            attr_embeddings[name] = emb

            emb_norm = F.normalize(emb, dim=-1, p=2)
            logits = torch.einsum('bc,bnc->bn', image_feature, emb_norm)
            attr_logits[name] = logits

            if self.prompt_cls_mode == 'hard':
                attr_preds[name] = logits.argmax(dim=1)
            else:
                attr_probs[name] = F.softmax(logits / (self.tau if self.prompt_cls_temperature_mode == 'tau' else self.ot_score_prior_temperature), dim=1)

        # 2. 生成文本嵌入
        text_embeddings = self._generate_prototypes_with_text_encoder(
            global_feat, b_size, attr_preds, attr_probs
        )

        return text_embeddings, (global_feat, attr_embeddings, attr_logits)

    def _generate_linear_embeddings(self, global_feat, b_size):
        """使用可学习类别原型 + 可学习场景向量的消融方案。

        使用余弦相似度进行分类，与 text_encoder 模式保持一致，可复用 /self.tau。
        """
        # 归一化图像特征 (与 text_encoder 模式一致)
        image_feature = F.normalize(global_feat.squeeze(1), dim=-1, p=2)  # [B, C]
        attr_logits = {}
        attr_preds = {}
        attr_probs = {}

        # 1. 使用可学习类别原型进行属性分类 (余弦相似度)
        prototypes = {
            'weather': self.weather_prototypes,  # [4, C]
            'light': self.light_prototypes,      # [3, C]
            'road': self.road_prototypes,        # [2, C]
        }
        if self.use_material_cls:
            prototypes['material'] = self.material_prototypes  # [10, C]

        for name, proto in prototypes.items():
            # 归一化类别原型
            proto_norm = F.normalize(proto, dim=-1, p=2)
            # 余弦相似度: [B, C] x [C, num_classes] -> [B, num_classes]
            logits = torch.matmul(image_feature, proto_norm.T)
            attr_logits[name] = logits

            if self.prompt_cls_mode == 'hard':
                attr_preds[name] = logits.argmax(dim=1)
            else:
                # 复用 tau 温度系数，与 text_encoder 模式一致
                attr_probs[name] = F.softmax(logits / (self.tau if self.prompt_cls_temperature_mode == 'tau' else self.ot_score_prior_temperature), dim=1)

        # 2. 使用可学习的场景原型
        if self.prompt_cls_mode == 'hard':
            # Hard mode: 选择对应的场景原型
            if self.use_material_cls:
                # 计算场景索引: w * 60 + l * 20 + r * 10 + m (240 种组合)
                scene_indices = (
                    attr_preds['weather'] * 60 +
                    attr_preds['light'] * 20 +
                    attr_preds['road'] * 10 +
                    attr_preds['material']
                )  # [B]
            else:
                # 计算场景索引: w * 6 + l * 2 + r (24 种组合)
                scene_indices = (
                    attr_preds['weather'] * 6 +
                    attr_preds['light'] * 2 +
                    attr_preds['road']
                )  # [B]
            text_embeddings = self.scene_prototypes[scene_indices]  # [B, 2, D]
        else:
            # Soft mode: 加权求和
            if self.use_material_cls:
                joint_probs = torch.einsum(
                    'bw,bl,br,bm->bwlrm',
                    attr_probs['weather'], attr_probs['light'], attr_probs['road'], attr_probs['material']
                ).reshape(b_size, -1)  # [B, 240]
            else:
                joint_probs = torch.einsum(
                    'bw,bl,br->bwlr',
                    attr_probs['weather'], attr_probs['light'], attr_probs['road']
                ).reshape(b_size, -1)  # [B, 24]

            # 应用 TopK 过滤 (训练/测试使用不同的 K 值)
            joint_probs = self._apply_topk_to_joint_probs(joint_probs)

            text_embeddings = torch.einsum('bk,kcd->bcd', joint_probs, self.scene_prototypes)

        # 注意：linear 模式没有 attr_embeddings，用 None 填充
        attr_names = ['weather', 'light', 'road']
        if self.use_material_cls:
            attr_names.append('material')
        attr_embeddings = {name: None for name in attr_names}

        return text_embeddings, (global_feat, attr_embeddings, attr_logits)

    def _generate_linear_text_embeddings(self, global_feat, b_size):
        """使用可学习类别原型 + text encoder 原型的混合方案。

        分类使用余弦相似度，场景原型由 text encoder 生成。
        """
        # 归一化图像特征 (与 text_encoder 模式一致)
        image_feature = F.normalize(global_feat.squeeze(1), dim=-1, p=2)  # [B, C]
        attr_logits = {}
        attr_preds = {}
        attr_probs = {}

        # 1. 使用可学习类别原型进行属性分类 (余弦相似度)
        prototypes = {
            'weather': self.weather_prototypes,  # [4, C]
            'light': self.light_prototypes,      # [3, C]
            'road': self.road_prototypes,        # [2, C]
        }
        if self.use_material_cls:
            prototypes['material'] = self.material_prototypes  # [10, C]

        for name, proto in prototypes.items():
            # 归一化类别原型
            proto_norm = F.normalize(proto, dim=-1, p=2)
            # 余弦相似度
            logits = torch.matmul(image_feature, proto_norm.T)
            attr_logits[name] = logits

            if self.prompt_cls_mode == 'hard':
                attr_preds[name] = logits.argmax(dim=1)
            else:
                # 复用 tau 温度系数
                attr_probs[name] = F.softmax(logits / (self.tau if self.prompt_cls_temperature_mode == 'tau' else self.ot_score_prior_temperature), dim=1)

        # 2. 使用 text encoder 生成场景原型
        text_embeddings = self._generate_prototypes_with_text_encoder(
            global_feat, b_size, attr_preds, attr_probs
        )

        # 注意：linear_text 模式没有 attr_embeddings，用 None 填充
        attr_names = ['weather', 'light', 'road']
        if self.use_material_cls:
            attr_names.append('material')
        attr_embeddings = {name: None for name in attr_names}

        return text_embeddings, (global_feat, attr_embeddings, attr_logits)

    def _generate_prototypes_with_text_encoder(self, global_feat, b_size, attr_preds, attr_probs):
        """使用 text encoder 生成场景原型（text_encoder 和 linear_text 模式共用）。"""
        if self.prompt_cls_mode == 'hard':
            # Hard Mode: 为每个样本生成独立的 prompt
            tokenized_prompts = []
            for i in range(b_size):
                if self.use_material_cls:
                    dynamic_prefix = (
                        f"A {self.WEATHER_CLASSES[attr_preds['weather'][i]]} scene "
                        f"during {self.LIGHT_CLASSES[attr_preds['light'][i]]} "
                        f"on a {self.ROAD_CLASSES[attr_preds['road'][i]]} "
                        f"with {self.MATERIAL_CLASSES[attr_preds['material'][i]]} surface, "
                    )
                else:
                    dynamic_prefix = (
                        f"A {self.WEATHER_CLASSES[attr_preds['weather'][i]]} scene "
                        f"during {self.LIGHT_CLASSES[attr_preds['light'][i]]} "
                        f"on a {self.ROAD_CLASSES[attr_preds['road'][i]]}, "
                    )

                trav_token = tokenize(
                    dynamic_prefix, context_length=self.context_length
                ).to(global_feat.device)
                trav_emb = self.text_encoder(trav_token, context=self.contexts_traversable)

                notrav_token = tokenize(
                    dynamic_prefix, context_length=self.context_length
                ).to(global_feat.device)
                notrav_emb = self.text_encoder(notrav_token, context=self.contexts_notraversable)

                tokenized_prompts.append(torch.cat([trav_emb, notrav_emb], dim=0))
            return torch.stack(tokenized_prompts, dim=0)
        else:
            # Soft Mode: 加权求和场景原型 (24 或 240 个)
            # 根据 use_material_cls 决定组合方式
            cache_key = 'all_combinations_with_material' if self.use_material_cls else 'all_combinations'
            tokens_key = 'all_combinations_tokens_with_material' if self.use_material_cls else 'all_combinations_tokens'

            if not hasattr(self, cache_key):
                if self.use_material_cls:
                    combinations = list(itertools.product(
                        self.WEATHER_CLASSES, self.LIGHT_CLASSES, self.ROAD_CLASSES, self.MATERIAL_CLASSES
                    ))
                else:
                    combinations = list(itertools.product(
                        self.WEATHER_CLASSES, self.LIGHT_CLASSES, self.ROAD_CLASSES
                    ))
                setattr(self, cache_key, combinations)

            combinations = getattr(self, cache_key)

            # 计算联合概率
            if self.use_material_cls:
                joint_probs = torch.einsum(
                    'bw,bl,br,bm->bwlrm',
                    attr_probs['weather'], attr_probs['light'], attr_probs['road'], attr_probs['material']
                ).reshape(b_size, -1)
            else:
                joint_probs = torch.einsum(
                    'bw,bl,br->bwlr',
                    attr_probs['weather'], attr_probs['light'], attr_probs['road']
                ).reshape(b_size, -1)

            # 应用 TopK 过滤 (训练/测试使用不同的 K 值)
            joint_probs = self._apply_topk_to_joint_probs(joint_probs)

            # 缓存 tokenized tokens
            if not hasattr(self, tokens_key):
                if self.use_material_cls:
                    prompts = [
                        f"A {w} scene during {l} on a {r} with {m} surface, "
                        for w, l, r, m in combinations
                    ]
                else:
                    prompts = [f"A {w} scene during {l} on a {r}, " for w, l, r in combinations]
                tokens = tokenize(prompts, context_length=self.context_length).to(global_feat.device)
                setattr(self, tokens_key, tokens)

            tokens = getattr(self, tokens_key)
            if tokens.device != global_feat.device:
                tokens = tokens.to(global_feat.device)
                setattr(self, tokens_key, tokens)

            prototypes_trav = self.text_encoder(tokens, context=self.contexts_traversable)
            prototypes_notrav = self.text_encoder(tokens, context=self.contexts_notraversable)

            final_trav_emb = torch.matmul(joint_probs, prototypes_trav)
            final_notrav_emb = torch.matmul(joint_probs, prototypes_notrav)

            return torch.stack([final_trav_emb, final_notrav_emb], dim=1)

    def _is_primary_rank(self):
        return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

    # =========================================================================
    # 可视化方法
    # =========================================================================

    def save_img_sne_merge(self, img, sne, img_metas, masks, info=None):
        """保存 RGB-SNE 融合可视化到 SwanLab。
        
        Args:
            img: RGB 图像
            sne: Surface Normal 图像
            img_metas: 图像元数据
            masks: 分割掩码
            info: 场景属性信息 (可选)
        """
        # 构建场景描述
        scene = ""
        if info is not None:
            scene = (
                f", weather: {self.WEATHER_CLASSES[info['weather_true'][0]]}"
                f"->{self.WEATHER_CLASSES[info['weather_pred'][0]]}, "
                f"road: {self.ROAD_CLASSES[info['road_true'][0]]}"
                f"->{self.ROAD_CLASSES[info['road_pred'][0]]}, "
                f"light: {self.LIGHT_CLASSES[info['light_true'][0]]}"
                f"->{self.LIGHT_CLASSES[info['light_pred'][0]]}"
            )

        # 准备张量
        im = img[0].detach().float().cpu()
        sn = sne[0].detach().float().cpu()
        ma = masks[0][0:1].detach().float().cpu() if masks is not None else torch.ones_like(im)

        def _normalize_to_01(x):
            """将张量归一化到 [0, 1] 范围。"""
            x = x.clone()
            x -= x.min()
            x /= (x.max() - x.min() + 1e-6)
            if x.size(0) == 1:
                x = x.repeat(3, 1, 1)
            return x.clamp(0, 1)

        im01 = _normalize_to_01(im)
        sn01 = _normalize_to_01(sn)
        ma01 = _normalize_to_01(ma)
        overlay = (0.6 * im01 + 0.4 * sn01).clamp(0, 1)

        # 拼接可视化
        row1 = torch.cat([im01, sn01], dim=2)
        row2 = torch.cat([overlay, ma01], dim=2)
        concat = torch.cat([row1, row2], dim=1)

        try:
            image_numpy = (concat * 255).byte().permute(1, 2, 0).numpy()
            base_name = os.path.basename(
                img_metas[0].get('ori_filename', f'{self.save_img_sne_sum}.png')
            )
            caption = f"img|sne|overlay|mask for {base_name}{scene}"
            image_to_log = swanlab.Image(image_numpy, caption=caption)

            if self.training:
                self.save_img_sne_sum += 1
                swanlab.log({"Train Samples": image_to_log})
            else:
                self.save_img_sne_sum_val += 1
                swanlab.log({"Val Samples": image_to_log})
        except Exception:
            pass

    # =========================================================================
    # 训练方法
    # =========================================================================

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        if self.force_reg_e0_eval:
            self.reg_E0.eval()
            self.text_encoder.eval()
            if not self._reg_e0_eval_logged:
                print("Reg_E0 set to eval mode during training.", self.reg_E0.training)
                self._reg_e0_eval_logged = True
        """训练前向传播。"""
        losses = dict()

        # 特征提取 (RGB + SNE)
        sne = kwargs.get('sne')
        masks = kwargs.get('gt_masks')

        x, sne_feature = self.extract_feat(img, sne)

        # 保存可视化样本
        if self.use_sne and sne is not None:
            if not hasattr(self, 'save_img_sne_sum'):
                self.save_img_sne_sum = 0
            if self.save_img_sne_sum < 10:
                self.save_img_sne_merge(img, sne, img_metas, masks)

        # 后处理特征
        x_orig, score_map_all, text_emb, global_feat, score = self.after_extract_feat(
            x, sne_feature
        )

        # 分离 RGB 和 SNE 特征
        if isinstance(x_orig, tuple):
            x_orig, sne_orig = x_orig
        else:
            sne_orig = None

        # ======== SNE 特征融合 (根据 fusion_stage) ========
        sne_for_decode = None  # 默认不传 SNE 给 decode_head
        pi_maps = None

        if self.use_sne and sne_orig is not None:
            if self.sne_fusion_stage == 'backbone':
                # backbone 阶段融合: 在 segmentor 中完成融合
                # 传入 score_map 用于 OT 融合的文本分布先验
                collect_pi = self.supervise_ot_pi and self.sne_fusion_mode == 'ot'
                x_orig, fusion_losses, pi_maps = self._fuse_sne_features(
                    x_orig,
                    sne_orig,
                    text_emb,
                    score_map=score_map_all,
                    collect_pi=collect_pi,
                    gt_semantic_seg=gt_semantic_seg,
                    img_metas=img_metas,
                    identity_head=self.identity_head,
                    train_cfg=self.train_cfg,
                    supervise_ot_pi=self.supervise_ot_pi,
                )
                losses.update(fusion_losses)
                # 融合已完成，sne_for_decode 保持为 None
            else:
                # pixel 阶段融合: 传 SNE 给 decode_head
                sne_for_decode = sne_orig

        # ======== 全局 OT 对齐 (可选) ========
        if self.use_ot_align and self.use_sne:
            losses.update(self._compute_global_ot_loss(score, text_emb))

        # Neck 处理
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        # 如果是 pixel 阶段融合，也要对 SNE 过 Neck
        if sne_for_decode is not None and self.neck is not None:
            sne_for_decode = list(self.neck(sne_for_decode))

        # ======== 场景属性损失 ========
        # learnable_only 模式没有场景属性分类，跳过
        if self.prompt_cls and self.prompt_cls_type != 'learnable_only':
            global_feat, losses_attr = self._compute_attribute_losses(global_feat, img_metas)
            losses.update(losses_attr)

        # ======== 正则化损失 ========
        losses.update(self._compute_regularization_losses(
            text_emb, img, global_feat,
            score_map=score_map_all,
            img_metas=img_metas,
            gt_semantic_seg=gt_semantic_seg,
            pi_map=pi_maps
        ))

        # ======== Decode Head 损失 ========
        loss_decode = self.decode_head.forward_train(
            x, text_emb, img_metas, gt_semantic_seg, self.train_cfg,
            kwargs['gt_labels'], kwargs['gt_masks'],
            sne_feature=sne_for_decode
        )
        losses.update(add_prefix(loss_decode, 'decode'))

        return losses

    def _fuse_sne_features(
        self,
        x_orig,
        sne_orig,
        text_emb,
        score_map=None,
        debug_store=None,
        collect_pi=False,
        gt_semantic_seg=None,
        img_metas=None,
        identity_head=None,
        train_cfg=None,
        supervise_ot_pi=False,
    ):
        """根据 sne_fusion_mode 融合 RGB 和 SNE 特征 (backbone 阶段)。
        
        Args:
            x_orig: RGB 多尺度特征列表
            sne_orig: SNE 多尺度特征列表
            text_emb: 文本嵌入
            score_map: 相似度图字典，用于 OT 融合的文本分布先验
            
        Returns:
            tuple: (融合后的特征, 损失字典, 可选的 pi 字典)
        """
        losses = {}
        pi_maps = None

        if self.sne_fusion_mode == 'proj':
            # proj 融合：concat + 1x1 conv
            x_orig = self._proj_fusion(x_orig, sne_orig)

        elif self.sne_fusion_mode == 'add':
            # add 融合：直接相加
            x_orig = self._add_fusion(x_orig, sne_orig)

        elif self.sne_fusion_mode == 'concat':
            # concat 融合：通道拼接
            x_orig = self._concat_fusion(x_orig, sne_orig)

        elif self.sne_fusion_mode == 'cross_attn':
            # 双向交叉注意力融合
            x_orig, attn_losses = self._cross_attn_fusion(x_orig, sne_orig)
            losses.update(attn_losses)

        elif self.sne_fusion_mode == 'ot':
            # 最优传输融合 (使用 score_map 初始化文本分布)
            x_orig, ot_losses, pi_maps = self._ot_fusion(
                x_orig, sne_orig, text_emb, score_map,
                debug_store=debug_store,
                collect_pi=collect_pi,
                gt_semantic_seg=gt_semantic_seg,
                img_metas=img_metas,
                identity_head=identity_head,
                train_cfg=train_cfg,
                supervise_ot_pi=supervise_ot_pi,
            )
            losses.update(ot_losses)

        return x_orig, losses, pi_maps

    def _proj_fusion(self, x_orig, sne_orig):
        """Proj 融合: concat + 1x1 conv 降维。"""
        for i in range(4):
            fused = torch.cat([x_orig[i], sne_orig[i]], dim=1)
            x_orig[i] = self.img_sne_proj[i](fused)
        return x_orig

    def _add_fusion(self, x_orig, sne_orig):
        """Add 融合: 直接相加。"""
        for i in range(4):
            x_orig[i] = x_orig[i] + sne_orig[i]
        return x_orig

    def _concat_fusion(self, x_orig, sne_orig):
        """Concat 融合: 通道拼接 (通道数翻倍，需要后续网络支持)。"""
        for i in range(4):
            x_orig[i] = torch.cat([x_orig[i], sne_orig[i]], dim=1)
        return x_orig

    def _cross_attn_fusion(self, x_orig, sne_orig):
        """双向交叉注意力融合。"""
        losses = {}
        for i in range(4):
            b, f, h, w = x_orig[i].shape

            # 展平特征
            x_flat = x_orig[i].view(b, f, -1).permute(0, 2, 1).contiguous()
            sne_flat = sne_orig[i].view(b, f, -1).permute(0, 2, 1).contiguous()

            # 双向交叉注意力
            x_enhanced, sne_enhanced, _, _ = self.cross_attn_modules[i](x_flat, sne_flat)

            # concat + proj 融合
            fused = torch.cat([x_enhanced, sne_enhanced], dim=-1)
            fused = fused.permute(0, 2, 1).contiguous().view(b, f * 2, h, w)
            x_orig[i] = self.img_sne_proj[i](fused)

        return x_orig, losses

    def _ot_fusion(
        self,
        x_orig,
        sne_orig,
        text_emb,
        score_map=None,
        debug_store=None,
        collect_pi=False,
        gt_semantic_seg=None,
        img_metas=None,
        identity_head=None,
        train_cfg=None,
        supervise_ot_pi=False,
    ):
        """最优传输融合 (与 text 对齐)。
        
        Args:
            x_orig: RGB 多尺度特征列表
            sne_orig: SNE 多尺度特征列表  
            text_emb: 文本嵌入
            score_map: 相似度图字典，用于初始化 OT 的文本分布
                       使用 score_map 反映实际的类别比例，而非均匀分布
        """
        losses = {}
        pi_maps = {'img': [], 'sne': []} if collect_pi else None
        new_text = self.text_proj(text_emb)
        new_text_norm = F.normalize(new_text, dim=-1, p=2)
        
        # 分别获取 img 和 sne 的 score_map
        img_score_map = None
        sne_score_map = None
        if score_map is not None:
            img_score_map = score_map.get('img')  # [B, K, H, W]
            sne_score_map = score_map.get('sne')  # [B, K, H, W]

        for i in range(1,4):
            b, f, h, w = x_orig[i].shape

            # 展平特征
            old_x = x_orig[i].view(b, f, -1).permute(0, 2, 1).contiguous()
            old_sne = sne_orig[i].view(b, f, -1).permute(0, 2, 1).contiguous()
            
            # 将 score_map resize 到当前尺度 (img 和 sne 分别处理)
            scale_img_score_map = None
            scale_sne_score_map = None
            if img_score_map is not None:
                scale_img_score_map = F.interpolate(
                    img_score_map, size=(h, w), mode='bilinear', align_corners=False
                )
            if sne_score_map is not None:
                scale_sne_score_map = F.interpolate(
                    sne_score_map, size=(h, w), mode='bilinear', align_corners=False
                )

            # 可选：在概率空间做 soft union，将融合后的 score_map 同时提供给 img/sne 分支
            if self.ot_softunion and (scale_img_score_map is not None) and (scale_sne_score_map is not None):
                probs_img = F.softmax(scale_img_score_map / self.ot_score_prior_temperature, dim=1)
                probs_sne = F.softmax(scale_sne_score_map / self.ot_score_prior_temperature, dim=1)
                probs_fused = torch.max(probs_img, probs_sne)
                fused_score_map = torch.log(probs_fused.clamp(min=1e-8))
                scale_img_score_map = fused_score_map
                scale_sne_score_map = fused_score_map

            # OT 对齐 (img 和 sne 使用各自的 score_map 作为文本分布先验)
            new_x, loss_x, pi_img = self.img_attn[i](old_x, new_text, score_map=scale_img_score_map, temperature=self.ot_score_prior_temperature)
            new_sne, loss_sne, pi_sne = self.sne_attn[i](old_sne, new_text, score_map=scale_sne_score_map, temperature=self.ot_score_prior_temperature)

            # 收集调试/监督用的传输计划 (pi)
            if debug_store is not None or collect_pi:
                if debug_store is not None:
                    if 'pi_img' not in debug_store:
                        debug_store['pi_img'] = []
                    if 'pi_sne' not in debug_store:
                        debug_store['pi_sne'] = []

                # 将 pi 还原为 [B, K, H, W] 方便插值保存/监督
                pi_img_map = pi_img.view(b, h, w, -1).permute(0, 3, 1, 2)
                pi_sne_map = pi_sne.view(b, h, w, -1).permute(0, 3, 1, 2)
                # debug_store 只存 detached 版本用于可视化
                if debug_store is not None:
                    debug_store['pi_img'].append(pi_img_map.detach())
                    debug_store['pi_sne'].append(pi_sne_map.detach())
                # 监督路径保留梯度，不要 detach
                if collect_pi and pi_maps is not None:
                    pi_maps['img'].append(pi_img_map)
                    pi_maps['sne'].append(pi_sne_map)

            if self.ot_fuse_mode == 'proj':
                fused = torch.cat([new_x, new_sne], dim=-1)
                fused = fused.permute(0, 2, 1).contiguous().view(b, f * 2, h, w)
                x_orig[i] = self.img_sne_proj[i](fused)
            elif self.ot_fuse_mode == 'mean':  # mean fusion
                fused = 0.5 * (new_x + new_sne)
                fused = fused.permute(0, 2, 1).contiguous().view(b, f, h, w)
                x_orig[i] = fused
            elif self.ot_fuse_mode == 'max':
                fused = torch.max(new_x, new_sne)
                fused = fused.permute(0, 2, 1).contiguous().view(b, f, h, w)
                x_orig[i] = fused
            elif self.ot_fuse_mode == 'config':
                # 基于 OT 传输计划的置信度加权融合
                # pi_img, pi_sne: [B*H*W, K] -> reshape to [B, H*W, K]
                pi_img_reshaped = pi_img.view(b, h * w, -1)
                pi_sne_reshaped = pi_sne.view(b, h * w, -1)

                # 计算置信度: [B, H*W]
                conf_img = self._compute_confidence_from_entropy(pi_img_reshaped)
                conf_sne = self._compute_confidence_from_entropy(pi_sne_reshaped)

                # 归一化置信度作为融合权重
                conf_sum = conf_img + conf_sne + 1e-8
                weight_img = (conf_img / conf_sum).unsqueeze(-1)  # [B, H*W, 1]
                weight_sne = (conf_sne / conf_sum).unsqueeze(-1)  # [B, H*W, 1]

                # new_x, new_sne: [B, H*W, C]
                fused = weight_img * new_x + weight_sne * new_sne
                fused = fused.permute(0, 2, 1).contiguous().view(b, f, h, w)
                x_orig[i] = fused

            losses.update(add_prefix({'loss': loss_x}, f'img_ot_{i}'))
            losses.update(add_prefix({'loss': loss_sne}, f'sne_ot_{i}'))

            # 可选：对 OT 对齐后的 new_x/new_sne 与 new_text 直接做分割监督
            if supervise_ot_pi and identity_head is not None and gt_semantic_seg is not None:
                def _score_and_loss(feat_bhwc, prefix):
                    feat_map = feat_bhwc.view(b, h, w, f).permute(0, 3, 1, 2)  # B,C,H,W
                    feat_norm = F.normalize(feat_map, dim=1, p=2)
                    score = torch.einsum('bchw,bkc->bkhw', feat_norm, new_text_norm)
                    loss_map = identity_head.forward_train(
                        score / self.tau, img_metas, gt_semantic_seg, train_cfg
                    )
                    losses.update(add_prefix(loss_map, prefix))

                _score_and_loss(new_x, f'ot_feat_img_{i}')
                _score_and_loss(new_sne, f'ot_feat_sne_{i}')

        return x_orig, losses, pi_maps

    def _apply_topk_to_joint_probs(self, joint_probs):
        """
        对 joint_probs 应用 TopK 过滤，类似 dropout 的训练/测试行为差异。
        Args:
            joint_probs: [B, K] 的联合概率分布
        Returns:
            filtered_probs: [B, K] 过滤后的概率分布 (TopK 外的置为 0 并重新归一化)
        """
        # 根据训练/测试模式选择 TopK 值
        if self.training:
            topk = self.prompt_cls_topk_train
        else:
            topk = self.prompt_cls_topk_test

        # 如果 topk 为 None，保持原来的行为
        if topk is None:
            return joint_probs

        # 确保 topk 不超过类别数
        k = min(topk, joint_probs.shape[-1])

        # 获取 TopK 的值和索引
        topk_vals, topk_indices = torch.topk(joint_probs, k, dim=-1)

        # 创建 mask，只保留 TopK
        mask = torch.zeros_like(joint_probs)
        mask.scatter_(-1, topk_indices, 1.0)

        # 过滤并重新归一化
        filtered_probs = joint_probs * mask
        filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-8)

        return filtered_probs

    def _compute_confidence_from_entropy(self, pi):
        """
        计算基于信息熵的置信度。
        Args:
            pi: [B, N, K] 或 [B, K] 的任意 Logits 或 Probability
        Returns:
            confidence: [B, N] (如果输入是三维) 或 [B] (如果输入是二维)
            取值范围 0.0 (完全不确定) ~ 1.0 (非常确信)
        """
        eps = 1e-10

        # 强制归一化 (Self-Correction)
        pi_sum = pi.sum(dim=-1, keepdim=True)
        pi_norm = pi / (pi_sum + eps)

        # 计算熵 H = -sum(p * log(p))
        log_pi = torch.log(torch.clamp(pi_norm, min=eps))
        entropy = -torch.sum(pi_norm * log_pi, dim=-1)

        # 归一化熵 (Relative Entropy)
        K = pi.shape[-1]
        max_entropy = torch.log(torch.tensor(K, dtype=pi.dtype, device=pi.device))
        normalized_entropy = entropy / max_entropy

        # 转为置信度
        confidence = 1.0 - normalized_entropy
        return confidence

    def _compute_global_ot_loss(self, score, text_emb):
        """计算全局 OT 对齐损失。"""
        losses = {}
        x_score, sne_score = score
        _, x_visual = x_score
        _, sne_visual = sne_score

        # 展平视觉特征
        x_flat = x_visual.view(x_visual.size(0), x_visual.size(1), -1)
        x_flat = x_flat.permute(0, 2, 1).contiguous()
        sne_flat = sne_visual.view(sne_visual.size(0), sne_visual.size(1), -1)
        sne_flat = sne_flat.permute(0, 2, 1).contiguous()

        # OT 对齐
        _, loss_img, _ = self.opt_img(x_flat, text_emb, temperature=self.ot_score_prior_temperature)
        _, loss_sne, _ = self.opt_sne(sne_flat, text_emb, temperature=self.ot_score_prior_temperature)

        losses.update(add_prefix({'loss': loss_sne}, 'sne_opt'))
        losses.update(add_prefix({'loss': loss_img}, 'img_opt'))

        return losses

    def _get_attribute_targets(self, img_metas):
        """从元数据中获取属性标签。"""
        weather_list, light_list, road_list = [], [], []
        material_list = [] if self.use_material_cls else None

        for meta in img_metas:
            scene_name = meta['ori_filename'].split('/')[1][1:].replace('_', '-')
            info = self.scene2info[scene_name]

            weather_list.append(self.WEATHER_CLASSES.index(info['weather']))
            light_list.append(self.LIGHT_CLASSES.index(info['light']))

            # 道路格式: {paved/unpaved}_{material}_road
            road_parts = info['road'].replace('_road', '').split('_')
            road_list.append(self.ROAD_CLASSES.index(road_parts[0] + ' road'))

            # 材质是中间部分 (可能有多个下划线，如 'paved_concrete_road')
            if self.use_material_cls:
                material = '_'.join(road_parts[1:])
                material_list.append(self.MATERIAL_CLASSES.index(material))

        device = next(self.parameters()).device
        result = (
            torch.tensor(weather_list, device=device),
            torch.tensor(light_list, device=device),
            torch.tensor(road_list, device=device),
        )
        if self.use_material_cls:
            result = result + (torch.tensor(material_list, device=device),)
        return result

    def _compute_attribute_losses(self, global_feat, img_metas):
        """计算场景属性分类损失。

        根据 prompt_cls_type 计算不同的损失:
        - 'text_encoder': 属性正则化损失 + 属性分类损失
        - 'linear' / 'linear_text': 仅属性分类损失 (没有 text embedding，无法计算正则化)
        """
        losses = {}
        global_feat_tensor, attr_embeddings, attr_logits = global_feat

        # 获取真实标签
        attr_names = ['weather', 'light', 'road']
        if self.use_material_cls:
            attr_names.append('material')
        attr_targets = dict(zip(attr_names, self._get_attribute_targets(img_metas)))

        # 统一计算属性损失
        for name in attr_names:
            emb = attr_embeddings[name]
            logits = attr_logits[name]
            target = attr_targets[name]

            # 属性正则化损失 (仅 text_encoder 模式)
            # linear / linear_text 模式没有 attr_embeddings，跳过正则化
            if emb is not None and self.prompt_cls_type == 'text_encoder':
                score = torch.einsum(
                    'blc,kc->bkl', F.normalize(emb, dim=-1, p=2), self.attr_t0[name].detach()
                )
                reg_loss = F.cross_entropy(
                    score / self.tau, self.attr_I[name].expand(score.shape[0], -1), reduction='mean'
                )
                losses.update(add_prefix({'loss': reg_loss}, f'reg.{name}'))

            # 属性分类损失 (所有模式都计算)
            cls_loss = F.cross_entropy(logits / self.tau, target, reduction='mean')
            losses.update(add_prefix({'loss': cls_loss}, f'attr.{name}'))

        # 计算准确率
        with torch.no_grad():
            total = attr_targets['weather'].size(0)
            self.acc_weather = (attr_logits['weather'].argmax(1) == attr_targets['weather']).sum().float() / total
            self.acc_light = (attr_logits['light'].argmax(1) == attr_targets['light']).sum().float() / total
            self.acc_road = (attr_logits['road'].argmax(1) == attr_targets['road']).sum().float() / total
            if self.use_material_cls:
                self.acc_material = (attr_logits['material'].argmax(1) == attr_targets['material']).sum().float() / total

        return global_feat_tensor, losses

    def _compute_regularization_losses(self, text_emb, img, global_feat, score_map=None, img_metas=None, gt_semantic_seg=None, pi_map=None):
        """计算正则化损失。

        Args:
            text_emb: 文本嵌入
            img: 输入图像
            global_feat: 全局特征
            score_map: 相似度图字典 (可选，用于 vision-language reg)
            img_metas: 图像元数据 (可选)
            gt_semantic_seg: 真实分割标签 (可选)
        """
        losses = {}

        # Vision-Language 正则化 (基于 score_map)
        if self.use_score_map_reg and self.identity_head is not None and score_map is not None:
            # RGB score map 正则化
            if 'img' in score_map:
                loss_score_map = self.identity_head.forward_train(
                    score_map['img'] / self.tau, img_metas, gt_semantic_seg, self.train_cfg
                )
                losses.update(add_prefix(loss_score_map, 'scr_map'))

            # SNE score map 正则化
            if 'sne' in score_map:
                loss_score_map_sne = self.identity_head.forward_train(
                    score_map['sne'] / self.tau, img_metas, gt_semantic_seg, self.train_cfg
                )
                losses.update(add_prefix(loss_score_map_sne, 'sne_map'))

        # OT 传输计划深监督 (可选)
        if self.supervise_ot_pi and self.identity_head is not None and pi_map is not None:
            def _select_highres(tlist):
                """Pick the 2nd-highest-resolution map to keep pyramid consistency."""
                if isinstance(tlist, (list, tuple)):
                    if len(tlist) >= 2:
                        return tlist[-2]  # use penultimate level (source for up/down sampling)
                    if len(tlist) == 1:
                        return tlist[0]
                return tlist

            if 'img' in pi_map and pi_map['img']:
                pi_img = _select_highres(pi_map['img'])
                loss_pi_img = self.identity_head.forward_train(
                    pi_img / self.tau, img_metas, gt_semantic_seg, self.train_cfg
                )
                losses.update(add_prefix(loss_pi_img, 'pi_img'))

            if 'sne' in pi_map and pi_map['sne']:
                pi_sne = _select_highres(pi_map['sne'])
                loss_pi_sne = self.identity_head.forward_train(
                    pi_sne / self.tau, img_metas, gt_semantic_seg, self.train_cfg
                )
                losses.update(add_prefix(loss_pi_sne, 'pi_sne'))

        # learnable_only 模式：跳过所有正则化损失 (无文本编码器可用于对齐)
        if self.prompt_cls and self.prompt_cls_type == 'learnable_only':
            return losses

        # 提取真实的 global_feat tensor
        if isinstance(global_feat, tuple):
            global_feat = global_feat[0]

        # 文本正则化
        if self.textual_reg:
            content_score = torch.einsum(
                'blc,kc->bkl', F.normalize(text_emb, dim=-1, p=2), self.reg_T0.detach()
            )
            loss = F.cross_entropy(
                content_score / self.tau,
                self.reg_I.expand(content_score.shape[0], -1),
                reduction='mean'
            )
            losses.update(add_prefix({'loss': loss}, 'reg.textual'))

        # 视觉正则化
        if self.visual_reg:
            with torch.no_grad():
                global_feat_0, _ = self.reg_E0.extract_feats(img)[-1]
            loss = nn.MSELoss(reduction='mean')(global_feat, global_feat_0)
            losses.update(add_prefix({'loss': loss}, 'reg.visual'))

        

        return losses

    # =========================================================================
    # 推理方法
    # =========================================================================

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with optional attention/debug outputs.

        Avoid让 return_attn/return_debug 透传到基类 whole_inference，直接使用 encode_decode
        以便返回 tuple 时不再被 resize 误处理。
        """
        return_attn = kwargs.pop('return_attn', self.test_cfg.get('return_attn', False))
        return_debug = kwargs.pop('return_debug', False)

        out = self.encode_decode(
            img, img_meta,
            return_attn=return_attn,
            return_debug=return_debug,
            **kwargs
        )

        attn = None
        debug_payload = None
        if return_attn and return_debug:
            seg_logit, attn, debug_payload = out
        elif return_attn:
            seg_logit, attn = out
        elif return_debug:
            seg_logit, debug_payload = out
        else:
            seg_logit = out

        seg_pred = seg_logit.argmax(dim=1)
        if self.save_seg_logit is True:
            self.seg_logit = seg_logit.cpu().numpy()
        if torch.onnx.is_in_onnx_export():
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)

        if return_attn and return_debug:
            return seg_pred, attn, debug_payload
        if return_attn:
            return seg_pred, attn
        if return_debug:
            return seg_pred, debug_payload
        return seg_pred

    def encode_decode(self, img, img_metas, return_attn=False, return_debug=False, debug_target_size=224, **kwargs):
        """编码解码推理，可选返回中间结果用于可视化/调试。"""
        # 特征提取 (RGB + SNE)
        sne = kwargs.get('sne')
        masks = kwargs.get('gt_masks')

        x, sne_feature = self.extract_feat(img, sne)

        x_orig, score_map_all, text_emb, global_feat, _ = self.after_extract_feat(
            x, sne_feature
        )

        # 调试容器：仅在需要保存中间结果时初始化
        debug_store = {'pi_img': [], 'pi_sne': []} if return_debug else None

        # 验证时计算属性准确率 (learnable_only 模式没有属性分类，跳过)
        if self.prompt_cls and self.prompt_cls_type != 'learnable_only':
            self._compute_val_attribute_accuracy(global_feat, img_metas, img, sne, masks)

        # 分离特征
        if isinstance(x_orig, tuple):
            x_orig, sne_orig = x_orig
        else:
            sne_orig = None

        # ======== SNE 特征融合 (根据 fusion_stage) ========
        sne_for_decode = None  # 默认不传 SNE 给 decode_head

        if self.use_sne and sne_orig is not None:
            if self.sne_fusion_stage == 'backbone':
                # backbone 阶段融合 (传入 score_map 用于 OT 融合)
                x_orig, _, _ = self._fuse_sne_features(
                    x_orig,
                    sne_orig,
                    text_emb,
                    score_map=score_map_all,
                    debug_store=debug_store,
                    collect_pi=False,
                    gt_semantic_seg=None,
                    img_metas=None,
                    identity_head=None,
                    train_cfg=None,
                    supervise_ot_pi=False,
                )
            else:
                # pixel 阶段融合: 传 SNE 给 decode_head
                sne_for_decode = sne_orig

        # Neck 处理
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        # 如果是 pixel 阶段融合，也要对 SNE 过 Neck
        if sne_for_decode is not None and self.neck is not None:
            sne_for_decode = list(self.neck(sne_for_decode))

        # Decode head 推理
        out = self.decode_head.forward_test(
            x, text_emb, img_metas, self.test_cfg,
            return_attn=return_attn,
            sne_feature=sne_for_decode
        )

        if return_attn:
            out, attn = out

        # 将输出恢复到原始图像尺寸以匹配 GT
        ori_size = img_metas[0]['ori_shape'][:2]
        out = resize(
            input=out,
            size=ori_size,
            mode='bilinear',
            align_corners=False
        )

        debug_payload = None
        if return_debug:
            target_hw = (debug_target_size, debug_target_size) if isinstance(debug_target_size, int) else debug_target_size
            debug_payload = {}

            # score map 插值到统一尺寸
            if score_map_all is not None:
                if 'img' in score_map_all:
                    debug_payload['score_map_img'] = F.interpolate(
                        score_map_all['img'], size=target_hw, mode='bilinear', align_corners=False
                    ).detach().cpu()
                if 'sne' in score_map_all:
                    debug_payload['score_map_sne'] = F.interpolate(
                        score_map_all['sne'], size=target_hw, mode='bilinear', align_corners=False
                    ).detach().cpu()

            # OT 传输计划 pi 列表插值保存，仅在 OT 融合时可用
            def _resize_pi_list(pi_list):
                if pi_list is None:
                    return []
                resized = []
                for pi_map in pi_list:
                    # 仅保留前 2 类，形状 [B, 2, H, W]
                    if pi_map.shape[1] > 2:
                        pi_map = pi_map[:, :2]
                    resized.append(
                        F.interpolate(pi_map, size=target_hw, mode='bilinear', align_corners=False).detach().cpu()
                    )
                return resized

            if debug_store is not None:
                debug_payload['pi_img_levels'] = _resize_pi_list(debug_store.get('pi_img'))
                debug_payload['pi_sne_levels'] = _resize_pi_list(debug_store.get('pi_sne'))

        if return_attn and return_debug:
            return out, attn, debug_payload
        if return_attn:
            return out, attn
        if return_debug:
            return out, debug_payload
        return out

    def _compute_val_attribute_accuracy(self, global_feat, img_metas, img, sne, masks):
        """验证时计算属性准确率。"""
        _, _, attr_logits = global_feat

        # 构建属性名列表
        attr_names = ['weather', 'light', 'road']
        if self.use_material_cls:
            attr_names.append('material')

        # 获取真实标签
        attr_targets = dict(zip(attr_names, self._get_attribute_targets(img_metas)))

        with torch.no_grad():
            total = attr_targets['weather'].size(0)
            attr_preds = {name: attr_logits[name].argmax(dim=1) for name in attr_names}

            self.val_acc_weather = (attr_preds['weather'] == attr_targets['weather']).sum().float() / total
            self.val_acc_light = (attr_preds['light'] == attr_targets['light']).sum().float() / total
            self.val_acc_road = (attr_preds['road'] == attr_targets['road']).sum().float() / total
            if self.use_material_cls:
                self.val_acc_material = (attr_preds['material'] == attr_targets['material']).sum().float() / total

            # 保存验证样本可视化
            if self.save_img_sne_sum_val < 10:
                info = {
                    f'{name}_true': attr_targets[name].tolist()
                    for name in attr_names
                }
                info.update({
                    f'{name}_pred': attr_preds[name].tolist()
                    for name in attr_names
                })
                self.save_img_sne_merge(img, sne, img_metas, masks, info)
