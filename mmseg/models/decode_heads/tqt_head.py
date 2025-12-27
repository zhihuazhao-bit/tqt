"""
TQT Decode Head - 基于 tqdmHead 的 RGB-SNE 双模态融合扩展

设计原则:
- forward() 主体逻辑与 tqdmHead 完全一致
- SNE 融合作为增量功能，仅在 sne_feature 不为 None 时生效

融合位置 (sne_fusion_stage):
- 'backbone': 融合在 tqt_eva_clip (segmentor) 中完成，head 收到的是融合后特征 (sne_feature=None)
- 'pixel': 融合在 head 中完成，使用独立的 pixel_decoder 并行处理 RGB 和 SNE

融合方式 (sne_fusion_mode):
- 'proj': concat + 1x1 conv 降维
- 'add': 直接相加
- 'concat': 通道拼接 (通道数翻倍，需配置正确的 decoder_dim)
- 'cross_attn': 双向交叉注意力 (仅 pixel 阶段)
- 'ot': 最优传输对齐 (仅 pixel 阶段)

使用场景:
- sne_feature=None: 与 tqdmHead 行为完全一致 (backbone 融合或不使用 SNE)
- sne_feature 有值: 在 head 中进行 pixel 阶段融合
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.tqdm_head import tqdmHead


@HEADS.register_module()
class tqtHead(tqdmHead):
    """TQT Decode Head: 支持 RGB-SNE 双模态融合的解码头。
    
    融合位置由 segmentor 控制:
    - backbone 融合: segmentor 完成融合后传 sne_feature=None
    - pixel 融合: segmentor 传 sne_feature，head 使用独立 pixel_decoder 并行处理后融合
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 text_proj=None,
                 # SNE pixel 阶段融合配置 (仅当 use_sne_pixel=True 时生效)
                 use_sne_pixel=False,     # 是否启用 pixel 阶段 SNE 融合
                 sne_fusion_mode='proj',  # 'proj' / 'add' / 'cross_attn' / 'ot'
                 **kwargs):
        super(tqtHead, self).__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            num_queries=num_queries,
            num_transformer_feat_level=num_transformer_feat_level,
            pixel_decoder=pixel_decoder,
            enforce_decoder_input_project=enforce_decoder_input_project,
            transformer_decoder=transformer_decoder,
            positional_encoding=positional_encoding,
            loss_cls=loss_cls,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            text_proj=text_proj,
            **kwargs
        )

        # SNE pixel 阶段融合配置
        self.use_sne_pixel = use_sne_pixel
        self.sne_fusion_mode = sne_fusion_mode
        assert sne_fusion_mode in ['proj', 'add', 'concat', 'cross_attn', 'ot']

        # 仅当 use_sne_pixel=True 时才初始化 SNE pixel 融合模块
        if use_sne_pixel:
            self._init_sne_pixel_modules(feat_channels, out_channels)
            print(f"[TQT Head] SNE pixel fusion enabled, mode={sne_fusion_mode}")
        else:
            # 不需要 pixel 阶段融合，不创建额外模块
            self.pixel_decoder_sne = None
            self.sne_mask_proj = None
            self.sne_memory_projs = None
            print(f"[TQT Head] SNE pixel fusion disabled (use_sne_pixel=False)")

    def _init_sne_pixel_modules(self, feat_channels, out_channels):
        """初始化 SNE pixel 阶段融合模块 (独立 pixel_decoder + 融合层)。"""
        # 独立的 SNE pixel_decoder (并行处理)
        self.pixel_decoder_sne = copy.deepcopy(self.pixel_decoder)

        # 根据融合方式初始化不同模块
        self.sne_mask_proj = None
        self.sne_memory_projs = None
        self.sne_cross_attn = None

        if self.sne_fusion_mode == 'proj':
            # concat + 1x1 conv 降维
            self.sne_mask_proj = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
            self.sne_memory_projs = nn.ModuleList([
                nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1)
                for _ in range(self.num_transformer_feat_level)
            ])

        elif self.sne_fusion_mode == 'cross_attn':
            # 双向交叉注意力
            self.sne_mask_cross_attn = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
            self.sne_mask_norm = nn.LayerNorm(out_channels)
            self.sne_memory_cross_attns = nn.ModuleList([
                nn.MultiheadAttention(feat_channels, num_heads=8, batch_first=True)
                for _ in range(self.num_transformer_feat_level)
            ])
            self.sne_memory_norms = nn.ModuleList([
                nn.LayerNorm(feat_channels)
                for _ in range(self.num_transformer_feat_level)
            ])

        # 'add' 和 'ot' 模式不需要额外模块

    def init_weights(self):
        super().init_weights()
        if self.pixel_decoder_sne is not None:
            self.pixel_decoder_sne.init_weights()

    # =========================================================================
    # SNE Pixel 阶段融合 (增量逻辑)
    # =========================================================================

    def _fuse_pixel_outputs(self, mask_rgb, memory_rgb, mask_sne, memory_sne):
        """融合 RGB 和 SNE 的 pixel_decoder 输出。
        
        Args:
            mask_rgb: RGB mask features [B, C, H, W]
            memory_rgb: RGB 多尺度 memory list
            mask_sne: SNE mask features [B, C, H, W]
            memory_sne: SNE 多尺度 memory list
            
        Returns:
            tuple: (融合后的 mask_features, 融合后的 multi_scale_memorys)
        """
        if self.sne_fusion_mode == 'add':
            # 直接相加
            mask_fused = mask_rgb + mask_sne
            memory_fused = [m_r + m_s for m_r, m_s in zip(memory_rgb, memory_sne)]

        elif self.sne_fusion_mode == 'proj':
            # concat + projection
            mask_fused = self.sne_mask_proj(torch.cat([mask_rgb, mask_sne], dim=1))
            memory_fused = [
                proj(torch.cat([m_r, m_s], dim=1))
                for proj, m_r, m_s in zip(self.sne_memory_projs, memory_rgb, memory_sne)
            ]

        elif self.sne_fusion_mode == 'cross_attn':
            # 双向交叉注意力
            mask_fused = self._cross_attn_fusion_2d(
                mask_rgb, mask_sne, self.sne_mask_cross_attn, self.sne_mask_norm)
            memory_fused = [
                self._cross_attn_fusion_2d(m_r, m_s, attn, norm)
                for m_r, m_s, attn, norm in zip(
                    memory_rgb, memory_sne, 
                    self.sne_memory_cross_attns, self.sne_memory_norms)
            ]

        elif self.sne_fusion_mode == 'concat':
            # 通道拼接 (通道数翻倍)
            mask_fused = torch.cat([mask_rgb, mask_sne], dim=1)
            memory_fused = [
                torch.cat([m_r, m_s], dim=1)
                for m_r, m_s in zip(memory_rgb, memory_sne)
            ]

        elif self.sne_fusion_mode == 'ot':
            # 最优传输 (简单版本: 基于相似度加权)
            mask_fused = self._ot_fusion_2d(mask_rgb, mask_sne)
            memory_fused = [
                self._ot_fusion_2d(m_r, m_s)
                for m_r, m_s in zip(memory_rgb, memory_sne)
            ]

        else:
            raise ValueError(f"Unknown sne_fusion_mode: {self.sne_fusion_mode}")

        return mask_fused, memory_fused

    def _cross_attn_fusion_2d(self, feat_rgb, feat_sne, cross_attn, norm):
        """对 2D 特征图进行交叉注意力融合。"""
        B, C, H, W = feat_rgb.shape
        # [B, C, H, W] -> [B, H*W, C]
        rgb_flat = feat_rgb.flatten(2).permute(0, 2, 1)
        sne_flat = feat_sne.flatten(2).permute(0, 2, 1)

        # RGB 用 SNE 增强
        attn_out, _ = cross_attn(query=rgb_flat, key=sne_flat, value=sne_flat)
        fused = norm(rgb_flat + attn_out)

        # [B, H*W, C] -> [B, C, H, W]
        return fused.permute(0, 2, 1).view(B, C, H, W)

    def _ot_fusion_2d(self, feat_rgb, feat_sne):
        """基于相似度的最优传输风格融合 (简化版)。"""
        # 计算通道间相似度作为传输权重
        B, C, H, W = feat_rgb.shape
        rgb_norm = F.normalize(feat_rgb, dim=1)
        sne_norm = F.normalize(feat_sne, dim=1)

        # 简单版本: 相似度加权融合
        similarity = (rgb_norm * sne_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        weight = torch.sigmoid(similarity)  # [0, 1]

        return weight * feat_rgb + (1 - weight) * feat_sne

    # =========================================================================
    # Forward 方法 - 主体逻辑与 tqdmHead 一致
    # =========================================================================

    def forward(self, feats, texts, img_metas, sne_feature=None,
                return_mask_features=False, get_similarity=False, return_attn=False):
        """前向传播 - 主体逻辑与 tqdmHead.forward() 一致。
        
        Args:
            feats: 多尺度 RGB 特征
            texts: 文本嵌入
            img_metas: 图像元数据
            sne_feature: SNE 特征
                - None: backbone 融合完成或不使用 SNE，行为与 tqdmHead 完全一致
                - 有值: pixel 阶段融合，使用独立 pixel_decoder 并行处理后融合
            return_mask_features: 是否返回 mask features
            get_similarity: 是否计算相似度
            return_attn: 是否返回注意力
        """
        batch_size = len(img_metas)

        # =====================================================================
        # Step 1: Pixel Decoder (与 tqdmHead 一致，增加 SNE 并行分支)
        # =====================================================================
        # 检测 pixel_decoder 类型: 有 text_proj 属性表示需要文本交叉注意力
        _need_text = hasattr(self.pixel_decoder, 'text_proj')

        attns = None
        if sne_feature is not None:
            # [增量] pixel 阶段融合: RGB 和 SNE 并行过各自的 pixel_decoder
            if _need_text:
                if return_attn:
                    mask_rgb, memory_rgb, attns_rgb = self.pixel_decoder(feats, texts)
                    mask_sne, memory_sne, attns_sne = self.pixel_decoder_sne(sne_feature, texts)
                    attns = {'rgb': attns_rgb, 'sne': attns_sne}
                else:
                    mask_rgb, memory_rgb = self.pixel_decoder(feats, texts)
                    mask_sne, memory_sne = self.pixel_decoder_sne(sne_feature, texts)
            else:
                # 标准 Mask2Former pixel_decoder，无文本交叉注意力
                mask_rgb, memory_rgb = self.pixel_decoder(feats)
                mask_sne, memory_sne = self.pixel_decoder_sne(sne_feature)
            # 融合两个分支的输出
            mask_features, multi_scale_memorys = self._fuse_pixel_outputs(
                mask_rgb, memory_rgb, mask_sne, memory_sne)
        else:
            # [与 tqdmHead 一致] 标准 pixel_decoder 处理
            if _need_text:
                if return_attn:
                    mask_features, multi_scale_memorys, attns_rgb = self.pixel_decoder(feats, texts)
                    attns = {'rgb': attns_rgb}
                else:
                    mask_features, multi_scale_memorys = self.pixel_decoder(feats, texts)
            else:
                # 标准 Mask2Former pixel_decoder (MSDeformAttnPixelDecoder)
                mask_features, multi_scale_memorys = self.pixel_decoder(feats)

        # =====================================================================
        # Step 2: 构建 Decoder 输入和位置编码 (与 tqdmHead 完全一致)
        # =====================================================================
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        # =====================================================================
        # Step 3: Query 初始化 (与 tqdmHead 完全一致)
        # =====================================================================
        query_feat = self.text_proj(texts).permute(1, 0, 2)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

        # =====================================================================
        # Step 4: 初始预测 (与 tqdmHead 完全一致)
        # =====================================================================
        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:], get_similarity=get_similarity)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        # =====================================================================
        # Step 5: Transformer Decoder 迭代 (与 tqdmHead 完全一致)
        # =====================================================================
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=[attn_mask, None],
                query_key_padding_mask=None,
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                get_similarity=get_similarity)
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        # =====================================================================
        # Step 6: 返回结果 (与 tqdmHead 完全一致)
        # =====================================================================
        if return_mask_features:
            return cls_pred_list, mask_pred_list, mask_features
        elif return_attn:
            return cls_pred_list, mask_pred_list, attns
        else:
            return cls_pred_list, mask_pred_list

    def forward_train(self, x, texts, img_metas, gt_semantic_seg, train_cfg,
                      gt_labels, gt_masks, sne_feature=None):
        """训练前向传播 - 增加 sne_feature 参数。"""
        all_cls_scores, all_mask_preds = self(x, texts, img_metas, sne_feature=sne_feature)
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)
        return losses

    def forward_test(self, inputs, texts, img_metas, test_cfg, return_attn=False, sne_feature=None):
        """测试前向传播 - 增加 sne_feature 参数。"""
        if return_attn:
            all_cls_scores, all_mask_preds, attns = self(inputs, texts, img_metas, sne_feature=sne_feature, return_attn=True)
        else:
            all_cls_scores, all_mask_preds = self(inputs, texts, img_metas, sne_feature=sne_feature)
        
        cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
        cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        
        if return_attn:
            return seg_mask, attns
        return seg_mask

    def forward_inference(self, inputs, texts, img_metas, test_cfg, sne_feature=None):
        """推理前向传播 - 增加 sne_feature 参数。"""
        all_cls_scores, all_mask_preds, mask_features = self(
            inputs, texts, img_metas, sne_feature=sne_feature, return_mask_features=True)
        return all_mask_preds, mask_features