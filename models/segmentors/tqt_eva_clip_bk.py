import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from models.segmentors import tqdm_EVA_CLIP

from ..backbones.eva_clip import get_backbone
from ..backbones.utils import tokenize

@SEGMENTORS.register_module()
class tqt_EVA_CLIP(tqdm_EVA_CLIP):

    def __init__(self,
                 eva_clip,
                 decode_head,
                 class_names,
                 context_length,
                 context_decoder=None,
                 token_embed_dim=512, 
                 text_dim=512,
                 neck=None,
                 identity_head=None,
                 visual_reg=True,
                 textual_reg=True,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 use_sne=True,
                 prompt_cls=False,
                 feature_phase='pixel',
                 feature_mode='proj',
                 use_context=True,
                 **args):

        # 1. 调用父类的 __init__ 方法，并传递所有它需要的参数
        #    让父类完成所有基础组件的初始化工作
        super(tqt_EVA_CLIP, self).__init__(
            eva_clip=eva_clip,
            decode_head=decode_head,
            class_names=class_names,
            context_length=context_length,
            context_decoder=context_decoder,
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

        # 2. 在父类初始化完成后，只执行子类自己新增的逻辑
        self.use_sne = use_sne
        self.prompt_cls = prompt_cls
        self.use_context = use_context
        assert feature_phase in ['pixel', 'context'], "feature_phase must be 'pixel' or 'context'"
        assert feature_mode in ['proj', 'add'], "feature_mode must be 'proj' or 'add'"
        self.feature_phase = feature_phase
        self.feature_mode = feature_mode
        print(f"use_sne: {self.use_sne}, prompt_cls: {self.prompt_cls}, feature_phase: {self.feature_phase}, feature_mode: {self.feature_mode}")
        with open('/root/tqdm/dataset/ORFD/english_scene_dict.json', 'r') as f:
                import json
                self.scene2info = json.load(f)
        if self.use_sne:
            # 保存示例图片的标志位
            self.save_img_sne_sum = 0
            # self.backbone 是在父类中被初始化的，这里可以直接使用
            self.context_decoder_sne = copy.deepcopy(self.context_decoder)
            in_channels = 768 * 2 
            out_channels = 768
            self.img_sne_proj = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(4)]
            )
            self.sne_backbone = copy.deepcopy(self.backbone)
            self.gamma_sne = nn.Parameter(torch.ones(text_dim) * 1e-4)
            nn.init.trunc_normal_(self.gamma_sne)

        if self.prompt_cls:
            self.weather_prompt = nn.Parameter(torch.randn(1, self.prompt_num, token_embed_dim))
            self.light_prompt = nn.Parameter(torch.randn(1, self.prompt_num, token_embed_dim))
            self.road_prompt = nn.Parameter(torch.randn(1, self.prompt_num, token_embed_dim))
            # --- 新增：保存类别名称列表 ---
            self.weather_class_names = ['sunny', 'snowy', 'foggy', 'rainy']
            self.road_class_names = ['paved road', 'unpaved road']
            self.light_class_names = ['daytime', 'dusk', 'nighttime']
            # -----------------------------

            # --- 修正变量名并使用上面定义的列表 ---
            self.weather_texts = torch.cat([tokenize(c, context_length=context_length) for c in self.weather_class_names]).to('cuda')
            self.weather_t0 = torch.cat([tokenize(
                texts=f"A photo of {c}", context_length=self.context_length)  for c in self.weather_class_names]).to('cuda')
            self.weather_t0 = F.normalize(self.text_encoder(self.weather_t0, context=None), dim=-1, p=2)
            self.weather_I = torch.arange(len(self.weather_class_names), device='cuda')

            self.road_texts = torch.cat([tokenize(c, context_length=context_length) for c in self.road_class_names]).to('cuda')
            self.road_t0 = torch.cat([tokenize(
                texts=f"A photo of {c}", context_length=self.context_length)  for c in self.road_class_names]).to('cuda')
            self.road_t0 = F.normalize(self.text_encoder(self.road_t0, context=None), dim=-1, p=2)
            self.road_I = torch.arange(len(self.road_class_names), device='cuda')

            self.light_texts = torch.cat([tokenize(c, context_length=context_length) for c in self.light_class_names]).to('cuda')
            self.light_t0 = torch.cat([tokenize(
                texts=f"A photo of {c}", context_length=self.context_length)  for c in self.light_class_names]).to('cuda')
            self.light_t0 = F.normalize(self.text_encoder(self.light_t0, context=None), dim=-1, p=2)
            self.light_I = torch.arange(len(self.light_class_names), device='cuda')

        if self.use_context is False:
            self.context_decoder = None

    def extract_feat(self, img):
        x = self.backbone.extract_feats(img)
        return x

    def after_extract_feat(self, x, sne=None):
        if self.use_sne:
            assert sne is not None, "sne feature is None"
        x_orig = list(x[:-1])
        if self.use_sne and sne is not None:
            sne_orig = list(sne[:-1])
            if self.feature_phase == 'context':
                if self.feature_mode == 'add':
                    new_orig = [x_orig[i]+sne_orig[i] for i in range(len(x_orig))]
                elif self.feature_mode == 'proj':
                    new_orig = [self.img_sne_proj[i](torch.cat([x_orig[i], sne_orig[i]], dim=1)) for i in range(len(x_orig))]
            else:
                new_orig = (x_orig, sne_orig)
        else:
            new_orig = x_orig
        # global_feat[b,1,c]是开头的那个cls token， visual_embeddings是后面的patch tokens
        global_feat, visual_embeddings = x[-1]
        visual_context = torch.cat([global_feat, visual_embeddings.flatten(-2).permute(0, 2, 1)], dim=1)
        b_size = global_feat.shape[0]

        if self.use_sne and sne is not None:
            global_sne_feat, sne_embeddings = sne[-1]
            sne_context = torch.cat([global_sne_feat, sne_embeddings.flatten(-2).permute(0, 2, 1)], dim=1)
        
        if self.prompt_cls:
            # [b,1,c]
            weather_embedding = self.text_encoder(self.weather_texts, context=self.weather_prompt).expand(b_size, -1, -1)
            light_embedding = self.text_encoder(self.light_texts, context=self.light_prompt).expand(b_size, -1, -1)
            road_embedding = self.text_encoder(self.road_texts, context=self.road_prompt).expand(b_size, -1, -1)

            image_feature = F.normalize(global_feat.squeeze(1), dim=-1, p=2)

            # 2. 归一化文本特征
            weather_feature = F.normalize(weather_embedding, dim=-1, p=2)
            light_feature = F.normalize(light_embedding, dim=-1, p=2)
            road_feature = F.normalize(road_embedding, dim=-1, p=2)

            # 3. 计算余弦相似度作为分类 logits
            #    'bc,bnc->bn' 表示 (B, C) @ (B, N, C) -> (B, N)
            weather_logits = torch.einsum('bc,bnc->bn', image_feature, weather_feature)
            light_logits = torch.einsum('bc,bnc->bn', image_feature, light_feature)
            road_logits = torch.einsum('bc,bnc->bn', image_feature, road_feature)

            attr_logits = (weather_logits, light_logits, road_logits)

            # --- 新增：根据 logits 获取预测的类别文本 ---
            # 1. 找到每个 logits 中得分最高的索引
            weather_preds_idx = weather_logits.argmax(dim=1)
            light_preds_idx = light_logits.argmax(dim=1)
            road_preds_idx = road_logits.argmax(dim=1)

            # 2. 根据索引从类别名称列表中查找文本
            #    这将为批次中的每个图像生成一个预测文本
            pred_weather_texts = [self.weather_class_names[i] for i in weather_preds_idx]
            pred_light_texts = [self.light_class_names[i] for i in light_preds_idx]
            pred_road_texts = [self.road_class_names[i] for i in road_preds_idx]

            tokenized_prompts_list = []
            for i in range(b_size):
                # 1. 构建当前图片专属的动态前缀 (已加入天气信息)
                dynamic_prefix = f"A {pred_weather_texts[i]} scene during {pred_light_texts[i]} on a {pred_road_texts[i]}, "
                
                # 2. 创建新的文本提示 (e.g., "A sunny scene during daytime on a paved road, a clean origami of a car")
                full_dynamic_prompts = [f"{dynamic_prefix}{c}" for c in self.class_names]
                
                # 3. Tokenize 新的提示
                tokenized_prompts = torch.cat([tokenize(p, context_length=self.context_length) for p in full_dynamic_prompts]).to(global_feat.device)
                current_dynamic_embedding = self.text_encoder(tokenized_prompts, context=self.contexts)
                tokenized_prompts_list.append(current_dynamic_embedding)

            # 5. 将列表中的所有张量堆叠成一个新的批次
            text_embeddings = torch.stack(tokenized_prompts_list, dim=0)
            global_feat = (global_feat, attr_logits)
        else:
            text_embeddings = self.text_encoder(self.texts, context=self.contexts).expand(b_size, -1, -1)

        if self.context_decoder is not None:
            # context decoder是使用text 从visual中提取信息，然后以较小的梯度加到text embeddings上
            text_diff = self.context_decoder(text_embeddings, visual_context)
            if self.use_sne and sne is not None:
                text_sne_diff = self.context_decoder_sne(text_embeddings, sne_context)
                text_embeddings = text_embeddings + self.gamma * text_diff + self.gamma_sne * text_sne_diff
            else:
                text_embeddings = text_embeddings + self.gamma * text_diff
        ret_text_emb = text_embeddings

        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
        # 做一次初步的分割, 余弦相似度
        score_map_img = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_embeddings)
        score_map = {
            'img': score_map_img,
        }
        if self.use_sne and sne is not None:
            sne_embeddings = F.normalize(sne_embeddings, dim=1, p=2)
            score_map_sne = torch.einsum('bchw,bkc->bkhw', sne_embeddings, text_embeddings)
            score_map.update(sne=score_map_sne)
        
        return new_orig, score_map, ret_text_emb, global_feat

    def save_img_sne_merge(self, img, sne, img_metas, masks):
        # 合并sne±img，并转化为img保存。
            im = img[0].detach().float().cpu()
            sn = sne[0].detach().float().cpu()
            ma = masks[0][0:1].detach().float().cpu()
            def _to01(x):
                x = x.clone()
                x -= x.min()
                denom = x.max() - x.min() + 1e-6
                x /= denom
                if x.size(0) == 1:
                    x = x.repeat(3, 1, 1)
                return x.clamp(0, 1)

            im01 = _to01(im)
            sn01 = _to01(sn)
            ma01 = _to01(ma)
            # 简单叠加可视化
            overlay = (0.6 * im01 + 0.4 * sn01).clamp(0, 1)

            concat1 = torch.cat([im01, sn01], dim=2)
            concat2 = torch.cat([overlay, ma01], dim=2)
            concat = torch.cat([concat1, concat2], dim=1)
            # 横向拼接 [C, H, W_total]

            # 文件名
            base = os.path.basename(img_metas[0].get('ori_filename', f'{0}.png'))
            os.makedirs(self.train_cfg['img_sne_save_path'], exist_ok=True)
            out_path = os.path.join(self.train_cfg['img_sne_save_path'], f'sne_vis_{base}')
            try:
                vutils.save_image(concat, out_path)
                self.save_img_sne_sum += 1
            except Exception:
                pass

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        losses = dict()

        x = self.extract_feat(img)
        sne = kwargs.get('sne', None)
        masks = kwargs.get('gt_masks', None)
        sne_feature = None
        if self.use_sne and sne is not None:
            if self.save_img_sne_sum < 100:
                self.save_img_sne_merge(img, sne, img_metas, masks)
            sne_feature = self.sne_backbone.extract_feats(sne)
        x_orig, score_map_all, text_emb, global_feat = self.after_extract_feat(x, sne_feature)
        score_map = score_map_all['img']

        if self.feature_phase == 'pixel' and self.use_sne and sne is not None:
            x_orig, sne_orig = x_orig
        else:
            sne_orig = None

        if 'sne' in score_map_all:
            score_map_sne = score_map_all['sne']
            # identity是什么？
            loss_score_map_sne = self.identity_head.forward_train(
                score_map_sne/self.tau, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_score_map_sne, 'sne_map'))
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        if self.prompt_cls:
            global_feat, attr_logits = global_feat
            weather_logits, light_logits, road_logits = attr_logits
            road_list = []
            weather_list = []
            light_list = []
            for i in range(len(img_metas)):
                scene_name = img_metas[i]['ori_filename'].split('/')[0][1:].replace('_', '-')
                road = self.scene2info[scene_name]['road'].split("_")[0] + ' road'
                weather = self.scene2info[scene_name]['weather']
                light = self.scene2info[scene_name]['light']
                # 使用预测对应的标签计算损失，可能会由于只有某个场景产生偏差。
                road_index = self.road_class_names.index(road)
                weather_index = self.weather_class_names.index(weather)
                light_index = self.light_class_names.index(light)
                road_list.append(road_index)
                weather_list.append(weather_index)
                light_list.append(light_index)
            road_target = torch.tensor(road_list).to(road_logits.device)
            weather_target = torch.tensor(weather_list).to(weather_logits.device)
            light_target = torch.tensor(light_list).to(light_logits.device)
            loss_weather = F.cross_entropy(weather_logits/self.tau, weather_target, reduction='mean')
            loss_road = F.cross_entropy(road_logits/self.tau, road_target, reduction='mean')
            loss_light = F.cross_entropy(light_logits/self.tau, light_target, reduction='mean')
            loss_weather_l = {'loss' : loss_weather}
            loss_road_l = {'loss' : loss_road}
            loss_light_l = {'loss' : loss_light}
            # --- 1. 新增：将损失加入总损失字典 ---
            losses.update(add_prefix(loss_weather_l, 'attr.weather'))
            losses.update(add_prefix(loss_road_l, 'attr.road'))
            losses.update(add_prefix(loss_light_l, 'attr.light'))
            # --- 2. 新增：计算准确率 ---
            with torch.no_grad():
                # 天气准确率
                pred_weather = torch.argmax(weather_logits, dim=1)
                correct_weather = (pred_weather == weather_target).sum()
                total_samples = weather_target.size(0)
                self.acc_weather = correct_weather.float() / total_samples

                # 道路准确率
                pred_road = torch.argmax(road_logits, dim=1)
                correct_road = (pred_road == road_target).sum()
                self.acc_road = correct_road.float() / total_samples

                # 光照准确率
                pred_light = torch.argmax(light_logits, dim=1)
                correct_light = (pred_light == light_target).sum()
                self.acc_light = correct_light.float() / total_samples

        # language regularization
        if self.textual_reg is True:
            content_score = torch.einsum('blc,kc->bkl', F.normalize(text_emb, dim=-1, p=2), self.reg_T0.detach())
            loss_reg_l = F.cross_entropy(content_score,
                self.reg_I.expand(content_score.shape[0], -1),
                reduction='mean')
            loss_reg_l = {'loss' : loss_reg_l}
            losses.update(add_prefix(loss_reg_l, 'reg.textual'))

        # vision regularization
        if self.visual_reg is True:
            with torch.no_grad():
                # _是什么？
                global_feat_0, _ = self.reg_E0.extract_feats(img)[-1]
            # 只限制第一个分类的距离，不限制patch的距离。
            loss_reg_v = nn.MSELoss(reduction='mean')(global_feat, global_feat_0)
            loss_reg_v = {'loss' : loss_reg_v}
            losses.update(add_prefix(loss_reg_v, 'reg.visual'))

        # vision-language regularization
        if self.identity_head is not None:
            # identity是什么？
            loss_score_map = self.identity_head.forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_score_map, 'scr_map'))

        # decode head loss
        loss_decode = self.decode_head.forward_train(
            x, text_emb, img_metas, gt_semantic_seg, self.train_cfg, kwargs['gt_labels'], kwargs['gt_masks'], sne_feature=sne_orig, feature_mode=self.feature_mode)
        losses.update(add_prefix(loss_decode, 'decode'))

        return losses

    def encode_decode(self, img, img_metas, return_attn=False, **kwargs):
        x = self.extract_feat(img)
        sne = kwargs.get('sne', None)
        sne_feature = None
        if self.use_sne and sne is not None:
            sne_feature = self.sne_backbone.extract_feats(sne)
        x_orig, score_map_all, text_emb, global_feat = self.after_extract_feat(x, sne_feature)
        score_map = score_map_all['img']

        if self.feature_phase == 'pixel' and self.use_sne and sne is not None:
            x_orig, sne_orig = x_orig
        else:
            sne_orig = None

        if 'sne' in score_map_all:
            score_map_sne = score_map_all['sne']
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        if self.prompt_cls:
            global_feat, attr_logits = global_feat
        
        out = self.decode_head.forward_test(
            x, text_emb, img_metas, self.test_cfg, return_attn=return_attn, sne_feature=sne_orig, feature_mode=self.feature_mode)
        
        if return_attn:
            out, attn = out
        out = resize(
            input=out,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=False)
        if return_attn:
            return out, attn
        else:
            return out
