import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

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
        if self.use_sne:
            # self.backbone 是在父类中被初始化的，这里可以直接使用
            self.sne_backbone = copy.deepcopy(self.backbone)
            self.sne_proj = nn.Linear(text_dim, 2)

    def extract_feat(self, img):
        x = self.backbone.extract_feats(img)
        return x

    def after_extract_feat(self, x):
        x_orig = list(x[:-1])
        # global_feat是开头的那个cls token， visual_embeddings是后面的patch tokens
        global_feat, visual_embeddings = x[-1]
        b_size = global_feat.shape[0]

        visual_context = torch.cat([global_feat, visual_embeddings.flatten(-2).permute(0, 2, 1)], dim=1)
        text_embeddings = self.text_encoder(self.texts, context=self.contexts).expand(b_size, -1, -1)

        if self.context_decoder is not None:
            # context decoder是使用text 从visual中提取信息，然后以较小的梯度加到text embeddings上
            text_diff = self.context_decoder(text_embeddings, visual_context)
            text_embeddings = text_embeddings + self.gamma * text_diff
        ret_text_emb = text_embeddings

        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
        # 做一次初步的分割, 余弦相似度
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_embeddings)

        return x_orig, score_map, ret_text_emb, global_feat

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        x = self.extract_feat(img)
        if self.use_sne:
            sne = kwargs['sne']
            _, sne_feature = self.sne_backbone.extract_feats(sne)[-1]
            b, c, h, w = sne_feature.shape
            sne_feature = self.sne_proj(sne_feature.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)).view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            sne_feature = resize(sne_feature, size=(128,128), mode='bilinear', align_corners=False)
            sne_sigmoid = torch.sigmoid(sne_feature)
        x_orig, score_map, text_emb, global_feat = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        losses = dict()

        if self.use_sne:
            loss_score_map = self.identity_head.forward_train(
                sne_sigmoid, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_score_map, 'sne_map'))

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
        if self.use_sne:
            loss_decode = self.decode_head.forward_train(
                x, text_emb, img_metas, gt_semantic_seg, self.train_cfg, kwargs['gt_labels'], kwargs['gt_masks'], sne_feature=sne_sigmoid[:,-1:])
        else:
            loss_decode = self.decode_head.forward_train(
                x, text_emb, img_metas, gt_semantic_seg, self.train_cfg, kwargs['gt_labels'], kwargs['gt_masks'])
        losses.update(add_prefix(loss_decode, 'decode'))

        return losses

    def encode_decode(self, img, img_metas, return_attn=False, **kwargs):
        x = self.extract_feat(img)
        if self.use_sne:
            sne = kwargs['sne']
            _, sne_feature = self.sne_backbone.extract_feats(sne)[-1]
            b, c, h, w = sne_feature.shape
            sne_feature = self.sne_proj(sne_feature.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)).view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            sne_feature = resize(sne_feature, size=(128,128), mode='bilinear', align_corners=False)
            sne_sigmoid = torch.sigmoid(sne_feature)
        x_orig, score_map, text_emb, global_feat = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        if self.use_sne:
            out = self.decode_head.forward_test(
                x, text_emb, img_metas, self.test_cfg, return_attn=return_attn, sne_feature=sne_sigmoid[:,-1:])
        else:
            out = self.decode_head.forward_test(
                x, text_emb, img_metas, self.test_cfg, return_attn=return_attn)
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
