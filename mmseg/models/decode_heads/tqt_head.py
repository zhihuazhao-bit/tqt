import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init, ConvModule
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32
from mmseg.models.builder import HEADS, build_loss
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.tqdm_head import tqdmHead

from ...core import build_sampler, multi_apply, reduce_mean
from ..builder import build_assigner
from ..utils import get_uncertain_point_coords_with_randomness


class _ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16, topk_ratio=None):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.activation = nn.Sigmoid()
        if topk_ratio is not None and topk_ratio <= 0:
            raise ValueError('topk_ratio must be greater than 0 when provided.')
        self.topk_ratio = topk_ratio

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attn = self.activation(avg_out + max_out)
        if self.topk_ratio is not None and self.topk_ratio < 1.0:
            b, c, _, _ = attn.shape
            k = max(int(round(c * self.topk_ratio)), 1)
            attn_flat = attn.view(b, c)
            _, topk_idx = torch.topk(attn_flat, k=k, dim=1, largest=True, sorted=False)
            keep_mask = attn_flat.new_zeros(attn_flat.shape)
            keep_mask.scatter_(1, topk_idx, 1.0)
            attn = attn * keep_mask.view(b, c, 1, 1)
        return attn


@HEADS.register_module()
class tqtHead(tqdmHead):

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
                 use_sne=True,
                 feature_phase='pixel',
                 feature_mode='proj',
                 channel_attn_topk_ratio=None,
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
        self.use_sne = use_sne
        assert feature_phase in ['pixel', 'context'], "feature_phase must be 'pixel' or 'context'"
        assert feature_mode in ['proj', 'add', 'concat'], "feature_mode must be 'proj', 'add', or 'concat'"
        self.feature_phase = feature_phase
        self.feature_mode = feature_mode
        self.channel_attn_topk_ratio = channel_attn_topk_ratio
        channel_list = self.in_channels if isinstance(self.in_channels, (list, tuple)) else [self.in_channels]
        reduction = 16
        # self.feat_channel_attn = nn.ModuleList([
        #     _ChannelAttention(ch, reduction, topk_ratio=channel_attn_topk_ratio) for ch in channel_list
        # ])
        self.img_sne_proj = None
        if self.feature_mode == 'proj':
            if isinstance(self.in_channels, (list, tuple)):
                in_ch_list = [in_ch * 2 for in_ch in self.in_channels]
                out_ch_list = list(self.in_channels)
            else:
                in_ch_list = [self.in_channels * 2]
                out_ch_list = [self.in_channels]
            self.img_sne_proj = nn.ModuleList([
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(in_ch_list, out_ch_list)
            ])

        self.pixel_decoder_sne = None
        self.context_mask_proj = None
        self.context_memory_projs = None
        if self.use_sne and self.feature_phase == 'context':
            self.pixel_decoder_sne = copy.deepcopy(self.pixel_decoder)
            if self.feature_mode in ['proj', 'concat']:
                self.context_mask_proj = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
                self.context_memory_projs = nn.ModuleList([
                    nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1)
                    for _ in range(self.num_transformer_feat_level)
                ])

    def init_weights(self):
        super().init_weights()
        if self.pixel_decoder_sne is not None:
            self.pixel_decoder_sne.init_weights()

    def forward(self, feats, texts, img_metas, sne_feature=None, return_mask_features=False, get_similarity=False, return_attn=False, feature_mode=None): ### need texts for pixel_decoder !
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_size = len(img_metas)
        current_feature_mode = feature_mode if feature_mode is not None else self.feature_mode
        attns = None

        # 这一步使用text作为key和value，使用visual作为query，构建多尺度的特征。
        
        if self.use_sne and self.feature_phase == 'context':
            assert sne_feature is not None, "sne_feature should not be None when use_sne is True"
            if self.pixel_decoder_sne is None:
                raise RuntimeError('pixel_decoder_sne is not initialized for context fusion.')
            if return_attn:
                mask_img, multi_scale_img, attns_img = self.pixel_decoder(feats, texts)
                mask_sne, multi_scale_sne, attns_sne = self.pixel_decoder_sne(sne_feature, texts)
            else:
                mask_img, multi_scale_img = self.pixel_decoder(feats, texts)
                mask_sne, multi_scale_sne = self.pixel_decoder_sne(sne_feature, texts)
            mask_features, multi_scale_memorys = self._merge_context_features(
                mask_img, multi_scale_img, mask_sne, multi_scale_sne, current_feature_mode)
            if return_attn:
                attns = {'img': attns_img, 'sne': attns_sne}
        else:
            if self.use_sne and self.feature_phase == 'pixel':
                assert sne_feature is not None, "sne_feature should not be None when use_sne is True"
                if current_feature_mode == 'concat':
                    feats = [torch.cat([feats[i], sne_feature[i]], dim=1) for i in range(len(feats))]
                    # feats = [feats[i] * self.feat_channel_attn[i](feats[i]) for i in range(len(feats))]
                elif current_feature_mode == 'add':
                    feats = [feats[i] + sne_feature[i] for i in range(len(feats))]
                elif current_feature_mode == 'proj':
                    if self.img_sne_proj is None:
                        raise RuntimeError('img_sne_proj is not initialized for proj fusion mode.')
                    feats = [self.img_sne_proj[i](torch.cat([feats[i], sne_feature[i]], dim=1)) for i in range(len(feats))]
            if return_attn:
                mask_features, multi_scale_memorys, attns = self.pixel_decoder(feats, texts) ### pixel_decoder need texts!
            else:
                mask_features, multi_scale_memorys = self.pixel_decoder(feats, texts) ### pixel_decoder need texts!

        decoder_inputs = []
        decoder_positional_encodings = []
        # 构建每一层的位置编码
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.text_proj(texts).permute(1, 0, 2)
        # query_feat = self.query_feat.weight.unsqueeze(1).repeat(
        #     (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:], get_similarity=get_similarity)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # 进行预测，cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:], get_similarity=get_similarity)

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        if return_mask_features:
            return cls_pred_list, mask_pred_list, mask_features
        elif return_attn and self.training is False:
            return cls_pred_list, mask_pred_list, attns
        else:
            return cls_pred_list, mask_pred_list

    def _merge_context_features(self, mask_img, multi_scale_img, mask_sne, multi_scale_sne, mode):
        if mode == 'add':
            mask_features = mask_img + mask_sne
            multi_scale_memorys = [m_img + m_sne for m_img, m_sne in zip(multi_scale_img, multi_scale_sne)]
        elif mode in ['proj', 'concat']:
            if self.context_mask_proj is None or self.context_memory_projs is None:
                raise RuntimeError('Context projection layers are not initialized for proj fusion mode.')
            mask_features = self.context_mask_proj(torch.cat([mask_img, mask_sne], dim=1))
            multi_scale_memorys = [
                proj(torch.cat([m_img, m_sne], dim=1))
                for proj, m_img, m_sne in zip(self.context_memory_projs, multi_scale_img, multi_scale_sne)
            ]
        else:
            raise ValueError(f'Unsupported feature fusion mode: {mode}')
        return mask_features, multi_scale_memorys

    def forward_train(self, x, texts, img_metas, gt_semantic_seg, train_cfg,
                      gt_labels, gt_masks, sne_feature=None, **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Multi-level features from the upstream network,
                each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_semantic_seg (list[tensor]):Each element is the ground truth
                of semantic segmentation with the shape (N, H, W).
            train_cfg (dict): The training config, which not been used in
                maskformer.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).

        Returns:
            losses (dict[str, Tensor]): a dictionary of loss components
        """

        # forward
        all_cls_scores, all_mask_preds = self(x, texts, img_metas, sne_feature=sne_feature, **kwargs)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)

        return losses

    def forward_test(self, inputs, texts, img_metas, test_cfg, return_attn=False, sne_feature=None, **kwargs):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            inputs (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            test_cfg (dict): Testing config.

        Returns:
            seg_mask (Tensor): Predicted semantic segmentation logits.
        """
        if return_attn:
            all_cls_scores, all_mask_preds, attns = self(inputs, texts, img_metas, sne_feature=sne_feature, return_attn=return_attn, **kwargs)
        else:
            all_cls_scores, all_mask_preds = self(inputs, texts, img_metas, sne_feature=sne_feature, **kwargs)
        cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
        ori_h, ori_w, _ = img_metas[0]['ori_shape']

        # semantic inference
        cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        if return_attn:
            return seg_mask, attns
        else:
            return seg_mask

    def forward_inference(self, inputs, texts, img_metas, test_cfg, sne_feature):
        all_cls_scores, all_mask_preds, mask_features =\
            self(inputs, texts, img_metas, sne_feature, return_mask_features=True)
        return all_mask_preds, mask_features