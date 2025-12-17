import os
from pathlib import Path

import cv2
import mmcv
import numpy as np
import swanlab
import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SaveTrainVisHook(Hook):
    """Save training-time visual previews.

    Runs before each train iter on the primary rank only. It triggers a
    lightweight eval forward (simple_test), builds the debug canvas in-hook
    (input RGB + prob maps on top, argmax/GT on bottom), and logs/saves it.
    Stops after max_samples.
    """

    def __init__(self, interval=1000, max_samples=5, save_debug=False, output_dir=None):
        self.interval = max(1, int(interval))
        self.max_samples = int(max_samples)
        self.save_debug = bool(save_debug)
        self.output_dir = output_dir
        self.saved = 0

    def before_train_iter(self, runner):
        if self.max_samples <= 0 or self.saved >= self.max_samples:
            return
        if runner.iter % self.interval != 0:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        if hasattr(model, '_is_primary_rank') and not model._is_primary_rank():
            return

        batch = runner.data_batch
        if batch is None:
            return

        device = next(model.parameters()).device

        img = batch['img'].data[0].to(device)
        img_metas = batch['img_metas'].data[0]
        sne = None
        if 'sne' in batch and batch['sne'] is not None:
            sne = batch['sne'].data[0].to(device)
        gt_seg = None
        if 'gt_semantic_seg' in batch and batch['gt_semantic_seg'] is not None:
            gt_seg = batch['gt_semantic_seg'].data[0].to(device)

        # optional debug payload
        return_debug = self.save_debug

        was_training = model.training
        if was_training:
            model.eval()

        debug_payload = None
        try:
            with torch.no_grad():
                out = model.simple_test(
                    img, img_metas, rescale=True, sne=sne,
                    return_debug=return_debug
                )
            if return_debug:
                seg_pred, debug_payload = out
            else:
                seg_pred = out
        except Exception as e:
            seg_pred = None
        finally:
            if was_training:
                model.train()

        if seg_pred is None:
            return

        if isinstance(seg_pred, (list, tuple)):
            seg_mask = seg_pred[0]
        else:
            seg_mask = seg_pred

        try:
            canvas = self._build_debug_canvas(model, img, sne, img_metas, gt_seg, seg_mask, debug_payload)
            if canvas is None:
                return

            base_name = os.path.basename(img_metas[0].get('ori_filename', f'{self.saved}.png'))
            caption_debug = f"train_debug::{base_name}"
            swanlab.log({"Train Debug": swanlab.Image(canvas, caption=caption_debug)})

            if self.output_dir:
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f'{base_name}_debug.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            self.saved += 1
        except Exception as e:
            print(e)
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_debug_canvas(self, model, img, sne, img_metas, gt_semantic_seg, seg_mask, debug_payload):
        # need debug payload to draw prob maps
        if debug_payload is None or not isinstance(debug_payload, dict):
            return None

        img_meta = img_metas[0]
        target_hw = img_meta.get('ori_shape', None)

        def _tensor_image_to_uint8(img_tensor, meta):
            img_np = img_tensor[0].detach().cpu().float().permute(1, 2, 0).numpy()
            norm_cfg = meta.get('img_norm_cfg', {}) if isinstance(meta, dict) else {}
            mean = np.array(norm_cfg.get('mean', [0, 0, 0]), dtype=np.float32)
            std = np.array(norm_cfg.get('std', [1, 1, 1]), dtype=np.float32)
            img_np = mmcv.imdenormalize(img_np, mean, std, to_bgr=False)
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            return img_np

        def _resize_to_target(arr):
            if target_hw is None:
                return arr
            if arr.shape[:2] != target_hw[:2]:
                return cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
            return arr

        def _argmax_to_bw(t, traversable_class=1):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.ndim != 3:
                return None
            label = t.argmax(dim=0).byte().numpy()
            mask = (label == traversable_class).astype(np.uint8) * 255
            return np.stack([mask] * 3, axis=-1)

        def _mask_to_color(mask, palette):
            max_label = int(mask.max()) if mask.size else 0
            pal = palette
            if max_label >= pal.shape[0]:
                extra = pal[-1][None, :]
                repeat = max_label - pal.shape[0] + 1
                pal = np.concatenate([pal, np.repeat(extra, repeat, axis=0)], axis=0)
            return pal[mask]

        def _add_title_band(img_arr, title, band_h=28):
            band = np.zeros((band_h, img_arr.shape[1], 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(band, title, (5, band_h - 8), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            return np.concatenate([band, img_arr], axis=0)

        def _to_uint8_img(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.ndim == 3 and t.shape[0] > 1:
                if t.shape[0] >= 3:
                    t = t[:3]
                else:
                    t = t[0:1]
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            t_min, t_max = float(t.min()), float(t.max())
            if t_max - t_min < 1e-8:
                arr = (t * 0).byte().numpy()
            else:
                arr = ((t - t_min) / (t_max - t_min) * 255.0).clamp(0, 255).byte().numpy()
            if arr.ndim == 2:
                arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
            else:
                arr = arr.transpose(1, 2, 0)
            return arr

        img_np = _tensor_image_to_uint8(img, img_meta)
        if target_hw is None:
            target_hw = img_np.shape[:2]

        palette = np.array(getattr(model, 'PALETTE', [[0, 0, 0], [0, 255, 0]]), dtype=np.uint8)

        top_row = []
        bottom_row = []

        # input first
        top_row.append(_add_title_band(_resize_to_target(img_np), 'input_rgb'))
        if gt_semantic_seg is not None:
            gt_mask = gt_semantic_seg[0].detach().cpu().numpy()
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.squeeze(0)
            gt_color = _mask_to_color(gt_mask.astype(np.int64), palette)
            gt_color = _resize_to_target(gt_color)
            bottom_row.append(_add_title_band(gt_color, 'gt_mask'))

        if 'score_map_img' in debug_payload:
            vis = _resize_to_target(_to_uint8_img(debug_payload['score_map_img']))
            top_row.append(_add_title_band(vis, 'score_map_img'))
            arg = _argmax_to_bw(debug_payload['score_map_img'])
            if arg is not None:
                bottom_row.append(_add_title_band(_resize_to_target(arg), 'score_map_img_argmax'))

        if 'score_map_sne' in debug_payload:
            vis = _resize_to_target(_to_uint8_img(debug_payload['score_map_sne']))
            top_row.append(_add_title_band(vis, 'score_map_sne'))
            arg = _argmax_to_bw(debug_payload['score_map_sne'])
            if arg is not None:
                bottom_row.append(_add_title_band(_resize_to_target(arg), 'score_map_sne_argmax'))

        for idx, pi_img in enumerate(debug_payload.get('pi_img_levels', [])):
            vis = _resize_to_target(_to_uint8_img(pi_img))
            top_row.append(_add_title_band(vis, f'pi_img_lvl{idx}'))
            arg = _argmax_to_bw(pi_img)
            if arg is not None:
                bottom_row.append(_add_title_band(_resize_to_target(arg), f'pi_img_lvl{idx}_argmax'))

        for idx, pi_sne in enumerate(debug_payload.get('pi_sne_levels', [])):
            vis = _resize_to_target(_to_uint8_img(pi_sne))
            top_row.append(_add_title_band(vis, f'pi_sne_lvl{idx}'))
            arg = _argmax_to_bw(pi_sne)
            if arg is not None:
                bottom_row.append(_add_title_band(_resize_to_target(arg), f'pi_sne_lvl{idx}_argmax'))

        if not top_row:
            return None

        top_concat = np.concatenate(top_row, axis=1)
        if bottom_row:
            bottom_concat = np.concatenate(bottom_row, axis=1)
            debug_canvas = np.concatenate([top_concat, bottom_concat], axis=0)
        else:
            debug_canvas = top_concat

        return debug_canvas
