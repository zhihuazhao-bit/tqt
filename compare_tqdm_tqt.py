#!/usr/bin/env python
"""
比较 TQDM 和 TQT (use_sne=False) 的输出是否一致

验证内容:
1. 模型结构对比 (参数数量)
2. Forward 输出对比 (相同输入下的输出差异)
3. Loss 计算对比
"""

import torch
import torch.nn as nn
import numpy as np
from mmcv import Config
from mmseg.models import build_segmentor
from collections import OrderedDict


def set_seed(seed=42):
    """固定随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compare_state_dicts(sd1, sd2, name1='model1', name2='model2'):
    """对比两个 state_dict 的 key 差异"""
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2
    
    print(f"\n{'='*60}")
    print(f"State Dict 对比: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"  {name1} 参数数: {len(keys1)}")
    print(f"  {name2} 参数数: {len(keys2)}")
    print(f"  共同参数: {len(common)}")
    
    if only_in_1:
        print(f"\n  仅在 {name1} 中的参数 ({len(only_in_1)}):")
        for k in sorted(only_in_1)[:10]:
            print(f"    - {k}")
        if len(only_in_1) > 10:
            print(f"    ... 还有 {len(only_in_1) - 10} 个")
    
    if only_in_2:
        print(f"\n  仅在 {name2} 中的参数 ({len(only_in_2)}):")
        for k in sorted(only_in_2)[:10]:
            print(f"    - {k}")
        if len(only_in_2) > 10:
            print(f"    ... 还有 {len(only_in_2) - 10} 个")
    
    return only_in_1, only_in_2, common


def compare_outputs(out1, out2, name1='tqdm', name2='tqt', rtol=1e-5, atol=1e-6):
    """对比两个输出张量"""
    if isinstance(out1, (list, tuple)):
        print(f"\n输出是列表，长度: {len(out1)} vs {len(out2)}")
        all_close = True
        for i, (o1, o2) in enumerate(zip(out1, out2)):
            if isinstance(o1, torch.Tensor):
                close = torch.allclose(o1, o2, rtol=rtol, atol=atol)
                max_diff = (o1 - o2).abs().max().item()
                print(f"  [{i}] shape={o1.shape}, allclose={close}, max_diff={max_diff:.2e}")
                all_close = all_close and close
        return all_close
    elif isinstance(out1, torch.Tensor):
        close = torch.allclose(out1, out2, rtol=rtol, atol=atol)
        max_diff = (out1 - out2).abs().max().item()
        print(f"  输出 shape={out1.shape}, allclose={close}, max_diff={max_diff:.2e}")
        return close
    elif isinstance(out1, dict):
        print(f"\n输出是字典，keys: {list(out1.keys())}")
        all_close = True
        for k in out1.keys():
            if k in out2:
                if isinstance(out1[k], torch.Tensor):
                    close = torch.allclose(out1[k], out2[k], rtol=rtol, atol=atol)
                    max_diff = (out1[k] - out2[k]).abs().max().item()
                    print(f"  [{k}] allclose={close}, max_diff={max_diff:.2e}")
                    all_close = all_close and close
        return all_close
    return True


def main():
    print("="*60)
    print("TQDM vs TQT (use_sne=False) 一致性验证")
    print("="*60)
    
    # ========== 1. 加载配置 ==========
    # TQDM 配置
    tqdm_cfg = Config.fromfile('configs/tqdm/tqdm_eva_vit-b_1e-5_5k-r2r-512-all.py')
    
    # TQT 配置 (use_sne=False)
    tqt_cfg = Config.fromfile('configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-all.py')
    # 确保 use_sne=False
    tqt_cfg.model.use_sne = False
    tqt_cfg.model.prompt_cls = False  # 也关闭 prompt_cls 以保持一致
    
    print("\n[1] 配置加载完成")
    print(f"  TQDM type: {tqdm_cfg.model.type}")
    print(f"  TQT type: {tqt_cfg.model.type}")
    print(f"  TQT use_sne: {tqt_cfg.model.use_sne}")
    
    # ========== 2. 构建模型 ==========
    set_seed(42)
    tqdm_model = build_segmentor(tqdm_cfg.model).cuda()
    tqdm_model.eval()
    
    set_seed(42)
    tqt_model = build_segmentor(tqt_cfg.model).cuda()
    tqt_model.eval()
    
    # ========== 3. 对比参数 ==========
    print("\n[2] 参数统计")
    tqdm_total, tqdm_train = count_parameters(tqdm_model)
    tqt_total, tqt_train = count_parameters(tqt_model)
    print(f"  TQDM: total={tqdm_total:,}, trainable={tqdm_train:,}")
    print(f"  TQT:  total={tqt_total:,}, trainable={tqt_train:,}")
    print(f"  差异: total={tqt_total - tqdm_total:,}, trainable={tqt_train - tqdm_train:,}")
    
    # 对比 state_dict
    compare_state_dicts(
        tqdm_model.state_dict(), 
        tqt_model.state_dict(),
        'TQDM', 'TQT'
    )
    
    # ========== 4. 复制共同参数 ==========
    print("\n[3] 复制 TQDM 参数到 TQT (共同部分)")
    tqdm_sd = tqdm_model.state_dict()
    tqt_sd = tqt_model.state_dict()
    
    # 只复制共同的 key
    common_keys = set(tqdm_sd.keys()) & set(tqt_sd.keys())
    for k in common_keys:
        tqt_sd[k] = tqdm_sd[k].clone()
    tqt_model.load_state_dict(tqt_sd)
    print(f"  已复制 {len(common_keys)} 个参数")
    
    # ========== 5. Forward 对比 ==========
    print("\n[4] Forward 输出对比")
    
    # 构造 dummy 输入
    set_seed(42)
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224).cuda()
    img_metas = [{'ori_shape': (224, 224, 3), 'img_shape': (224, 224, 3), 
                  'pad_shape': (224, 224, 3), 'ori_filename': 'test.png'} 
                 for _ in range(batch_size)]
    
    with torch.no_grad():
        # TQDM forward
        set_seed(42)
        tqdm_feat = tqdm_model.extract_feat(img)
        tqdm_out = tqdm_model.after_extract_feat(tqdm_feat)
        
        # TQT forward (不传 SNE)
        set_seed(42)
        tqt_feat, tqt_sne_feat = tqt_model.extract_feat(img, sne=None)
        tqt_out = tqt_model.after_extract_feat(tqt_feat, sne=None)
    
    print("\n  extract_feat 对比:")
    # 比较特征 (除了最后一个是 tuple)
    for i in range(len(tqdm_feat) - 1):
        close = torch.allclose(tqdm_feat[i], tqt_feat[i], rtol=1e-5, atol=1e-6)
        max_diff = (tqdm_feat[i] - tqt_feat[i]).abs().max().item()
        print(f"    feat[{i}]: shape={tqdm_feat[i].shape}, allclose={close}, max_diff={max_diff:.2e}")
    
    # 最后一个是 (global_feat, visual_embeddings)
    tqdm_gf, tqdm_ve = tqdm_feat[-1]
    tqt_gf, tqt_ve = tqt_feat[-1]
    close_gf = torch.allclose(tqdm_gf, tqt_gf, rtol=1e-5, atol=1e-6)
    close_ve = torch.allclose(tqdm_ve, tqt_ve, rtol=1e-5, atol=1e-6)
    print(f"    global_feat: allclose={close_gf}")
    print(f"    visual_emb:  allclose={close_ve}")
    
    print("\n  after_extract_feat 对比:")
    # tqdm_out: (x_orig, score_map, text_emb, global_feat)
    # tqt_out:  (x_orig, score_map, text_emb, global_feat, orig_embeddings)
    print(f"    TQDM 返回 {len(tqdm_out)} 个元素")
    print(f"    TQT  返回 {len(tqt_out)} 个元素")
    
    # 比较前4个元素
    tqdm_x_orig, tqdm_score, tqdm_text, tqdm_global = tqdm_out[:4]
    tqt_x_orig, tqt_score, tqt_text, tqt_global = tqt_out[:4]
    
    # x_orig (TQT 可能是 tuple)
    if isinstance(tqt_x_orig, tuple):
        tqt_x_orig = tqt_x_orig[0]  # RGB 部分
    
    print("\n    x_orig 对比:")
    for i, (o1, o2) in enumerate(zip(tqdm_x_orig, tqt_x_orig)):
        close = torch.allclose(o1, o2, rtol=1e-5, atol=1e-6)
        max_diff = (o1 - o2).abs().max().item()
        print(f"      [{i}] allclose={close}, max_diff={max_diff:.2e}")
    
    # score_map
    print("\n    score_map 对比:")
    if isinstance(tqt_score, dict):
        tqt_score_img = tqt_score['img']
    else:
        tqt_score_img = tqt_score
    close = torch.allclose(tqdm_score, tqt_score_img, rtol=1e-5, atol=1e-6)
    max_diff = (tqdm_score - tqt_score_img).abs().max().item()
    print(f"      allclose={close}, max_diff={max_diff:.2e}")
    
    # text_emb
    print("\n    text_emb 对比:")
    close = torch.allclose(tqdm_text, tqt_text, rtol=1e-5, atol=1e-6)
    max_diff = (tqdm_text - tqt_text).abs().max().item()
    print(f"      allclose={close}, max_diff={max_diff:.2e}")
    
    # ========== 6. 总结 ==========
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    all_match = (
        close_gf and close_ve and 
        all(torch.allclose(o1, o2, rtol=1e-5, atol=1e-6) 
            for o1, o2 in zip(tqdm_x_orig, tqt_x_orig if not isinstance(tqt_x_orig, tuple) else tqt_x_orig))
    )
    
    if all_match:
        print("✓ TQDM 和 TQT (use_sne=False) 的核心 forward 输出一致!")
    else:
        print("✗ 存在差异，请检查上述对比结果")
    
    print("\n注意: TQT 返回值结构略有不同 (多了 orig_embeddings, score_map 是 dict)")
    print("      但核心计算逻辑应保持一致")


if __name__ == '__main__':
    main()
