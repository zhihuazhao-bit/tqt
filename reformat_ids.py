import ast
import pprint
import sys

# 1. 定义 ID 生成逻辑 (风格统一化)
def generate_standard_id(cfg):
    parts = []
    
    # --- Part 1: Backbone & Size ---
    # E224, D512
    bb = 'D' if 'DenseVLM' in cfg['weight'] else 'E'
    parts.append(f"{bb}{cfg['size']}")
    
    # --- Part 2: SNE & Fusion ---
    if not cfg['sne']:
        parts.append("Base")
    else:
        mode = cfg['sne_mode'] # proj, ot, cross_attn
        stage = cfg['sne_fusion_stage'] # backbone, pixel
        
        if mode == 'proj':
            # BbPr (Backbone-Proj), PxPr (Pixel-Proj)
            s_str = 'Px' if stage == 'pixel' else 'Bb'
            # 默认为 proj, 如果是其他简单模式可以直接加
            s_str += 'Pr'
        elif mode == 'cross_attn':
            s_str = 'CA'
        elif mode == 'ot':
            # OT 特性组合
            s_str = 'OT'
            
            # Prior: P (argmax), Pp (prob), nothing (No prior)
            if cfg.get('ot_prior'):
                s_str += 'P'
                if cfg.get('ot_prior_mode') == 'prob':
                    s_str += 'p' # Probabilistic
            
            # Soft Union
            if cfg.get('ot_softunion'):
                s_str += 'U'
                
            # Cost Type: L2 (default is cos, omitted)
            if cfg.get('ot_cost_type') == 'l2':
                s_str += 'L2'
                
            # Fuse Mode: m (mean), x (max), default (proj) omitted
            if cfg.get('ot_fuse_mode') == 'mean':
                s_str += 'm'
            elif cfg.get('ot_fuse_mode') == 'max':
                s_str += 'x'
                
            # Fuse Output: Nr (No Residual / No Output fusion?)
            # 这里的字段是 ot_fuse_output (bool). True=Residual, False=NoResidual
            if cfg.get('ot_fuse_output') is False:
                s_str += 'Nr'
        else:
            s_str = mode
            
        parts.append(s_str)

    # --- Part 3: Prompt ---
    if cfg['prompt']:
        p_str = 'P'
        # cls mode
        if cfg.get('prompt_cls_mode') == 'soft':
            p_str += 's'
        # hard is default
        parts.append(p_str)
    
    # --- Part 4: Modules ---
    if not cfg.get('context_decoder', True):
        parts.append("NoCD")
    
    if cfg.get('patch_fpn'):
        parts.append("FPN")
    elif cfg.get('patch_fpn_xsam'):
        parts.append("XSam")
        
    if cfg.get('pi_sup'):
        parts.append("Pi")
        
    if cfg.get('reg_e0_eval'):
        parts.append("RegE0")

    return "-".join(parts)

# 2. 处理文件
def reformat_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 提取字典
    start_idx = content.find('ABLATION_EXPERIMENTS = {')
    if start_idx == -1:
        print("Dictionary not found!")
        return
    
    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(content)):
        char = content[i]
        if char == '{': brace_count += 1
        elif char == '}': 
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    dict_str = content[start_idx:end_idx].split('=', 1)[1].strip()
    try:
        current_experiments = ast.literal_eval(dict_str)
    except Exception as e:
        print(f"Error parsing dict: {e}")
        return

    # 重构字典
    new_experiments = {}
    
    # 为了保持一定的顺序，可以先排序
    # 但我们更希望保持原有顺序或按配置逻辑排序
    # 这里简单按 key 排序遍历
    
    for old_id, exp_cfg in current_experiments.items():
        # 生成新 ID
        new_id = generate_standard_id(exp_cfg)
        
        # 存入新字典 (注意：如果配置完全相同，可能会覆盖，但消融实验不应有完全重复的配置)
        if new_id in new_experiments:
            print(f"Warning: Duplicate ID generated: {new_id}. Previous: {new_experiments[new_id]['name']}, Current: {exp_cfg['name']}")
            # 如果冲突，加上后缀区分
            suffix = 1
            while f"{new_id}_v{suffix}" in new_experiments:
                suffix += 1
            new_id = f"{new_id}_v{suffix}"
            
        new_experiments[new_id] = exp_cfg

    # 排序：按 ID 字母序，这样 E224 会在一起，E512 会在一起
    sorted_experiments = dict(sorted(new_experiments.items()))

    # 生成字符串
    new_dict_str = "ABLATION_EXPERIMENTS = " + pprint.pformat(sorted_experiments, indent=4, width=120, sort_dicts=False)
    
    # 替换
    new_content = content[:start_idx] + new_dict_str + content[end_idx:]
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("Successfully reformatted IDs.")

reformat_file('statis_results_miou_focused.py')
