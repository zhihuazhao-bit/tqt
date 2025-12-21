import ast
import re
import pprint

def extract_legacy_mapping(file_path):
    """从旧脚本提取 name -> (short_id, csv_orfd, csv_road3d) 的映射"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    start_idx = content.find('ABLATION_EXPERIMENTS = {')
    if start_idx == -1: return {}
    
    # 简单的括号匹配提取字典内容
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
    
    mapping = {}
    try:
        experiments = ast.literal_eval(dict_str)
        for short_id, val in experiments.items():
            name = val.get('name')
            if name:
                mapping[name] = {
                    'id': short_id,
                    'csv_orfd': val.get('csv_orfd', ''),
                    'csv_road3d': val.get('csv_road3d', '')
                }
    except Exception as e:
        print(f"Error parsing legacy dict: {e}")
        
    return mapping

def generate_short_id(cfg):
    """为没有旧 ID 的实验生成新的简短 ID"""
    # 示例: 224_EVA_SNE(OT)_P -> E224-SNE-OT-P
    parts = []
    # 1. Size
    # parts.append(str(cfg['size'])) 
    
    # 2. Backbone (D=DenseVLM, E=EVA02)
    bb = 'D' if 'DenseVLM' in cfg['weight'] else 'E'
    parts.append(f"{bb}{cfg['size']}") # D224
    
    # 3. SNE
    if not cfg['sne']:
        parts.append("Base")
    else:
        mode = cfg['sne_mode'] # proj, ot, cross_attn
        stage = cfg['sne_fusion_stage'] # backbone, pixel
        
        if mode == 'proj': 
            s_str = 'Px' if stage == 'pixel' else 'Bb'
        elif mode == 'ot':
            s_str = 'OT'
            # OT details
            if cfg.get('ot_prior'):
                s_str += 'P' # Prior
                if cfg.get('ot_prior_mode') == 'prob': s_str += 'p'
            if cfg.get('ot_softunion'): s_str += 'U'
            if cfg.get('ot_fuse_mode') == 'mean': s_str += 'm'
            if cfg.get('ot_fuse_mode') == 'max': s_str += 'x'
            if cfg.get('ot_cost_type') == 'l2': s_str += 'L2'
            if cfg.get('ot_fuse_output') is False: s_str += 'Nr' # NoRes
        elif mode == 'cross_attn':
            s_str = 'CA'
        else:
            s_str = mode
        parts.append(s_str)

    # 4. Prompt
    if cfg['prompt']:
        p_str = 'P'
        if cfg.get('prompt_cls_mode') == 'soft': p_str += 's'
        parts.append(p_str)
    
    # 5. Others
    if not cfg.get('context_decoder', True): parts.append("NoCD")
    if cfg.get('patch_fpn'): parts.append("FPN")
    if cfg.get('patch_fpn_xsam'): parts.append("XSam")
    if cfg.get('pi_sup'): parts.append("Pi")
    
    return "-".join(parts)

def update_focused_script(target_file, legacy_mapping):
    with open(target_file, 'r') as f:
        content = f.read()
        
    start_idx = content.find('ABLATION_EXPERIMENTS = {')
    if start_idx == -1: return
    
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
        print(f"Error parsing target dict: {e}")
        return

    new_experiments = {}
    
    # 遍历当前自动生成的配置
    for _, exp_cfg in current_experiments.items():
        name = exp_cfg['name']
        
        # 1. 尝试匹配旧 ID 并迁移 CSV
        if name in legacy_mapping:
            legacy_info = legacy_mapping[name]
            new_id = legacy_info['id']
            
            # 迁移 CSV 路径 (如果新配置为空)
            if not exp_cfg.get('csv_orfd') and legacy_info['csv_orfd']:
                exp_cfg['csv_orfd'] = legacy_info['csv_orfd']
            if not exp_cfg.get('csv_road3d') and legacy_info['csv_road3d']:
                exp_cfg['csv_road3d'] = legacy_info['csv_road3d']
        else:
            # 2. 生成新 ID
            new_id = generate_short_id(exp_cfg)
            
        # 存入新字典
        new_experiments[new_id] = exp_cfg

    # 重新生成字典字符串
    new_dict_str = "ABLATION_EXPERIMENTS = " + pprint.pformat(new_experiments, indent=4, width=120, sort_dicts=False)
    
    # 替换原始内容
    new_content = content[:start_idx] + new_dict_str + content[end_idx:]
    
    with open(target_file, 'w') as f:
        f.write(new_content)
    print("Successfully finalized ablation script with short IDs and CSV paths.")

# Execute
legacy_map = extract_legacy_mapping('statis_ablation_results.py')
update_focused_script('statis_results_miou_focused.py', legacy_map)
