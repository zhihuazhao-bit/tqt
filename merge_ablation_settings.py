import ast
import re

# 1. 提取旧脚本中的 CSV 路径
def extract_old_csv_paths(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 提取 ABLATION_EXPERIMENTS 字典
    # 假设它是文件中定义的一个大字典
    match = re.search(r'ABLATION_EXPERIMENTS\s*=\s*(\{.*?\})\n', content, re.DOTALL)
    if not match:
        # 如果 regex 失败（可能因为字典太大或格式不规则），尝试使用 ast 解析
        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'ABLATION_EXPERIMENTS':
                            # 找到了，但提取值比较麻烦，因为可能有变量引用
                            # 这里简单起见，使用 exec 执行该文件的一部分（只包含字典定义的部分）
                            # 或者更简单：直接 regex 提取字典中的 key 和 csv_orfd/csv_road3d
                            pass
        except:
            pass
            
    # 正则表达式提取 key 和 csv 路径
    # pattern: 'ExpID': { ... 'csv_orfd': 'path', ... }
    # 这种方式比较脆弱，最好是 ast.literal_eval
    
    # 为了稳健，我们手动解析该字典的每一项
    exp_data = {}
    
    # 将字典内容隔离出来
    start_idx = content.find('ABLATION_EXPERIMENTS = {')
    if start_idx == -1: return {}
    
    # 寻找配对的闭合括号 (简单统计)
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
        # 使用 ast.literal_eval 解析字典字符串
        experiments = ast.literal_eval(dict_str)
        for key, val in experiments.items():
            exp_data[key] = {
                'csv_orfd': val.get('csv_orfd', ''),
                'csv_road3d': val.get('csv_road3d', '')
            }
    except Exception as e:
        print(f"Error parsing old dict: {e}")
        
    return exp_data

# 2. 读取 tqt_eva_clip.py 中的默认值 (硬编码提取)
# 根据 tqt_eva_clip.py 的 __init__ 函数
DEFAULTS = {
    'use_sne': False,
    'sne_fusion_stage': 'backbone',
    'sne_fusion_mode': 'proj',
    'use_ot_align': False,
    'ot_use_score_prior': True,
    'ot_score_prior_mode': 'argmax',
    'ot_score_prior_temperature': 1.0,
    'ot_cost_type': 'cos',
    'ot_fuse_output': True,
    'ot_fuse_mode': 'proj',
    'ot_softunion': False,
    'prompt_cls': False,
    'prompt_cls_mode': 'hard', # default 'hard' based on previous context
    'use_context_decoder': True,
    'use_learnable_prompt': True,
    'patch_fpn': False,
    'patch_fpn_xsam': False,
    'supervise_ot_pi': False,
    'force_reg_e0_eval': True
}

# 3. 读取并更新 statis_results_miou_focused.py
def update_focused_script(file_path, old_paths):
    with open(file_path, 'r') as f:
        content = f.read()
        
    # 同样提取 ABLATION_EXPERIMENTS
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
        print(f"Error parsing new dict: {e}")
        return

    # 更新逻辑
    for exp_id, exp_cfg in current_experiments.items():
        # 1. 迁移 CSV 路径
        if exp_id in old_paths:
            old_orfd = old_paths[exp_id]['csv_orfd']
            old_road = old_paths[exp_id]['csv_road3d']
            
            # 只有当新路径为空时才迁移，避免覆盖可能的新值
            if not exp_cfg.get('csv_orfd') and old_orfd:
                exp_cfg['csv_orfd'] = old_orfd
            if not exp_cfg.get('csv_road3d') and old_road:
                exp_cfg['csv_road3d'] = old_road
        
        # 2. 填充默认值
        if exp_cfg.get('sne'): # 只有启用 SNE 时才填充 SNE 相关默认值
            if 'sne_mode' not in exp_cfg or not exp_cfg['sne_mode']:
                exp_cfg['sne_mode'] = DEFAULTS['sne_fusion_mode']
            if 'sne_fusion_stage' not in exp_cfg or not exp_cfg['sne_fusion_stage']:
                exp_cfg['sne_fusion_stage'] = DEFAULTS['sne_fusion_stage']
            
            if exp_cfg.get('sne_mode') == 'ot':
                if 'ot_prior' not in exp_cfg: exp_cfg['ot_prior'] = DEFAULTS['ot_use_score_prior']
                if 'ot_prior_mode' not in exp_cfg: exp_cfg['ot_prior_mode'] = DEFAULTS['ot_score_prior_mode']
                if 'ot_cost_type' not in exp_cfg: exp_cfg['ot_cost_type'] = DEFAULTS['ot_cost_type']
                if 'ot_fuse_mode' not in exp_cfg: exp_cfg['ot_fuse_mode'] = DEFAULTS['ot_fuse_mode']
                if 'ot_fuse_output' not in exp_cfg: exp_cfg['ot_fuse_output'] = DEFAULTS['ot_fuse_output']
                if 'ot_softunion' not in exp_cfg: exp_cfg['ot_softunion'] = DEFAULTS['ot_softunion']
        
        if exp_cfg.get('prompt'):
             if 'prompt_cls_mode' not in exp_cfg: exp_cfg['prompt_cls_mode'] = DEFAULTS['prompt_cls_mode']

    # 重新生成字典字符串
    import pprint
    new_dict_str = "ABLATION_EXPERIMENTS = " + pprint.pformat(current_experiments, indent=4, width=120, sort_dicts=False)
    
    # 替换原始内容
    new_content = content[:start_idx] + new_dict_str + content[end_idx:]
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("Successfully updated ablation settings.")

# Execute
old_paths = extract_old_csv_paths('statis_ablation_results.py')
update_focused_script('statis_results_miou_focused.py', old_paths)
