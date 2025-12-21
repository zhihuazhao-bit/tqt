import os
import glob
import ast

def parse_config(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 提取变量的简单且鲁棒的方法：执行代码提取局部变量
    # 注意：为了安全和避免依赖缺失，我们只提取简单的赋值语句，或者使用 exec 但 mock 掉 _base_ 和 mmcv 等依赖
    
    # 使用 ast 解析更安全
    tree = ast.parse(content)
    config = {}
    
    # 预设默认值
    defaults = {
        'model_name': 'Unknown',
        'use_sne': False,
        'sne_fusion_stage': '-',
        'sne_fusion_mode': '-',
        'ot_use_score_prior': None,
        'ot_score_prior_mode': None,
        'ot_cost_type': None,
        'ot_fuse_mode': None,
        'ot_fuse_output': None,
        'ot_softunion': None,
        'prompt_cls': False,
        'prompt_cls_mode': 'hard', # default
        'use_context_decoder': True,
        'patch_fpn': False,
        'patch_fpn_xsam': False,
        'supervise_ot_pi': False,
        'img_size': (224, 224),
        'exp_name': os.path.basename(file_path).replace('.py', ''),
        'force_reg_e0_eval': False
    }
    config.update(defaults)

    # 简单的文本行匹配提取，比 AST 更能容忍 import 错误
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        
        # 处理赋值语句
        if '=' in line:
            parts = line.split('=')
            key = parts[0].strip()
            value_str = '='.join(parts[1:]).strip().split('#')[0].strip() # 去掉注释
            
            if key in config:
                try:
                    # 尝试解析字面量
                    val = ast.literal_eval(value_str)
                    config[key] = val
                except:
                    pass # 忽略复杂表达式

    # 特殊处理 model 字典中的嵌套值 (如果顶层没有定义)
    # 这里主要依赖顶层定义的变量，因为配置文件通常是 
    # use_sne = True
    # model = dict(..., use_sne=use_sne, ...)
    # 所以解析顶层变量通常足够。
    
    return config

def generate_entry(file_path, dataset_type):
    cfg = parse_config(file_path)
    
    # 生成简短 ID (需要手动或半自动映射，这里生成建议的 ID)
    # 这里的 ID 生成逻辑仅供参考，最终我会手动调整以匹配现有 ID
    short_id = cfg['exp_name'] 
    
    # 生成描述
    size = cfg['img_size'][0] if isinstance(cfg['img_size'], tuple) else 224
    backbone = 'EVA02' if 'EVA02' in cfg['model_name'] else 'DenseVLM'
    if 'densevlm' in str(cfg.get('pretrained_weight', '')).lower(): 
        backbone = 'DenseVLM' # 修正 backbone 判断
        
    desc_parts = [f"{size}", backbone]
    
    if cfg['use_sne']:
        sne_str = f"SNE({cfg['sne_fusion_stage']}-{cfg['sne_fusion_mode']})"
        if cfg['sne_fusion_mode'] == 'ot':
            prior = 'T' if cfg['ot_use_score_prior'] else 'F'
            sne_str += f"[Prior={prior}"
            if cfg['ot_use_score_prior']:
                if cfg['ot_score_prior_mode'] == 'prob': sne_str += ",Prob"
                if cfg['ot_softunion']: sne_str += ",SoftU"
            
            if cfg['ot_cost_type'] and cfg['ot_cost_type'] != 'cos': sne_str += f",{cfg['ot_cost_type']}"
            if cfg['ot_fuse_mode'] and cfg['ot_fuse_mode'] != 'proj': sne_str += f",{cfg['ot_fuse_mode']}"
            sne_str += "]"
        desc_parts.append(sne_str)
    else:
        desc_parts.append("NoSNE")
        
    if cfg['prompt_cls']:
        p_str = "Prompt"
        if cfg.get('prompt_cls_mode') == 'soft': p_str += "(Soft)"
        desc_parts.append(p_str)
    else:
        desc_parts.append("NoPrompt")
        
    if not cfg['use_context_decoder']:
        desc_parts.append("NoCtxDec")
        
    if cfg['patch_fpn']:
        desc_parts.append("PFPN")
    if cfg['patch_fpn_xsam']:
        desc_parts.append("PFPN(XSam)")
        
    if cfg['supervise_ot_pi']: # 注意配置文件里变量名可能叫 pi_sup 或 supervise_ot_pi
        desc_parts.append("PiSup")

    desc = " + ".join(desc_parts)
    
    # 构造字典条目字符串
    entry = f"    '{cfg['exp_name']}': {{\n"
    entry += f"        'name': '{cfg['exp_name']}',\n"
    entry += f"        'desc': '{desc}',\n"
    entry += f"        'size': {size},\n"
    entry += f"        'weight': '{backbone}',\n"
    entry += f"        'sne': {cfg['use_sne']},\n"
    entry += f"        'sne_fusion_stage': '{cfg['sne_fusion_stage']}',\n"
    entry += f"        'sne_mode': '{cfg['sne_fusion_mode']}',\n"
    
    if cfg['use_sne'] and cfg['sne_fusion_mode'] == 'ot':
        entry += f"        'ot_prior': {cfg['ot_use_score_prior']},\n"
        if cfg['ot_score_prior_mode']: entry += f"        'ot_prior_mode': '{cfg['ot_score_prior_mode']}',\n"
        if cfg['ot_cost_type']: entry += f"        'ot_cost_type': '{cfg['ot_cost_type']}',\n"
        if cfg['ot_fuse_mode']: entry += f"        'ot_fuse_mode': '{cfg['ot_fuse_mode']}',\n"
        if cfg['ot_fuse_output'] is not None: entry += f"        'ot_fuse_output': {cfg['ot_fuse_output']},\n"
        if cfg['ot_softunion'] is not None: entry += f"        'ot_softunion': {cfg['ot_softunion']},\n"

    entry += f"        'prompt': {cfg['prompt_cls']},\n"
    if cfg['prompt_cls']:
         entry += f"        'prompt_cls_mode': '{cfg['prompt_cls_mode']}',\n"

    entry += f"        'context_decoder': {cfg['use_context_decoder']},\n"
    
    if cfg['patch_fpn']: entry += f"        'patch_fpn': True,\n"
    if cfg['patch_fpn_xsam']: entry += f"        'patch_fpn_xsam': True,\n"
    if cfg['supervise_ot_pi']: entry += f"        'pi_sup': True,\n"
    if cfg['force_reg_e0_eval']: entry += f"        'reg_e0_eval': True,\n"

    # 根据数据集类型预填 CSV 键
    if dataset_type == 'orfd':
        entry += f"        'csv_orfd': '', # Auto-search\n"
        entry += f"        'csv_road3d': '',\n"
    else:
        entry += f"        'csv_orfd': '',\n"
        entry += f"        'csv_road3d': '', # Auto-search\n"
        
    entry += "    },"
    return entry

# Main execution
files_orfd = sorted(glob.glob('configs/ablation/*.py'))
files_road = sorted(glob.glob('configs/ablation_road/*.py'))

print("ABLATION_EXPERIMENTS = {")
for f in files_orfd:
    print(generate_entry(f, 'orfd'))
for f in files_road:
    print(generate_entry(f, 'road3d'))
print("}")
