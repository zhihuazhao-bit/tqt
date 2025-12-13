# TQT 消融实验方案

## 实验目标
验证以下因素对可通行区域分割性能的影响：
1. **图像尺寸**: 224 vs 512
2. **预训练权重**: EVA02_CLIP_B (原始) vs densevlm_coco (微调)
3. **SNE 双模态融合**: 有 vs 无
4. **动态场景提示 (Prompt Cls)**: 有 vs 无

## 实验矩阵

| 实验ID | 配置文件 | 尺寸 | 权重 | SNE | Prompt | 说明 |
|--------|---------|------|------|-----|--------|------|
| A1 | exp_224_eva02_nosne_noprompt | 224 | EVA02 | ❌ | ❌ | 基准-小尺寸 |
| A2 | exp_512_eva02_nosne_noprompt | 512 | EVA02 | ❌ | ❌ | 基准-大尺寸 |
| B2 | exp_224_densevlm_nosne_noprompt | 224 | densevlm | ❌ | ❌ | COCO微调权重 |
| C2 | exp_224_densevlm_sne_noprompt | 224 | densevlm | ✅ backbone-proj | ❌ | +SNE |
| D2 | exp_224_densevlm_sne_prompt | 224 | densevlm | ✅ | ✅ | +Prompt |
| E1 | exp_512_densevlm_sne_prompt | 512 | densevlm | ✅ | ✅ | 完整模型 |

## 对比分析

### 1. 尺寸影响 (A1 vs A2)
- 固定: EVA02原始权重, 无SNE, 无Prompt
- 对比: 224 vs 512
- 预期: 512 应该更好（更多细节）

### 2. 预训练权重影响 (A1 vs B2)
- 固定: 224尺寸, 无SNE, 无Prompt
- 对比: EVA02原始 vs densevlm_coco微调
- 预期: densevlm 应该更好（已在分割任务上微调）

### 3. SNE 影响 (B2 vs C2)
- 固定: 224尺寸, densevlm权重, 无Prompt
- 对比: 无SNE vs 有SNE(backbone-proj)
- 预期: 有SNE应该更好（几何信息补充）

### 4. Prompt 影响 (C2 vs D2)
- 固定: 224尺寸, densevlm权重, 有SNE
- 对比: 无Prompt vs 有Prompt
- 预期: 有Prompt应该更好（场景自适应）

### 5. 完整模型 vs 基准 (A1 vs E1)
- 对比: 最简基准 vs 完整模型
- 验证: 各组件累积效果

## 运行命令

```bash
# 运行所有消融实验
python run_ablation.py --gpu 0

# 运行单个实验
python run_ablation.py --gpu 0 --exp A1

# 列出所有实验
python run_ablation.py --list
```

## 结果记录

| 实验 | mIoU | mAcc | traversable IoU | notraversable IoU | 备注 |
|------|------|------|-----------------|-------------------|------|
| A1 | | | | | |
| A2 | | | | | |
| B2 | | | | | |
| C2 | | | | | |
| D2 | | | | | |
| E1 | | | | | |
