#!/bin/bash
# 消融实验测试脚本 (Road3D 数据集) - 串行运行所有测试
#
# 使用方法:
#   1. 修改下方 CKPT_* 变量，填入各实验的 checkpoint 路径
#   2. 设置 USE_TRAIN_SET=true 可在训练集上评估 (默认 false 在测试集评估)
#   3. 运行: bash test_all_ablation_road.sh
#   4. 测试完成后，运行 python statis_ablation_results.py 统计结果

GPU=0
USE_TRAIN_SET=false  # 设为 true 则在训练集上评估

# ============================================================================
# 请在此处填入各实验的 checkpoint 路径 (Road3D 数据集)
# ============================================================================

# A1: 224 + EVA02 + NoSNE + NoPrompt
CKPT_A1="/root/tqdm/work_dirs/ablation_224_eva02_nosne_noprompt_road/20251203_2312/exp_224_eva02_nosne_noprompt_road/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_A1="configs/ablation_road/exp_224_eva02_nosne_noprompt_road.py"

# A2: 512 + EVA02 + NoSNE + NoPrompt  
CKPT_A2="/root/tqdm/work_dirs/ablation_512_eva02_nosne_noprompt_road/20251203_2313/exp_512_eva02_nosne_noprompt_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_A2="configs/ablation_road/exp_512_eva02_nosne_noprompt_road.py"

# B2: 224 + DenseVLM + NoSNE + NoPrompt
CKPT_B2="/root/tqdm/work_dirs/ablation_224_densevlm_nosne_noprompt_road/20251203_2311/exp_224_densevlm_nosne_noprompt_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_B2="configs/ablation_road/exp_224_densevlm_nosne_noprompt_road.py"

# C2: 224 + DenseVLM + SNE + NoPrompt
CKPT_C2="/root/tqdm/work_dirs/ablation_224_densevlm_sne_noprompt_road/20251203_2311/exp_224_densevlm_sne_noprompt_road/best_mIoU_iter_2000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_C2="configs/ablation_road/exp_224_densevlm_sne_noprompt_road.py"

# D2: 224 + DenseVLM + SNE + Prompt ## 
CKPT_D2="/root/tqdm/work_dirs/ablation_224_densevlm_sne_prompt_road/20251203_2312/exp_224_densevlm_sne_prompt_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_D2="configs/ablation_road/exp_224_densevlm_sne_prompt_road.py"

# E1: 512 + DenseVLM + SNE + Prompt ##
CKPT_E1="/root/tqdm/work_dirs/ablation_512_densevlm_sne_prompt_road/20251203_2312/exp_512_densevlm_sne_prompt_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_E1="configs/ablation_road/exp_512_densevlm_sne_prompt_road.py"

# E2: 512 + DenseVLM + NoSNE + Prompt
CKPT_E2=""  # TODO: 填入 checkpoint 路径
CONFIG_E2="configs/ablation_road/exp_512_densevlm_nosne_prompt_road.py"

# F1a: 224 + DenseVLM + SNE(OT, prior=False) + NoPrompt
CKPT_F1a=""  # TODO: 填入 checkpoint 路径
CONFIG_F1a="configs/ablation_road/exp_224_densevlm_sneotFalse_noprompt_road.py"

# F1b: 224 + DenseVLM + SNE(OT, prior=True) + NoPrompt
CKPT_F1b=""  # TODO: 填入 checkpoint 路径
CONFIG_F1b="configs/ablation_road/exp_224_densevlm_sneotTrue_noprompt_road.py"

# F2a: 224 + EVA02 + SNE(OT, prior=False) + NoPrompt
CKPT_F2a=""  # TODO: 填入 checkpoint 路径
CONFIG_F2a="configs/ablation_road/exp_224_eva02_sneotFalse_noprompt_road.py"

# F2b: 224 + EVA02 + SNE(OT, prior=True) + NoPrompt
CKPT_F2b=""  # TODO: 填入 checkpoint 路径
CONFIG_F2b="configs/ablation_road/exp_224_eva02_sneotTrue_noprompt_road.py"

# F2p-mean-cos: 224 + EVA02 + SNE(OT, prior=True, cos, mean) + Patch-FPN + piSup + Prompt
CKPT_F2p_mean_cos="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/20251215_1728/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2p_mean_cos="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road.py"

# F2p-mean-cos-512: 512 + EVA02 + SNE(OT, prior=True, cos, mean) + Patch-FPN + piSup + Prompt
CKPT_F2p_mean_cos_512="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/20251216_1157/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/best_mIoU_iter_3000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2p_mean_cos_512="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road.py"

# F2p-mean-cos-soft-512: 512 + EVA02 + SNE(OT, prior=True, cos, mean) + Patch-FPN + piSup + Prompt(Soft)
CKPT_F2pSoft_mean_cos_soft_512="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_road/20251217_1300/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_road/best_mIoU_iter_3000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2pSoft_mean_cos_soft_512="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_road.py"

# F2p-mean-prob-soft: 224 + EVA02 + SNE(OT, prior=prob+softunion, cos, mean) + Patch-FPN + piSup + Prompt
CKPT_F2p_mean_prob_soft="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/20251215_1727/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2p_mean_prob_soft="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road.py"

# F2p-mean-prob-soft-512: 512 + EVA02 + SNE(OT, prior=prob+softunion, cos, mean) + Patch-FPN + piSup + Prompt
CKPT_F2p_mean_prob_soft_512="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/20251216_1157/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2p_mean_prob_soft_512="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road.py"

# F2pSoft-learnableT-promptTau-224: 224 + EVA02 + SNE(OT, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + piSup + Prompt(Soft, Tau)
CKPT_F2pSoft_learnableT_promptTau_224="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251220_0944/exp_224_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_5000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_learnableT_promptTau_224="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# F2pSoft-learnableT-promptTau-1024: 1024 + EVA02 + SNE(OT, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + piSup + Prompt(Soft, Tau)
CKPT_F2pSoft_learnableT_promptTau_1024="/root/tqdm/work_dirs/ablation_1024_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251220_2253/exp_1024_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_5000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_learnableT_promptTau_1024="configs/ablation_road/exp_1024_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# F2p-mean-prob-soft-0.5-512: 512 + EVA02 + SNE(OT, prior=prob+softunion, T=0.5, cos, mean) + Patch-FPN + piSup + Prompt(Soft)
CKPT_F2pSoft_mean_prob_soft_512_T05="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_0.5/20251217_1304/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_0.5/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2pSoft_mean_prob_soft_512_T05="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_0.5.py"

# F2p-mean-prob-soft-0.5-512-hard: 512 + EVA02 + SNE(OT, prior=prob+softunion, T=0.5, cos, mean) + Patch-FPN + piSup + Prompt(Hard)
CKPT_F2pHard_mean_prob_soft_512_T05="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road_0.5/20251217_1304/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road_0.5/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2pHard_mean_prob_soft_512_T05="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road_0.5.py"

# F2c+pi-mean-cos: 224 + EVA02 + SNE(OT, prior=True, cos, mean) + Patch-FPN + piSup (NoPrompt)
CKPT_F2c_pi_mean_cos="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road/20251215_1727/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2c_pi_mean_cos="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road.py"

# F2c+pi-mean-cos-prob-softunion: 224 + EVA02 + SNE(OT, prior=prob+softunion, cos, mean) + Patch-FPN + piSup (NoPrompt)
CKPT_F2c_pi_mean_prob_soft="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road/20251215_1726/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2c_pi_mean_prob_soft="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road.py"

# F2pSoft-learnableT-promptTau: 512 + EVA02 + SNE(OT, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + piSup + Prompt(Soft, uses Tau)
CKPT_F2pSoft_learnableT_promptTau_1="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251218_1243/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_4000.pth" # TODO: 填入 checkpoint 路径
CKPT_F2pSoft_learnableT_promptTau_0_1="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251218_1320/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_4000.pth" # TODO: 填入 checkpoint 路径
CKPT_F2pSoft_learnableT_promptTau_0_1x4="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251227_2212x4/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_5000.pth" # TODO: 填入 checkpoint 路径
CONFIG_F2pSoft_learnableT_promptTau="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# F2pSoft-learnableT-promptTau-linear: 512 + EVA02 + PromptCls(linear)
CKPT_F2pSoft_learnableT_promptTau_linear="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_linear/20251221_1432/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_linear/best_mIoU_iter_5000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_learnableT_promptTau_linear="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_linear.py"

# F2pSoft-learnableT-promptTau-linear_text: 512 + EVA02 + PromptCls(linear_text)
CKPT_F2pSoft_learnableT_promptTau_linear_text="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_linear_text/20251221_1432/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_linear_text/best_mIoU_iter_3000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_learnableT_promptTau_linear_text="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_linear_text.py"

# F2pSoft-fixedT0.01-promptTau: 512 + EVA02 + Fixed OT T=0.01 + Learnable Tau
CKPT_F2pSoft_fixedT0_01_road="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_fixedT0.01_promptTau/20251222_2205/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_fixedT0.01_promptTau/best_mIoU_iter_4000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_fixedT0_01_road="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_fixedT0.01_promptTau.py"

# F2pSoft-config: 512 + EVA02 + SNE(OT, prior=prob, config fusion) + 无TopK
CKPT_F2pSoft_config_road="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_road_learnableT_promptTau/20251222_1646/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_2000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_config_road="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_road_learnableT_promptTau.py"

# F2pSoft-config-topk: 512 + EVA02 + SNE(OT, prior=prob, config fusion) + TopK(10/3)
CKPT_F2pSoft_config_topk_road="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_road_learnableT_promptTau_topk/20251222_1651/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_road_learnableT_promptTau_topk/best_mIoU_iter_2000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_config_topk_road="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_road_learnableT_promptTau_topk.py"

# F2pSoft-proj-learnableT-promptTau: 512 + EVA02 + SNE(OT, prior=prob, Learnable T, cos, proj, softunion) + Patch-FPN + piSup + Prompt(Soft, uses Tau)
CKPT_F2pSoft_proj_learnableT_promptTau="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_proj_prob_softunion_road_learnableT_promptTau/20251218_1243/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_proj_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_1000.pth" # TODO: 填入 checkpoint 路径
CONFIG_F2pSoft_proj_learnableT_promptTau="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_proj_prob_softunion_road_learnableT_promptTau.py"

# DenseVLM-learnableT-promptTau-road: 512 + DenseVLM + SNE(OT, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + piSup + Prompt(Soft, Tau)
CKPT_DenseVLM_learnableT_promptTau_road="/root/tqdm/work_dirs/ablation_512_densevlm_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251222_1115/exp_512_densevlm_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_4000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_DenseVLM_learnableT_promptTau_road="configs/ablation_road/exp_512_densevlm_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# CrossAttn-learnableT-promptTau-road: 512 + EVA02 + SNE(backbone-cross_attn) + Patch-FPN + Prompt(Soft, Tau)
CKPT_Proj_learnableT_promptTau_road="/root/tqdm/work_dirs/ablation_512_eva02_sneProjTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251222_1145/exp_512_eva02_sneProjTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_1000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_Proj_learnableT_promptTau_road="configs/ablation_road/exp_512_eva02_sneProjTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# LearnableOnly-road: 512 + EVA02 + learnable_only (仅可学习掩码向量，无文本编码)
CKPT_LearnableOnly_road="/root/tqdm/work_dirs/ablation_512_eva02_learnable_only_road/20251224_0723/exp_512_eva02_learnable_only_road/best_mIoU_iter_4000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_LearnableOnly_road="configs/ablation_road/exp_512_eva02_learnable_only_road.py"

# LearnableOnly + M2F Decoder (Road3D): 512 + EVA02 + learnable_only + 标准 Mask2Former pixel decoder
CKPT_LearnableOnly_M2F_road="/root/tqdm/work_dirs/ablation_512_eva02_learnable_only_m2f_decoder_road/20251224_2157/exp_512_eva02_learnable_only_m2f_decoder_road/best_mIoU_iter_4000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_LearnableOnly_M2F_road="configs/ablation_road/exp_512_eva02_learnable_only_m2f_decoder_road.py"

# SNE-Proj + LearnableOnly + M2F Decoder (Road3D): 512 + EVA02 + SNE(proj) + learnable_only + 标准 Mask2Former pixel decoder
CKPT_SNEProj_LearnableOnly_M2F_road="/root/tqdm/work_dirs/ablation_512_eva02_sneProj_learnable_only_m2f_decoder_road/20251224_2303/exp_512_eva02_sneProj_learnable_only_m2f_decoder_road/best_mIoU_iter_5000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_SNEProj_LearnableOnly_M2F_road="configs/ablation_road/exp_512_eva02_sneProj_learnable_only_m2f_decoder_road.py"

# SNE-OT + LearnableOnly-road: 512 + EVA02 + SNE(OT) + Patch-FPN + piSup + learnable_only (保留OT，无文本编码)
CKPT_SNEOT_LearnableOnly_road="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_learnable_only_road/20251224_0724/exp_512_eva02_sneotTrue_patchfpn_pisup_learnable_only_road/best_mIoU_iter_4000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_SNEOT_LearnableOnly_road="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_learnable_only_road.py"

# Material Classification-road: 512 + EVA02 + SNE(OT) + Patch-FPN + piSup + Prompt(Soft, Tau) + Material (场景库 24 -> 240)
CKPT_Material_road="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_material/20251224_1301/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_material/best_mIoU_iter_3000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_Material_road="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau_material.py"

# LearnableOnly + M2F Decoder + No Pretrain (Road3D): 512 + EVA02 + learnable_only + 标准 Mask2Former pixel decoder + 无预训练权重
CKPT_LearnableOnly_M2F_NoPretrain_road="/root/tqdm/work_dirs/ablation_512_eva02_learnable_only_m2f_decoder_no_pretrain_road/20251225_1726/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain_road/best_mIoU_iter_5000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_LearnableOnly_M2F_NoPretrain_road="configs/ablation_road/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain_road.py"

# LearnableOnly + M2F Decoder + No Score Map Reg (Road3D): 512 + EVA02 + learnable_only + 标准 Mask2Former pixel decoder + 无 score_map 正则化
CKPT_LearnableOnly_M2F_NoScoreMapReg_road="/root/tqdm/work_dirs/ablation_512_eva02_learnable_only_m2f_decoder_no_score_map_reg_road/20251225_1829/exp_512_eva02_learnable_only_m2f_decoder_no_score_map_reg_road/best_mIoU_iter_4000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_LearnableOnly_M2F_NoScoreMapReg_road="configs/ablation_road/exp_512_eva02_learnable_only_m2f_decoder_no_score_map_reg_road.py"

# ============================================================================
# 预训练权重消融实验 (8个配置) - Road3D
# ============================================================================

# P1: SNE Proj + Learnable Only + No Pretrain (text=F, img=F)
CKPT_P1_road="/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_no_pretrain_road/20251226_1231/exp_512_eva02_sneProj_m2f_no_pretrain_road/best_mIoU_iter_5000.pth"
CONFIG_P1_road="configs/ablation_road/exp_512_eva02_sneProj_m2f_no_pretrain_road.py"

# P2: SNE Proj + Text Pretrain (text=T, img=F, prompt_cls=F)
CKPT_P2_road="/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_text_pretrain_road/20251226_1236/exp_512_eva02_sneProj_m2f_text_pretrain_road/best_mIoU_iter_5000.pth"
CONFIG_P2_road="configs/ablation_road/exp_512_eva02_sneProj_m2f_text_pretrain_road.py"

# P3: SNE Proj + Full Pretrain (text=T, img=T, prompt_cls=F)
CKPT_P3_road="/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_full_pretrain_road/20251226_1238/exp_512_eva02_sneProj_m2f_full_pretrain_road/best_mIoU_iter_5000.pth"
CONFIG_P3_road="configs/ablation_road/exp_512_eva02_sneProj_m2f_full_pretrain_road.py"

# P4: SNE Proj + Full Pretrain + PromptCls (text=T, img=T, prompt_cls=T)
CKPT_P4_road="/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_full_pretrain_promptcls_road/20251226_1240/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls_road/best_mIoU_iter_4000.pth"
CONFIG_P4_road="configs/ablation_road/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls_road.py"

# P5: SNE Proj + Text Pretrain + PromptCls (text=T, img=F, prompt_cls=T)
CKPT_P5_road="/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_text_pretrain_promptcls_road/20251226_1240/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls_road/best_mIoU_iter_5000.pth"
CONFIG_P5_road="configs/ablation_road/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls_road.py"

# P6: SNE OT + Patch-FPN + Text Pretrain + PromptCls (text=T, img=F)
CKPT_P6_road="/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_text_pretrain_road/20251226_1248/exp_512_eva02_sneot_patchfpn_text_pretrain_road/best_mIoU_iter_5000.pth"
CONFIG_P6_road="configs/ablation_road/exp_512_eva02_sneot_patchfpn_text_pretrain_road.py"

# P7: SNE OT + Patch-FPN + Text Pretrain + No PromptCls (text=T, img=F, prompt_cls=F)
CKPT_P7_road="/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls_road/20251226_1248/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls_road/best_mIoU_iter_5000.pth"
CONFIG_P7_road="configs/ablation_road/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls_road.py"

# P8: SNE OT + Patch-FPN + Full Pretrain + No PromptCls (text=T, img=T, prompt_cls=F)
CKPT_P8_road="/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls_road/20251226_1704/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls_road/best_mIoU_iter_4000.pth"
CONFIG_P8_road="configs/ablation_road/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls_road.py"

# ============================================================================

echo "=========================================="
echo "消融实验测试 - 串行运行"
echo "GPU: $GPU"
echo "=========================================="

# 测试函数
# 用法: run_test EXP_ID CONFIG CHECKPOINT [USE_TRAIN]
#   USE_TRAIN: 可选，传入 "train" 则在训练集上评估，覆盖全局 USE_TRAIN_SET
run_test() {
    local EXP_ID=$1
    local CONFIG=$2
    local CHECKPOINT=$3
    local USE_TRAIN=${4:-}  # 可选第4参数

    # 判断是否使用训练集：优先使用第4参数，否则使用全局变量
    local USE_TRAIN_FLAG=$USE_TRAIN_SET
    if [ "$USE_TRAIN" = "train" ]; then
        USE_TRAIN_FLAG=true
    fi

    # 自动从 checkpoint 路径获取保存目录
    local SAVE_DIR=$(dirname "$CHECKPOINT")/test_results/
    if [ "$USE_TRAIN_FLAG" = true ]; then
        SAVE_DIR=$(dirname "$CHECKPOINT")/train_results/
    fi

    echo ""
    echo "=========================================="
    echo "[$EXP_ID] 开始测试"
    echo "Config: $CONFIG"
    echo "Checkpoint: $CHECKPOINT"
    echo "Save Dir: $SAVE_DIR"
    echo "Use Train Set: $USE_TRAIN_FLAG"
    echo "=========================================="

    if [ ! -f "$CHECKPOINT" ]; then
        echo "警告: Checkpoint 文件不存在: $CHECKPOINT, 跳过此实验"
        return 1
    fi

    # 构建额外参数
    local EXTRA_ARGS=""
    if [ "$USE_TRAIN_FLAG" = true ]; then
        EXTRA_ARGS="--use-train-set"
    fi

    CUDA_VISIBLE_DEVICES=$GPU python test.py \
        --config $CONFIG \
        --checkpoint $CHECKPOINT \
        --eval mIoU mFscore \
        --show-dir $SAVE_DIR \
        --save_dir $SAVE_DIR \
        $EXTRA_ARGS

    echo "[$EXP_ID] 测试完成!"
    echo "CSV 保存位置: $SAVE_DIR"
}

# 串行运行所有测试
# run_test "A1" "$CONFIG_A1" "$CKPT_A1"
# run_test "A2" "$CONFIG_A2" "$CKPT_A2"
# run_test "B2" "$CONFIG_B2" "$CKPT_B2"
# run_test "C2" "$CONFIG_C2" "$CKPT_C2"
# run_test "D2" "$CONFIG_D2" "$CKPT_D2"
# run_test "E1" "$CONFIG_E1" "$CKPT_E1"
# run_test "E2" "$CONFIG_E2" "$CKPT_E2"
# run_test "F1a" "$CONFIG_F1a" "$CKPT_F1a"
# run_test "F1b" "$CONFIG_F1b" "$CKPT_F1b"
# run_test "F2a" "$CONFIG_F2a" "$CKPT_F2a"
# run_test "F2b" "$CONFIG_F2b" "$CKPT_F2b"
# run_test "F2p-mean-cos" "$CONFIG_F2p_mean_cos" "$CKPT_F2p_mean_cos"
# run_test "F2p-mean-cos-512" "$CONFIG_F2p_mean_cos_512" "$CKPT_F2p_mean_cos_512"
# run_test "F2p-mean-cos-soft-512" "$CONFIG_F2p_mean_cos_soft_512" "$CKPT_F2p_mean_cos_soft_512"
# run_test "F2p-mean-prob-soft" "$CONFIG_F2p_mean_prob_soft" "$CKPT_F2p_mean_prob_soft"
# run_test "F2p-mean-prob-soft-512" "$CONFIG_F2p_mean_prob_soft_512" "$CKPT_F2p_mean_prob_soft_512"
# run_test "F2p-mean-prob-soft-0.5-512" "$CONFIG_F2p_mean_prob_soft_512_T05" "$CKPT_F2p_mean_prob_soft_512_T05"
# run_test "F2p-mean-prob-soft-0.5-512-hard" "$CONFIG_F2p_mean_prob_soft_512_T05_hard" "$CKPT_F2p_mean_prob_soft_512_T05_hard"
# run_test "F2pSoft-learnableT-promptTau-224" "$CONFIG_F2pSoft_learnableT_promptTau_224" "$CKPT_F2pSoft_learnableT_promptTau_224"
# run_test "F2c+pi-mean-cos" "$CONFIG_F2c_pi_mean_cos" "$CKPT_F2c_pi_mean_cos"
# run_test "F2c+pi-mean-cos-prob-softunion" "$CONFIG_F2c_pi_mean_prob_soft" "$CKPT_F2c_pi_mean_prob_soft"
# run_test "F2pSoft-mean-prob-soft-0.5-512" "$CONFIG_F2pSoft_mean_prob_soft_512_T05" "$CKPT_F2pSoft_mean_prob_soft_512_T05"
# run_test "F2pHard-mean-prob-soft-0.5-512" "$CONFIG_F2pHard_mean_prob_soft_512_T05" "$CKPT_F2pHard_mean_prob_soft_512_T05"
# run_test "F2p-mean-cos-soft-512" "$CONFIG_F2pSoft_mean_cos_soft_512" "$CKPT_F2pSoft_mean_cos_soft_512"
# run_test "F2pSoft-learnableT-promptTau" "$CONFIG_F2pSoft_learnableT_promptTau" "$CKPT_F2pSoft_learnableT_promptTau_1"
# run_test "F2pSoft-learnableT-promptTau" "$CONFIG_F2pSoft_learnableT_promptTau" "$CKPT_F2pSoft_learnableT_promptTau_0_1" "train"
run_test "F2pSoft-learnableT-promptTau" "$CONFIG_F2pSoft_learnableT_promptTau" "$CKPT_F2pSoft_learnableT_promptTau_0_1x4"
# run_test "F2pSoft-proj-learnableT-promptTau" "$CONFIG_F2pSoft_proj_learnableT_promptTau" "$CKPT_F2pSoft_proj_learnableT_promptTau"

# run_test "F2pSoft-learnableT-promptTau-linear-road" "$CONFIG_F2pSoft_learnableT_promptTau_linear" "$CKPT_F2pSoft_learnableT_promptTau_linear"
# run_test "F2pSoft-learnableT-promptTau-linear_text-road" "$CONFIG_F2pSoft_learnableT_promptTau_linear_text" "$CKPT_F2pSoft_learnableT_promptTau_linear_text"
## 1024 分辨率
# run_test "F2pSoft-learnableT-promptTau-1024" "$CONFIG_F2pSoft_learnableT_promptTau_1024" "$CKPT_F2pSoft_learnableT_promptTau_1024"

# Fixed OT Temperature 消融实验
# run_test "F2pSoft-fixedT0.01-promptTau-road" "$CONFIG_F2pSoft_fixedT0_01_road" "$CKPT_F2pSoft_fixedT0_01_road"

# Config fusion 消融实验
# run_test "F2pSoft-config-road" "$CONFIG_F2pSoft_config_road" "$CKPT_F2pSoft_config_road"
# run_test "F2pSoft-config-topk-road" "$CONFIG_F2pSoft_config_topk_road" "$CKPT_F2pSoft_config_topk_road"
# run_test "F2pSoft-learnableT-promptTau_224" "$CONFIG_F2pSoft_learnableT_promptTau_224" "$CKPT_F2pSoft_learnableT_promptTau_224"

# DenseVLM 和 CrossAttn 消融实验 (Road3D)
# run_test "DenseVLM-learnableT-promptTau-road" "$CONFIG_DenseVLM_learnableT_promptTau_road" "$CKPT_DenseVLM_learnableT_promptTau_road"
# run_test "Proj-learnableT-promptTau-road" "$CONFIG_Proj_learnableT_promptTau_road" "$CKPT_Proj_learnableT_promptTau_road"

# LearnableOnly 消融实验 (验证文本模态必要性)
# run_test "LearnableOnly-road" "$CONFIG_LearnableOnly_road" "$CKPT_LearnableOnly_road"

# LearnableOnly + M2F Decoder 消融实验 (标准 Mask2Former pixel decoder，无文本交叉注意力)
# run_test "LearnableOnly-M2F-road" "$CONFIG_LearnableOnly_M2F_road" "$CKPT_LearnableOnly_M2F_road"

# SNE-Proj + LearnableOnly + M2F Decoder 消融实验 (SNE proj 融合 + 标准 Mask2Former pixel decoder)
# run_test "SNEProj-LearnableOnly-M2F-road" "$CONFIG_SNEProj_LearnableOnly_M2F_road" "$CKPT_SNEProj_LearnableOnly_M2F_road"

# SNE-OT + LearnableOnly 消融实验 (保留OT架构，验证文本编码器必要性)
# run_test "SNEOT-LearnableOnly-road" "$CONFIG_SNEOT_LearnableOnly_road" "$CKPT_SNEOT_LearnableOnly_road"

# Material Classification 消融实验 (场景库 24 -> 240)
# run_test "Material-road" "$CONFIG_Material_road" "$CKPT_Material_road"

# LearnableOnly + M2F Decoder + No Pretrain 消融实验 (无预训练权重，验证预训练对性能的影响)
# run_test "LearnableOnly-M2F-NoPretrain-road" "$CONFIG_LearnableOnly_M2F_NoPretrain_road" "$CKPT_LearnableOnly_M2F_NoPretrain_road"

# LearnableOnly + M2F Decoder + No Score Map Reg 消融实验 (无 score_map 正则化，验证 Vision-Language 正则化对性能的影响)
# run_test "LearnableOnly-M2F-NoScoreMapReg-road" "$CONFIG_LearnableOnly_M2F_NoScoreMapReg_road" "$CKPT_LearnableOnly_M2F_NoScoreMapReg_road"

# ============================================================================
# 预训练权重消融实验 (填入 checkpoint 后取消注释运行) - Road3D
# ============================================================================
# run_test "P1-NoPretrain-road" "$CONFIG_P1_road" "$CKPT_P1_road"
# run_test "P2-TextPretrain-road" "$CONFIG_P2_road" "$CKPT_P2_road"
# run_test "P3-FullPretrain-road" "$CONFIG_P3_road" "$CKPT_P3_road"
# run_test "P4-FullPretrain-PromptCls-road" "$CONFIG_P4_road" "$CKPT_P4_road"
# run_test "P5-TextPretrain-PromptCls-road" "$CONFIG_P5_road" "$CKPT_P5_road"
# run_test "P6-SNEOT-TextPretrain-road" "$CONFIG_P6_road" "$CKPT_P6_road"
# run_test "P7-SNEOT-TextPretrain-NoPromptCls-road" "$CONFIG_P7_road" "$CKPT_P7_road"
# run_test "P8-SNEOT-FullPretrain-NoPromptCls-road" "$CONFIG_P8_road" "$CKPT_P8_road"

echo ""
echo "=========================================="
echo "所有测试完成!"
echo ""
echo "CSV 文件位置 (在各自 checkpoint 目录的 test_results/ 下):"
echo "  查找命令: find ./work_dirs -name 'testing_eval_file_stats_*.csv'"
echo ""
echo "统计结果: python statis_ablation_results.py"
echo "=========================================="
