#!/usr/bin/env python3
"""
消融实验结果统计脚本 - mIoU 聚焦版

自动收集所有消融实验的测试结果并生成合并的对比表格。
重点关注 mIoU 指标，将 Overall/Known/Unknown 的 mIoU 合并展示在同一张表中。

Usage:
    python statis_results_miou_focused.py
    python statis_results_miou_focused.py --dataset orfd
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# 从公共配置文件导入场景配置
from utils.scene_config import DATASET_UNKNOWN_SCENES, ABNORMAL_SCENES

# 复用原脚本中的辅助函数
from statis_ablation_results import (
    find_csv_for_experiment,
    analyze_csv,
    per_class_metrics_from_conf,
    macro_averages,
    freq_weighted_iou,
    getScores_self
)

# ============================================================================
# 自动生成的消融实验配置
# ============================================================================
ABLATION_EXPERIMENTS = {   'D224-Base': {   'name': 'ablation_224_densevlm_nosne_noprompt',
                     'desc': '224 + DenseVLM + NoSNE + NoPrompt',
                     'size': 224,
                     'weight': 'DenseVLM',
                     'sne': False,
                     'sne_fusion_stage': 'pixel',
                     'sne_mode': 'proj',
                     'prompt': False,
                     'context_decoder': True,
                     'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_191544.csv',
                     'csv_road3d': '/root/tqdm/work_dirs/ablation_224_densevlm_nosne_noprompt_road/20251203_2311/exp_224_densevlm_nosne_noprompt_road/test_results/exp_224_densevlm_nosne_noprompt_road/testing_eval_file_stats_20251204_110935.csv'},
    'D224-BbPr': {   'name': 'ablation_224_densevlm_sne_noprompt',
                     'desc': '224 + DenseVLM + SNE(backbone-proj) + NoPrompt',
                     'size': 224,
                     'weight': 'DenseVLM',
                     'sne': True,
                     'sne_fusion_stage': 'backbone',
                     'sne_mode': 'proj',
                     'prompt': False,
                     'context_decoder': True,
                     'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_192219.csv',
                     'csv_road3d': '/root/tqdm/work_dirs/ablation_224_densevlm_sne_noprompt_road/20251203_2311/exp_224_densevlm_sne_noprompt_road/test_results/exp_224_densevlm_sne_noprompt_road/testing_eval_file_stats_20251204_113833.csv'},
    'D224-OT': {   'name': 'ablation_224_densevlm_sneotFalse_noprompt',
                   'desc': '224 + DenseVLM + SNE(backbone-ot)[Prior=F] + NoPrompt',
                   'size': 224,
                   'weight': 'DenseVLM',
                   'sne': True,
                   'sne_fusion_stage': 'backbone',
                   'sne_mode': 'ot',
                   'ot_prior': False,
                   'prompt': False,
                   'context_decoder': True,
                   'csv_orfd': '/root/tqdm/work_dirs/ablation_224_densevlm_sneot_noprompt/20251204_1638/exp_224_densevlm_sneot_noprompt/test_results/exp_224_densevlm_sneotFalse_noprompt/testing_eval_file_stats_20251204_191728.csv',
                   'csv_road3d': '',
                   'ot_prior_mode': 'argmax',
                   'ot_cost_type': 'cos',
                   'ot_fuse_mode': 'proj',
                   'ot_fuse_output': True,
                   'ot_softunion': False},
    'D224-OTP': {   'name': 'ablation_224_densevlm_sneotTrue_noprompt',
                    'desc': '224 + DenseVLM + SNE(backbone-ot)[Prior=T] + NoPrompt',
                    'size': 224,
                    'weight': 'DenseVLM',
                    'sne': True,
                    'sne_fusion_stage': 'backbone',
                    'sne_mode': 'ot',
                    'ot_prior': True,
                    'prompt': False,
                    'context_decoder': True,
                    'csv_orfd': '/root/tqdm/work_dirs/ablation_224_densevlm_sneotTrue_noprompt/20251204_1724/exp_224_densevlm_sneotTrue_noprompt/test_results/exp_224_densevlm_sneotTrue_noprompt/testing_eval_file_stats_20251204_192342.csv',
                    'csv_road3d': '',
                    'ot_prior_mode': 'argmax',
                    'ot_cost_type': 'cos',
                    'ot_fuse_mode': 'proj',
                    'ot_fuse_output': True,
                    'ot_softunion': False},
    'D224-OTPmNr-P-FPN-Pi': {   'name': 'ablation_224_densevlm_sneotTrue_patchfpn_pisup_prompt_no_cos_mean',
                                'desc': '224 + DenseVLM + SNE(backbone-ot)[Prior=T,mean] + Prompt + PFPN + PiSup',
                                'size': 224,
                                'weight': 'DenseVLM',
                                'sne': True,
                                'sne_fusion_stage': 'backbone',
                                'sne_mode': 'ot',
                                'ot_prior': True,
                                'ot_cost_type': 'cos',
                                'ot_fuse_mode': 'mean',
                                'ot_fuse_output': False,
                                'prompt': True,
                                'prompt_cls_mode': 'hard',
                                'context_decoder': True,
                                'patch_fpn': True,
                                'pi_sup': True,
                                'csv_orfd': '/root/tqdm/work_dirs/ablation_224_densevlm_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/exp_224_densevlm_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/test_results/exp_224_densevlm_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/testing_eval_file_stats_20251215_151954.csv',
                                'csv_road3d': '',
                                'ot_prior_mode': 'argmax',
                                'ot_softunion': False},
    'D224-PxPr-P': {   'name': 'ablation_224_densevlm_sne_prompt',
                       'desc': '224 + DenseVLM + SNE(pixel-proj) + Prompt',
                       'size': 224,
                       'weight': 'DenseVLM',
                       'sne': True,
                       'sne_fusion_stage': 'pixel',
                       'sne_mode': 'proj',
                       'prompt': True,
                       'prompt_cls_mode': 'hard',
                       'context_decoder': True,
                       'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_204424.csv',
                       'csv_road3d': '/root/tqdm/work_dirs/ablation_224_densevlm_sne_prompt_road/20251203_2312/exp_224_densevlm_sne_prompt_road/test_results/exp_224_densevlm_sne_prompt_road/testing_eval_file_stats_20251204_135458.csv'},
    'E224-Base': {   'name': 'ablation_224_eva02_nosne_noprompt',
                     'desc': '224 + EVA02 + NoSNE + NoPrompt',
                     'size': 224,
                     'weight': 'EVA02',
                     'sne': False,
                     'sne_fusion_stage': 'pixel',
                     'sne_mode': 'proj',
                     'prompt': False,
                     'context_decoder': True,
                     'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_190458.csv',
                     'csv_road3d': '/root/tqdm/work_dirs/ablation_224_eva02_nosne_noprompt_road/20251203_2312/exp_224_eva02_nosne_noprompt_road/test_results/exp_224_eva02_nosne_noprompt_road/testing_eval_file_stats_20251204_101553.csv'},
    'E224-Base-NoCD': {   'name': 'ablation_224_eva02_nosne_noprompt_nocd',
                          'desc': '224 + EVA02 + NoSNE + NoPrompt + NoCtxDec',
                          'size': 224,
                          'weight': 'EVA02',
                          'sne': False,
                          'sne_fusion_stage': 'pixel',
                          'sne_mode': 'proj',
                          'prompt': False,
                          'context_decoder': False,
                          'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_nosne_noprompt_nocd/20251212_1823/exp_224_eva02_nosne_noprompt_nocd/test_results/exp_224_eva02_nosne_noprompt_nocd/testing_eval_file_stats_20251212_190055.csv',
                          'csv_road3d': ''},
    'E224-Base_v1': {   'name': 'ablation_224_eva02_nosne_noprompt_road',
                        'desc': '224 + EVA02 + NoSNE + NoPrompt',
                        'size': 224,
                        'weight': 'EVA02',
                        'sne': False,
                        'sne_fusion_stage': 'pixel',
                        'sne_mode': 'proj',
                        'prompt': False,
                        'context_decoder': True,
                        'csv_orfd': '',
                        'csv_road3d': ''},
    'E224-BbPr': {   'name': 'ablation_224_eva02_sneBackboneProj_noprompt',
                     'desc': '224 + EVA02 + SNE(backbone-proj) + NoPrompt',
                     'size': 224,
                     'weight': 'EVA02',
                     'sne': True,
                     'sne_fusion_stage': 'backbone',
                     'sne_mode': 'proj',
                     'prompt': False,
                     'context_decoder': True,
                     'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneBackboneProj_noprompt/exp_224_eva02_sneBackboneProj_noprompt/test_results/exp_224_eva02_sneBackboneProj_noprompt/testing_eval_file_stats_20251215_083836.csv',
                     'csv_road3d': ''},
    'E224-BbPr-RegE0': {   'name': 'ablation_224_eva02_sneBackboneProj_noprompt_regE0eval',
                           'desc': '224 + EVA02 + SNE(backbone-proj) + NoPrompt',
                           'size': 224,
                           'weight': 'EVA02',
                           'sne': True,
                           'sne_fusion_stage': 'backbone',
                           'sne_mode': 'proj',
                           'prompt': False,
                           'context_decoder': True,
                           'reg_e0_eval': True,
                           'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneBackboneProj_noprompt_regE0eval_textEncodereval/exp_224_eva02_sneBackboneProj_noprompt_regE0eval/test_results/exp_224_eva02_sneBackboneProj_noprompt_regE0eval/testing_eval_file_stats_20251215_085058.csv',
                           'csv_road3d': ''},
    'E224-BbPr_v1': {   'name': 'ablation_224_densevlm_sne_noprompt_road',
                        'desc': '224 + EVA02 + SNE(backbone-proj) + NoPrompt',
                        'size': 224,
                        'weight': 'EVA02',
                        'sne': True,
                        'sne_fusion_stage': 'backbone',
                        'sne_mode': 'proj',
                        'prompt': False,
                        'context_decoder': True,
                        'csv_orfd': '',
                        'csv_road3d': ''},
    'E224-CA': {   'name': 'ablation_224_eva02_sneBackboneCrossAttn_noprompt',
                   'desc': '224 + EVA02 + SNE(backbone-cross_attn) + NoPrompt',
                   'size': 224,
                   'weight': 'EVA02',
                   'sne': True,
                   'sne_fusion_stage': 'backbone',
                   'sne_mode': 'cross_attn',
                   'prompt': False,
                   'context_decoder': True,
                   'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneBackboneCrossAttn_noprompt/exp_224_eva02_sneBackboneCrossAttn_noprompt/test_results/exp_224_eva02_sneBackboneCrossAttn_noprompt/testing_eval_file_stats_20251215_085704.csv',
                   'csv_road3d': ''},
    'E224-OT': {   'name': 'ablation_224_eva02_sneotFalse_noprompt',
                   'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=F] + NoPrompt',
                   'size': 224,
                   'weight': 'EVA02',
                   'sne': True,
                   'sne_fusion_stage': 'backbone',
                   'sne_mode': 'ot',
                   'ot_prior': False,
                   'prompt': False,
                   'context_decoder': True,
                   'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneot_noprompt/20251204_1638/exp_224_eva02_sneot_noprompt/test_results/exp_224_eva02_sneotFalse_noprompt/testing_eval_file_stats_20251204_193006.csv',
                   'csv_road3d': '',
                   'ot_prior_mode': 'argmax',
                   'ot_cost_type': 'cos',
                   'ot_fuse_mode': 'proj',
                   'ot_fuse_output': True,
                   'ot_softunion': False},
    'E224-OTP': {   'name': 'ablation_224_eva02_sneotTrue_noprompt',
                    'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T] + NoPrompt',
                    'size': 224,
                    'weight': 'EVA02',
                    'sne': True,
                    'sne_fusion_stage': 'backbone',
                    'sne_mode': 'ot',
                    'ot_prior': True,
                    'ot_cost_type': 'cos',
                    'ot_fuse_output': True,
                    'prompt': False,
                    'context_decoder': True,
                    'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_noprompt/20251204_1727/exp_224_eva02_sneotTrue_noprompt/test_results/exp_224_eva02_sneotTrue_noprompt/testing_eval_file_stats_20251204_193627.csv',
                    'csv_road3d': '',
                    'ot_prior_mode': 'argmax',
                    'ot_fuse_mode': 'proj',
                    'ot_softunion': False},
    'E224-OTP-FPN': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_noprompt',
                        'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T] + NoPrompt + PFPN',
                        'size': 224,
                        'weight': 'EVA02',
                        'sne': True,
                        'sne_fusion_stage': 'backbone',
                        'sne_mode': 'ot',
                        'ot_prior': True,
                        'prompt': False,
                        'context_decoder': True,
                        'patch_fpn': True,
                        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_noprompt/20251207_1303/exp_224_eva02_sneotTrue_patchfpn_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_noprompt/testing_eval_file_stats_20251207_140213.csv',
                        'csv_road3d': '',
                        'ot_prior_mode': 'argmax',
                        'ot_cost_type': 'cos',
                        'ot_fuse_mode': 'proj',
                        'ot_fuse_output': True,
                        'ot_softunion': False},
    'E224-OTP-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt',
                           'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T] + NoPrompt + PFPN + PiSup',
                           'size': 224,
                           'weight': 'EVA02',
                           'sne': True,
                           'sne_fusion_stage': 'backbone',
                           'sne_mode': 'ot',
                           'ot_prior': True,
                           'ot_cost_type': 'cos',
                           'ot_fuse_output': True,
                           'prompt': False,
                           'context_decoder': True,
                           'patch_fpn': True,
                           'pi_sup': True,
                           'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt/20251212_2239/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/testing_eval_file_stats_20251213_095931.csv',
                           'csv_road3d': '',
                           'ot_prior_mode': 'argmax',
                           'ot_fuse_mode': 'proj',
                           'ot_softunion': False},
    'E224-OTP-XSam-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt',
                            'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T] + NoPrompt + PFPN(XSam) + PiSup',
                            'size': 224,
                            'weight': 'EVA02',
                            'sne': True,
                            'sne_fusion_stage': 'backbone',
                            'sne_mode': 'ot',
                            'ot_prior': True,
                            'ot_cost_type': 'cos',
                            'ot_fuse_output': True,
                            'prompt': False,
                            'context_decoder': True,
                            'patch_fpn_xsam': True,
                            'pi_sup': True,
                            'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/20251212_1850/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/testing_eval_file_stats_20251212_201334.csv',
                            'csv_road3d': '',
                            'ot_prior_mode': 'argmax',
                            'ot_fuse_mode': 'proj',
                            'ot_softunion': False},
    'E224-OTPL2-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_l2',
                             'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,l2] + NoPrompt + PFPN + PiSup',
                             'size': 224,
                             'weight': 'EVA02',
                             'sne': True,
                             'sne_fusion_stage': 'backbone',
                             'sne_mode': 'ot',
                             'ot_prior': True,
                             'ot_cost_type': 'l2',
                             'ot_fuse_output': True,
                             'prompt': False,
                             'context_decoder': True,
                             'patch_fpn': True,
                             'pi_sup': True,
                             'csv_orfd': '',
                             'csv_road3d': '',
                             'ot_prior_mode': 'argmax',
                             'ot_fuse_mode': 'proj',
                             'ot_softunion': False},
    'E224-OTPL2Nr-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_l2',
                               'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,l2] + NoPrompt + PFPN + PiSup',
                               'size': 224,
                               'weight': 'EVA02',
                               'sne': True,
                               'sne_fusion_stage': 'backbone',
                               'sne_mode': 'ot',
                               'ot_prior': True,
                               'ot_cost_type': 'l2',
                               'ot_fuse_output': False,
                               'prompt': False,
                               'context_decoder': True,
                               'patch_fpn': True,
                               'pi_sup': True,
                               'csv_orfd': '',
                               'csv_road3d': '',
                               'ot_prior_mode': 'argmax',
                               'ot_fuse_mode': 'proj',
                               'ot_softunion': False},
    'E224-OTPL2mNr-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_l2_mean',
                                'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,l2,mean] + NoPrompt + PFPN + PiSup',
                                'size': 224,
                                'weight': 'EVA02',
                                'sne': True,
                                'sne_fusion_stage': 'backbone',
                                'sne_mode': 'ot',
                                'ot_prior': True,
                                'ot_cost_type': 'l2',
                                'ot_fuse_mode': 'mean',
                                'ot_fuse_output': False,
                                'prompt': False,
                                'context_decoder': True,
                                'patch_fpn': True,
                                'pi_sup': True,
                                'csv_orfd': '',
                                'csv_road3d': '',
                                'ot_prior_mode': 'argmax',
                                'ot_softunion': False},
    'E224-OTPNr-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos',
                             'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T] + NoPrompt + PFPN + PiSup',
                             'size': 224,
                             'weight': 'EVA02',
                             'sne': True,
                             'sne_fusion_stage': 'backbone',
                             'sne_mode': 'ot',
                             'ot_prior': True,
                             'ot_cost_type': 'cos',
                             'ot_fuse_output': False,
                             'prompt': False,
                             'context_decoder': True,
                             'patch_fpn': True,
                             'pi_sup': True,
                             'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos/20251213_1826/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos/testing_eval_file_stats_20251213_234932.csv',
                             'csv_road3d': '',
                             'ot_prior_mode': 'argmax',
                             'ot_fuse_mode': 'proj',
                             'ot_softunion': False},
    'E224-OTP_v1': {   'name': 'exp_224_eva02_sneotTrue_noprompt_road',
                       'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T] + NoPrompt',
                       'size': 224,
                       'weight': 'EVA02',
                       'sne': True,
                       'sne_fusion_stage': 'backbone',
                       'sne_mode': 'ot',
                       'ot_prior': True,
                       'prompt': False,
                       'context_decoder': True,
                       'csv_orfd': '',
                       'csv_road3d': '',
                       'ot_prior_mode': 'argmax',
                       'ot_cost_type': 'cos',
                       'ot_fuse_mode': 'proj',
                       'ot_fuse_output': True,
                       'ot_softunion': False},
    'E224-OTPmNr-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean',
                              'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,mean] + NoPrompt + PFPN + PiSup',
                              'size': 224,
                              'weight': 'EVA02',
                              'sne': True,
                              'sne_fusion_stage': 'backbone',
                              'sne_mode': 'ot',
                              'ot_prior': True,
                              'ot_cost_type': 'cos',
                              'ot_fuse_mode': 'mean',
                              'ot_fuse_output': False,
                              'prompt': False,
                              'context_decoder': True,
                              'patch_fpn': True,
                              'pi_sup': True,
                              'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean/20251214_0009/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean/testing_eval_file_stats_20251214_090459.csv',
                              'csv_road3d': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road/20251215_1727/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road/testing_eval_file_stats_20251216_095930.csv',
                              'ot_prior_mode': 'argmax',
                              'ot_softunion': False},
    'E224-OTPmNr-FPN-Pi_v1': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road',
                                 'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,mean] + NoPrompt + PFPN + PiSup',
                                 'size': 224,
                                 'weight': 'EVA02',
                                 'sne': True,
                                 'sne_fusion_stage': 'backbone',
                                 'sne_mode': 'ot',
                                 'ot_prior': True,
                                 'ot_cost_type': 'cos',
                                 'ot_fuse_mode': 'mean',
                                 'ot_fuse_output': False,
                                 'prompt': False,
                                 'context_decoder': True,
                                 'patch_fpn': True,
                                 'pi_sup': True,
                                 'csv_orfd': '',
                                 'csv_road3d': '',
                                 'ot_prior_mode': 'argmax',
                                 'ot_softunion': False},
    'E224-OTPmNr-P-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean',
                                'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,mean] + Prompt + PFPN + PiSup',
                                'size': 224,
                                'weight': 'EVA02',
                                'sne': True,
                                'sne_fusion_stage': 'backbone',
                                'sne_mode': 'ot',
                                'ot_prior': True,
                                'ot_cost_type': 'cos',
                                'ot_fuse_mode': 'mean',
                                'ot_fuse_output': False,
                                'prompt': True,
                                'prompt_cls_mode': 'hard',
                                'context_decoder': True,
                                'patch_fpn': True,
                                'pi_sup': True,
                                'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/testing_eval_file_stats_20251215_145845.csv',
                                'csv_road3d': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/20251215_1728/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/testing_eval_file_stats_20251216_085216.csv',
                                'ot_prior_mode': 'argmax',
                                'ot_softunion': False},
    'E224-OTPmNr-P-FPN-Pi_v1': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road',
                                   'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,mean] + Prompt + PFPN + PiSup',
                                   'size': 224,
                                   'weight': 'EVA02',
                                   'sne': True,
                                   'sne_fusion_stage': 'backbone',
                                   'sne_mode': 'ot',
                                   'ot_prior': True,
                                   'ot_cost_type': 'cos',
                                   'ot_fuse_mode': 'mean',
                                   'ot_fuse_output': False,
                                   'prompt': True,
                                   'prompt_cls_mode': 'hard',
                                   'context_decoder': True,
                                   'patch_fpn': True,
                                   'pi_sup': True,
                                   'csv_orfd': '',
                                   'csv_road3d': '',
                                   'ot_prior_mode': 'argmax',
                                   'ot_softunion': False},
    'E224-OTPpUmNr-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion',
                                'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,Prob,SoftU,mean] + NoPrompt + PFPN + '
                                        'PiSup',
                                'size': 224,
                                'weight': 'EVA02',
                                'sne': True,
                                'sne_fusion_stage': 'backbone',
                                'sne_mode': 'ot',
                                'ot_prior': True,
                                'ot_prior_mode': 'prob',
                                'ot_cost_type': 'cos',
                                'ot_fuse_mode': 'mean',
                                'ot_fuse_output': False,
                                'ot_softunion': True,
                                'prompt': False,
                                'context_decoder': True,
                                'patch_fpn': True,
                                'pi_sup': True,
                                'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion/20251214_1032/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion/testing_eval_file_stats_20251214_165207.csv',
                                'csv_road3d': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road/20251215_1726/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road/testing_eval_file_stats_20251216_103318.csv'},
    'E224-OTPpUmNr-FPN-Pi_v1': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road',
                                   'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,Prob,SoftU,mean] + NoPrompt + PFPN '
                                           '+ PiSup',
                                   'size': 224,
                                   'weight': 'EVA02',
                                   'sne': True,
                                   'sne_fusion_stage': 'backbone',
                                   'sne_mode': 'ot',
                                   'ot_prior': True,
                                   'ot_prior_mode': 'prob',
                                   'ot_cost_type': 'cos',
                                   'ot_fuse_mode': 'mean',
                                   'ot_fuse_output': False,
                                   'ot_softunion': True,
                                   'prompt': False,
                                   'context_decoder': True,
                                   'patch_fpn': True,
                                   'pi_sup': True,
                                   'csv_orfd': '',
                                   'csv_road3d': ''},
    'E224-OTPpUmNr-P-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion',
                                  'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,Prob,SoftU,mean] + Prompt + PFPN + '
                                          'PiSup',
                                  'size': 224,
                                  'weight': 'EVA02',
                                  'sne': True,
                                  'sne_fusion_stage': 'backbone',
                                  'sne_mode': 'ot',
                                  'ot_prior': True,
                                  'ot_prior_mode': 'prob',
                                  'ot_cost_type': 'cos',
                                  'ot_fuse_mode': 'mean',
                                  'ot_fuse_output': False,
                                  'ot_softunion': True,
                                  'prompt': True,
                                  'prompt_cls_mode': 'hard',
                                  'context_decoder': True,
                                  'patch_fpn': True,
                                  'pi_sup': True,
                                  'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion/testing_eval_file_stats_20251215_150552.csv',
                                  'csv_road3d': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/20251215_1727/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/testing_eval_file_stats_20251216_092802.csv'},
    'E224-OTPpUmNr-P-FPN-Pi_v1': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road',
                                     'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,Prob,SoftU,mean] + Prompt + PFPN '
                                             '+ PiSup',
                                     'size': 224,
                                     'weight': 'EVA02',
                                     'sne': True,
                                     'sne_fusion_stage': 'backbone',
                                     'sne_mode': 'ot',
                                     'ot_prior': True,
                                     'ot_prior_mode': 'prob',
                                     'ot_cost_type': 'cos',
                                     'ot_fuse_mode': 'mean',
                                     'ot_fuse_output': False,
                                     'ot_softunion': True,
                                     'prompt': True,
                                     'prompt_cls_mode': 'hard',
                                     'context_decoder': True,
                                     'patch_fpn': True,
                                     'pi_sup': True,
                                     'csv_orfd': '',
                                     'csv_road3d': ''},
    'E224-OTPpmNr-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob',
                               'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,Prob,mean] + NoPrompt + PFPN + PiSup',
                               'size': 224,
                               'weight': 'EVA02',
                               'sne': True,
                               'sne_fusion_stage': 'backbone',
                               'sne_mode': 'ot',
                               'ot_prior': True,
                               'ot_prior_mode': 'prob',
                               'ot_cost_type': 'cos',
                               'ot_fuse_mode': 'mean',
                               'ot_fuse_output': False,
                               'prompt': False,
                               'context_decoder': True,
                               'patch_fpn': True,
                               'pi_sup': True,
                               'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob/20251214_1005/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob/testing_eval_file_stats_20251214_164507.csv',
                               'csv_road3d': '',
                               'ot_softunion': False},
    'E224-OTPxNr-P-FPN-Pi': {   'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_max',
                                'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=T,max] + Prompt + PFPN + PiSup',
                                'size': 224,
                                'weight': 'EVA02',
                                'sne': True,
                                'sne_fusion_stage': 'backbone',
                                'sne_mode': 'ot',
                                'ot_prior': True,
                                'ot_cost_type': 'cos',
                                'ot_fuse_mode': 'max',
                                'ot_fuse_output': False,
                                'prompt': True,
                                'prompt_cls_mode': 'hard',
                                'context_decoder': True,
                                'patch_fpn': True,
                                'pi_sup': True,
                                'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_max/20251214_2027/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_max/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_max/testing_eval_file_stats_20251215_145144.csv',
                                'csv_road3d': '',
                                'ot_prior_mode': 'argmax',
                                'ot_softunion': False},
    'E224-OT_v1': {   'name': 'exp_224_eva02_sneotFalse_noprompt_road',
                      'desc': '224 + EVA02 + SNE(backbone-ot)[Prior=F] + NoPrompt',
                      'size': 224,
                      'weight': 'EVA02',
                      'sne': True,
                      'sne_fusion_stage': 'backbone',
                      'sne_mode': 'ot',
                      'ot_prior': False,
                      'prompt': False,
                      'context_decoder': True,
                      'csv_orfd': '',
                      'csv_road3d': '',
                      'ot_prior_mode': 'argmax',
                      'ot_cost_type': 'cos',
                      'ot_fuse_mode': 'proj',
                      'ot_fuse_output': True,
                      'ot_softunion': False},
    'E224-PxPr': {   'name': 'ablation_224_eva02_snePixelProj_noprompt',
                     'desc': '224 + EVA02 + SNE(pixel-proj) + NoPrompt',
                     'size': 224,
                     'weight': 'EVA02',
                     'sne': True,
                     'sne_fusion_stage': 'pixel',
                     'sne_mode': 'proj',
                     'prompt': False,
                     'context_decoder': True,
                     'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_snePixelProj_noprompt/exp_224_eva02_snePixelProj_noprompt/test_results/exp_224_eva02_snePixelProj_noprompt/testing_eval_file_stats_20251215_090312.csv',
                     'csv_road3d': ''},
    'E224-PxPr-P': {   'name': 'ablation_224_densevlm_sne_prompt_road',
                       'desc': '224 + EVA02 + SNE(pixel-proj) + Prompt',
                       'size': 224,
                       'weight': 'EVA02',
                       'sne': True,
                       'sne_fusion_stage': 'pixel',
                       'sne_mode': 'proj',
                       'prompt': True,
                       'prompt_cls_mode': 'hard',
                       'context_decoder': True,
                       'csv_orfd': '',
                       'csv_road3d': ''},
    'E512-Base': {   'name': 'ablation_512_eva02_nosne_noprompt',
                     'desc': '512 + EVA02 + NoSNE + NoPrompt',
                     'size': 512,
                     'weight': 'EVA02',
                     'sne': False,
                     'sne_fusion_stage': 'pixel',
                     'sne_mode': 'proj',
                     'prompt': False,
                     'context_decoder': True,
                     'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_191028.csv',
                     'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_nosne_noprompt_road/20251203_2313/exp_512_eva02_nosne_noprompt_road/test_results/exp_512_eva02_nosne_noprompt_road/testing_eval_file_stats_20251204_104203.csv'},
    'E512-Base-P': {   'name': 'ablation_512_densevlm_nosne_prompt',
                       'desc': '512 + EVA02 + NoSNE + Prompt',
                       'size': 512,
                       'weight': 'EVA02',
                       'sne': False,
                       'sne_fusion_stage': 'pixel',
                       'sne_mode': 'proj',
                       'prompt': True,
                       'prompt_cls_mode': 'hard',
                       'context_decoder': True,
                       'csv_orfd': '/root/tqdm/work_dirs/ablation_512_densevlm_nosne_prompt/20251204_0815/exp_512_densevlm_nosne_prompt/test_results/exp_512_densevlm_nosne_prompt/testing_eval_file_stats_20251204_115858.csv',
                       'csv_road3d': ''},
    'E512-Base-P_v1': {   'name': 'ablation_512_densevlm_nosne_prompt_road',
                          'desc': '512 + EVA02 + NoSNE + Prompt',
                          'size': 512,
                          'weight': 'EVA02',
                          'sne': False,
                          'sne_fusion_stage': 'pixel',
                          'sne_mode': 'proj',
                          'prompt': True,
                          'prompt_cls_mode': 'hard',
                          'context_decoder': True,
                          'csv_orfd': '',
                          'csv_road3d': ''},
    'E512-Base_v1': {   'name': 'ablation_512_eva02_nosne_noprompt_road',
                        'desc': '512 + EVA02 + NoSNE + NoPrompt',
                        'size': 512,
                        'weight': 'EVA02',
                        'sne': False,
                        'sne_fusion_stage': 'pixel',
                        'sne_mode': 'proj',
                        'prompt': False,
                        'context_decoder': True,
                        'csv_orfd': '',
                        'csv_road3d': ''},
    'E512-OTPmNr-P-FPN-Pi': {   'name': 'ablation_512_densevlm_sneotTrue_patchfpn_pisup_prompt_no_cos_mean',
                                'desc': '512 + EVA02 + SNE(backbone-ot)[Prior=T,mean] + Prompt + PFPN + PiSup',
                                'size': 512,
                                'weight': 'EVA02',
                                'sne': True,
                                'sne_fusion_stage': 'backbone',
                                'sne_mode': 'ot',
                                'ot_prior': True,
                                'ot_cost_type': 'cos',
                                'ot_fuse_mode': 'mean',
                                'ot_fuse_output': False,
                                'prompt': True,
                                'prompt_cls_mode': 'hard',
                                'context_decoder': True,
                                'patch_fpn': True,
                                'pi_sup': True,
                                'csv_orfd': '/root/tqdm/work_dirs/ablation_512_densevlm_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/20251214_2026/exp_512_densevlm_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/test_results/exp_512_densevlm_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/testing_eval_file_stats_20251215_083244.csv',
                                'csv_road3d': '',
                                'ot_prior_mode': 'argmax',
                                'ot_softunion': False},
    'E512-OTPmNr-P-FPN-Pi_v1': {   'name': 'ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean',
                                   'desc': '512 + EVA02 + SNE(backbone-ot)[Prior=T,mean] + Prompt + PFPN + PiSup',
                                   'size': 512,
                                   'weight': 'EVA02',
                                   'sne': True,
                                   'sne_fusion_stage': 'backbone',
                                   'sne_mode': 'ot',
                                   'ot_prior': True,
                                   'ot_cost_type': 'cos',
                                   'ot_fuse_mode': 'mean',
                                   'ot_fuse_output': False,
                                   'prompt': True,
                                   'prompt_cls_mode': 'hard',
                                   'context_decoder': True,
                                   'patch_fpn': True,
                                   'pi_sup': True,
                                   'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/20251214_2025/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/test_results/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean/testing_eval_file_stats_20251215_151248.csv',
                                   'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/20251216_1157/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/test_results/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/testing_eval_file_stats_20251217_094817.csv',
                                   'ot_prior_mode': 'argmax',
                                   'ot_softunion': False},
    'E512-OTPmNr-P-FPN-Pi_v2': {   'name': 'ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road',
                                   'desc': '512 + EVA02 + SNE(backbone-ot)[Prior=T,mean] + Prompt + PFPN + PiSup',
                                   'size': 512,
                                   'weight': 'EVA02',
                                   'sne': True,
                                   'sne_fusion_stage': 'backbone',
                                   'sne_mode': 'ot',
                                   'ot_prior': True,
                                   'ot_cost_type': 'cos',
                                   'ot_fuse_mode': 'mean',
                                   'ot_fuse_output': False,
                                   'prompt': True,
                                   'prompt_cls_mode': 'hard',
                                   'context_decoder': True,
                                   'patch_fpn': True,
                                   'pi_sup': True,
                                   'csv_orfd': '',
                                   'csv_road3d': '',
                                   'ot_prior_mode': 'argmax',
                                   'ot_softunion': False},
    'E512-OTPmNr-Ps-FPN-Pi_v1': {   'name': 'ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_road',
                                    'desc': '512 + EVA02 + SNE(backbone-ot)[Prior=T,mean] + Prompt(Soft) + PFPN + PiSup',
                                    'size': 512,
                                    'weight': 'EVA02',
                                    'sne': True,
                                    'sne_fusion_stage': 'backbone',
                                    'sne_mode': 'ot',
                                    'ot_prior': True,
                                    'ot_cost_type': 'cos',
                                    'ot_fuse_mode': 'mean',
                                    'ot_fuse_output': False,
                                    'prompt': True,
                                    'prompt_cls_mode': 'soft',
                                    'context_decoder': True,
                                    'patch_fpn': True,
                                    'pi_sup': True,
                                    'csv_orfd': '',
                                    'csv_road3d': '',
                                    'ot_prior_mode': 'argmax',
                                    'ot_softunion': False},
    'E512-OTPpUmNr-P-FPN-Pi': {   'name': 'ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road',
                                  'desc': '512 + EVA02 + SNE(backbone-ot)[Prior=T,Prob,SoftU,mean] + Prompt + PFPN + '
                                          'PiSup',
                                  'size': 512,
                                  'weight': 'EVA02',
                                  'sne': True,
                                  'sne_fusion_stage': 'backbone',
                                  'sne_mode': 'ot',
                                  'ot_prior': True,
                                  'ot_prior_mode': 'prob',
                                  'ot_cost_type': 'cos',
                                  'ot_fuse_mode': 'mean',
                                  'ot_fuse_output': False,
                                  'ot_softunion': True,
                                  'prompt': True,
                                  'prompt_cls_mode': 'hard',
                                  'context_decoder': True,
                                  'patch_fpn': True,
                                  'pi_sup': True,
                                  'csv_orfd': '',
                                  'csv_road3d': ''},
    'E512-PxPr-P': {   'name': 'ablation_512_densevlm_sne_prompt',
                       'desc': '512 + EVA02 + SNE(pixel-proj) + Prompt',
    'E512-OTPpUmNr-Ps-FPN-Pi_T05': {   'name': 'ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_0.5',
                                       'desc': '512 + EVA02 + SNE(backbone-ot)[Prior=T,Prob,SoftU,mean,T=0.5] + Prompt(Soft) + PFPN + PiSup',
                                       'size': 512,
                                       'weight': 'EVA02',
                                       'sne': True,
                                       'sne_fusion_stage': 'backbone',
                                       'sne_mode': 'ot',
                                       'ot_prior': True,
                                       'ot_cost_type': 'cos',
                                       'ot_fuse_mode': 'mean',
                                       'ot_fuse_output': False,
                                       'prompt': True,
                                       'prompt_cls_mode': 'soft',
                                       'context_decoder': True,
                                       'patch_fpn': True,
                                       'pi_sup': True,
                                       'csv_orfd': '',
                                       'csv_road3d': '',
                                       'ot_prior_mode': 'prob',
                                       'ot_softunion': True},
                       'size': 512,
                       'weight': 'EVA02',
                       'sne': True,
                       'sne_fusion_stage': 'pixel',
                       'sne_mode': 'proj',
                       'prompt': True,
                       'prompt_cls_mode': 'hard',
                       'context_decoder': True,
                       'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_205051.csv',
                       'csv_road3d': '/root/tqdm/work_dirs/ablation_512_densevlm_sne_prompt_road/20251203_2312/exp_512_densevlm_sne_prompt_road/test_results/exp_512_densevlm_sne_prompt_road/testing_eval_file_stats_20251204_142944.csv'},
    'E512-PxPr-P_v1': {   'name': 'ablation_512_densevlm_sne_prompt_road',
                          'desc': '512 + EVA02 + SNE(pixel-proj) + Prompt',
                          'size': 512,
                          'weight': 'EVA02',
                          'sne': True,
                          'sne_fusion_stage': 'pixel',
                          'sne_mode': 'proj',
                          'prompt': True,
                          'prompt_cls_mode': 'hard',
                          'context_decoder': True,
                          'csv_orfd': '',
                          'csv_road3d': ''}}

def main():
    'E512-OTPpUmNr-P-FPN-Pi_T05': {   'name': 'ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road_0.5',
                                      'desc': '512 + EVA02 + SNE(backbone-ot)[Prior=T,Prob,SoftU,mean,T=0.5] + Prompt + PFPN + PiSup',
                                      'size': 512,
                                      'weight': 'EVA02',
                                      'sne': True,
                                      'sne_fusion_stage': 'backbone',
                                      'sne_mode': 'ot',
                                      'ot_prior': True,
                                      'ot_cost_type': 'cos',
                                      'ot_fuse_mode': 'mean',
                                      'ot_fuse_output': False,
                                      'prompt': True,
                                      'context_decoder': True,
                                      'patch_fpn': True,
                                      'pi_sup': True,
                                      'csv_orfd': '',
                                      'csv_road3d': '',
                                      'ot_prior_mode': 'prob',
                                      'ot_softunion': True},
    parser = argparse.ArgumentParser(description='消融实验结果统计 (mIoU 聚焦版)')
    parser.add_argument('--csv-list', type=str, default=None,
                        help='包含 CSV 路径列表的文件 (本脚本暂未使用)')
    parser.add_argument('--work-dirs', type=str, default='./work_dirs',
                        help='工作目录')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['road3d', 'orfd', 'both'],
                        help='指定数据集用于 known/unknown 场景分割，both 表示同时处理两个数据集')
    parser.add_argument('--sort-by', type=str, default='all',
                        choices=['all', 'unk', 'known'],
                        help='排序依据: all (Overall mIoU), unk (Unknown mIoU), known (Known mIoU)')
    args = parser.parse_args()
    
    # 默认处理两个数据集
    if args.dataset is None:
        args.dataset = 'both'
    
    datasets_to_process = ['orfd', 'road3d'] if args.dataset == 'both' else [args.dataset]
    
    for dataset_name in datasets_to_process:
        process_dataset_miou_focused(dataset_name, args)


def process_dataset_miou_focused(dataset_name, args):
    """处理单个数据集的消融实验结果，生成 mIoU 聚合表"""
    csv_key = f'csv_{dataset_name}'
    dataset_display = dataset_name.upper()
    
    print("\n" + "=" * 140)
    print(f"TQT 消融实验结果统计 (mIoU 聚焦) - {dataset_display}")
    print("=" * 140)
    
    results_aggregated = []
    
    # 收集各实验结果
    for exp_id, exp_info in ABLATION_EXPERIMENTS.items():
        # 优先使用手动指定的 CSV 路径
        csv_path = exp_info.get(csv_key)
        if csv_path:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                # print(f"[{exp_id}] {exp_info['desc']}: 指定的 CSV 不存在: {csv_path}")
                csv_path = None
        else:
            # 自动搜索
            csv_path = find_csv_for_experiment(exp_info['name'], args.work_dirs)
        
        # === SNE 配置解析 ===
        if exp_info['sne']:
            sne_mode = exp_info.get('sne_mode', 'proj')
            sne_stage = exp_info.get('sne_fusion_stage', 'backbone')
            # 缩写: backbone->Bb, pixel->Px; proj->Pr, cross_attn->CA, ot->OT
            stage_abbr = 'Bb' if sne_stage == 'backbone' else 'Px'
            mode_abbr = {'proj': 'Pr', 'cross_attn': 'CA', 'ot': 'OT', 'add': 'Ad', 'concat': 'Ct'}.get(sne_mode, sne_mode)
            sne_config_str = f"{stage_abbr}/{mode_abbr}"
        else:
            sne_config_str = '-'

        # === OT 配置解析 ===
        ot_prior_raw = exp_info.get('ot_prior', None)
        if exp_info.get('sne_mode') == 'ot': # 使用原始 config 判断
            ot_prior_str = 'T' if ot_prior_raw is True else ('F' if ot_prior_raw is False else '-')
            ot_prior_mode = exp_info.get('ot_prior_mode', '-')
            if ot_prior_mode == 'argmax': ot_prior_mode = 'Arg'
            elif ot_prior_mode == 'prob': ot_prior_mode = 'Prb'
            
            ot_fuse = exp_info.get('ot_fuse_mode', 'proj') # default proj
            ot_cost = exp_info.get('ot_cost_type', 'cos')  # default cos
            
            # 组合展示 OT 配置以节省列空间: cos/proj
            ot_config_str = f"{ot_cost}/{ot_fuse}"
            if exp_info.get('ot_softunion'):
                ot_config_str += "+SU"
            if exp_info.get('ot_fuse_output') is False:
                ot_config_str += "(NoRes)" # No Residual
        else:
            ot_prior_str = '-'
            ot_prior_mode = '-'
            ot_config_str = '-'

        # === Prompt 配置解析 ===
        prompt_use = exp_info.get('prompt', False)
        prompt_cls = exp_info.get('prompt_cls', False)
        
        if prompt_use:
            prompt_str = "Stat" # Static
        elif prompt_cls:
            p_mode = exp_info.get('prompt_cls_mode', 'hard')
            prompt_str = f"Dyn({p_mode[0].upper()})" # Dyn(H) or Dyn(S)
        else:
            prompt_str = "-"

        # 基础行信息
        row = {
            'Exp': exp_id,
            'Desc': exp_info['desc'],
            'Size': exp_info['size'],
            'Weight': 'EVA' if 'EVA' in exp_info['weight'] else 'DVL', # 缩写权重
            'SNE_Cfg': sne_config_str,
            'OT_Prior': f"{ot_prior_str}({ot_prior_mode})" if ot_prior_str != '-' else '-',
            'OT_Cfg': ot_config_str,
            'Prompt': prompt_str,
            'CtxDec': '+' if exp_info.get('context_decoder', True) else '-',
            'PFPN': '+' if exp_info.get('patch_fpn') or exp_info.get('patch_fpn_xsam') else '-',
            'PiSup': '+' if exp_info.get('pi_sup') else '-',
        }
        
        # 初始化指标为 N/A
        row['mIoU(All)'] = -1.0 
        row['mIoU(Kn)'] = -1.0
        row['mIoU(Unk)'] = -1.0
        
        # 格式化显示的字符串
        row_display = row.copy()
        row_display['mIoU(All)'] = '-'
        row_display['mIoU(Kn)'] = '-'
        row_display['mIoU(Unk)'] = '-'

        if csv_path:
            try:
                metrics = analyze_csv(csv_path, dataset_name)
                
                # Overall
                m_overall = metrics['overall']
                row['mIoU(All)'] = m_overall['mIoU'] * 100 # 转为百分比
                
                row_display['mIoU(All)'] = f"{m_overall['mIoU']*100:.2f}"
                
                # Known/Unknown
                if 'known' in metrics:
                    m_known = metrics['known']
                    m_unknown = metrics['unknown']
                    
                    row['mIoU(Kn)'] = m_known['mIoU'] * 100
                    row['mIoU(Unk)'] = m_unknown['mIoU'] * 100
                    
                    row_display['mIoU(Kn)'] = f"{m_known['mIoU']*100:.2f}"
                    row_display['mIoU(Unk)'] = f"{m_unknown['mIoU']*100:.2f}"
                
            except Exception as e:
                pass
        
        # 将原始数值保存在 row 中用于排序，将显示字符串保存在 row_display 中
        row['_display'] = row_display
        results_aggregated.append(row)

    # === 排序逻辑 ===
    sort_key = 'mIoU(All)'
    if args.sort_by == 'unk':
        sort_key = 'mIoU(Unk)'
    elif args.sort_by == 'known':
        sort_key = 'mIoU(Kn)'
    
    # 过滤掉没有数据的行进行排序，然后把没数据的放到最后
    valid_results = [r for r in results_aggregated if r[sort_key] != -1.0]
    empty_results = [r for r in results_aggregated if r[sort_key] == -1.0]
    
    valid_results.sort(key=lambda x: x[sort_key], reverse=True)
    sorted_results = valid_results + empty_results
    
    # === 构建最终 DataFrame 用于打印 ===
    # 提取 display 字典并重组
    final_rows = []
    for r in sorted_results:
        # 使用 display 中的格式化字符串，覆盖数值
        d = r['_display']
        # 确保关键列的顺序
        ordered_row = {
            'ID': d['Exp'],
            'Size': d.get('Size', '-'),
            'Backbone': d['Weight'],
            'SNE_Cfg': d['SNE_Cfg'], # stage/mode
            'OT_Prior': d['OT_Prior'],
            'OT_Cfg': d['OT_Cfg'],
            'Prompt': d['Prompt'],   # Static/Dyn(mode)/-
            'CtxDec': d['CtxDec'],
            'PFPN': d['PFPN'],
            'PiSup': d['PiSup'],
            # 核心指标放后面
            'mIoU(All)': d['mIoU(All)'],
            'mIoU(Kn)': d['mIoU(Kn)'],
            'mIoU(Unk)': d['mIoU(Unk)'],
        }
        final_rows.append(ordered_row)

    df_final = pd.DataFrame(final_rows)

    
    # 打印表格
    print(df_final.to_string(index=False))
    
    # 保存结果
    output_path = Path(args.work_dirs) / f'ablation_miou_summary_{dataset_name}.csv'
    df_final.to_csv(output_path, index=False)
    print(f"\n[INFO] 汇总结果已保存至: {output_path}")

    # === 简短的对比分析 (打印数值增益) ===
    print("\n" + "-" * 60)
    print(f"关键对比 (mIoU Gain/Loss)")
    print("-" * 60)
    
    comparisons = [
        ('SNE 效果', 'B2', 'C2'),
        ('OT Prior 效果', 'F2a', 'F2b'),
        ('Prompt 效果', 'C2', 'D2'),
        ('Patch-FPN 效果', 'F2b', 'F2c'),
        ('Pi 监督效果', 'F2c', 'F2d'),
    ]
    
    for desc, id1, id2 in comparisons:
        r1 = next((r for r in results_aggregated if r['Exp'] == id1), None)
        r2 = next((r for r in results_aggregated if r['Exp'] == id2), None)
        
        if r1 and r2 and r1['mIoU(All)'] > 0 and r2['mIoU(All)'] > 0:
            diff_all = r2['mIoU(All)'] - r1['mIoU(All)']
            diff_unk = r2['mIoU(Unk)'] - r1['mIoU(Unk)']
            
            print(f"{desc:20s}: {id1} -> {id2}")
            print(f"  Overall: {diff_all:+.2f}%  ( {r1['mIoU(All)']:.2f} -> {r2['mIoU(All)']:.2f} )")
            print(f"  Unknown: {diff_unk:+.2f}%  ( {r1['mIoU(Unk)']:.2f} -> {r2['mIoU(Unk)']:.2f} )")
            print()

if __name__ == '__main__':
    main()
