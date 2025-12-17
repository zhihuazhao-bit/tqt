import os
import os.path as osp
import numpy as np
import cv2
import torch
import pandas as pd
import warnings
import time
import torch.distributed as dist
from collections import defaultdict
from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics, eval_metrics_per
from prettytable import PrettyTable
from collections import OrderedDict
import json
from .dataset_utils import getScores_self
import swanlab

@DATASETS.register_module()
class ORFDDataset(CustomDataset):
    # CLASSES = ('notraversable', 'traversable')
    CLASSES = ('Vehicle-accessible areas', 'High-risk terrain blocks')
    PALETTE = [[0,0,0],[0,128,0]]

    def __init__(self, force_scene_list=None, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if kwargs.get('class_names') is not None:
            self.CLASSES = kwargs.get('class_names')
        if 'split' in kwargs:
            kwargs.pop('split')
        # self.all_scene_map = {
        #     'val':{
        #         'snow':[
        #             'x2021_0223_1856',
        #         ],
        #         'sun':[
        #             # from test datset
        #             'y0602_1330',
        #         ],
        #         'rain':[
        #             'y0609_1947',
        #             'x2021_0222_1745'
        #             ],
        #         'fog':[

        #         ]
        #     },
        #     'test':{
        #         'snow':[
        #             'x2021_0223_1756',
        #         ],
        #         'sun':[
        #             'y0613_1220',
        #             'y0613_1242',
        #         ],
        #         'rain':[
        #             'y0609_1947',
        #             'y0609_1923',
        #             'y2021_0228_1802'
        #             ],
        #         'fog':[
                    
        #         ]
        #     },
        #     'train':{
        #         'snow':[
        #             # 'x2021_0223_1856',
        #             'x2021_0222_1810',
        #             'x2021_0223_1310',
        #             'x2021_0223_1316',
        #             'y2021_0228_1807'
        #         ],
        #         'sun':[
        #             'c2021_0228_1819',
        #             'x2021_0222_1720',
        #             'y0602_1228',
        #             'y0602_1235',
        #             'y0609_1959_2',
        #             'y0613_1238',
        #             'y0613_1248',
        #             'y0613_1252',
        #             'y0613_1304',
        #             'y0613_1509',
        #         ],
        #         'rain':[
        #             'x0613_1627',
        #             'y0609_1639',
        #             'y0609_1750',
        #             'y0609_1954',
        #             'y0613_1632',
        #             'y0616_1750',
        #             'y0616_1950'
        #             ],
        #         'fog':[
                    
        #         ]
        #     }
        # }
        self.scene_attr, self.map2scene = self.load_dicts()
        self.scene_type = kwargs.get('scene_type', None)
        assert self.scene_type in self.map2scene.keys(), f"scene_type should be in {self.map2scene.keys()}"
        if self.scene_type == 'road':
            self.all_scene_map = {
                'paved':[],
                'unpaved':[]
            }
            for key, value in self.map2scene['road'].items():
                t = key.split('_')[0]
                self.all_scene_map[t].extend(value)
        else:
            self.all_scene_map = self.map2scene[self.scene_type]
        self.scene_scope = kwargs.get('scene_scope', None)
        self.all_scene = []
        for s in self.scene_scope:
            assert s in self.all_scene_map.keys(), f"scene_scope should be in {self.all_scene_map.keys()}"
            self.all_scene.extend(self.all_scene_map[s])
        # 允许通过 force_scene_list 在推理时显式指定场景集合（例如只跑部分场景）
        if force_scene_list is not None:
            if isinstance(force_scene_list, (list, tuple)):
                self.all_scene = list(force_scene_list)
            elif isinstance(force_scene_list, str):
                # 兼容单字符串逗号分隔的配置写法
                self.all_scene = [x.strip() for x in force_scene_list.split(',') if x.strip()]
            else:
                raise ValueError('force_scene_list must be list/tuple or comma-separated string')
        # self.all_scene = [
        #     # '0602-1107',  # 森林未铺装道路，晴天、白天
        #     # '0613-1507-2' # 数据标注有问题
        #     '2021-0222-1757'
        #     ]
        # self.all_scene = ['2021-0403-1858']
        # self.all_scene = [
        #     '2021-0403-1744', 
        #     '0602-1107', '2021-0222-1743', '2021-0403-1858'
        # ]
        # self.all_scene = ['0609-1923', '2021-0223-1756']
        # self.all_scene = ['2021-0223-1756']
        
        print_log(f"Classes: {self.CLASSES}")
        
        super(ORFDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
        self.img_dir, _ = osp.split(self.img_dir)
        self.ann_dir, _ = osp.split(self.ann_dir)
        # if kwargs.get('img_dir', 'training') == 'testing':
        #     know_path = os.path.join(self.data_root, 'testing')
        #     self.img_infos = self.load_annotations(know_path, self.img_suffix, know_path, self.seg_map_suffix, self.split)
        #     # image_list_known = self.load_annotations(os.path.join(self.root, 'testing'), '.png', self.all_scene)
        #     all_scene = []
        #     type_map = {
        #         'road': ['unpaved'],
        #         'weather': ['rainy', 'snowy'],
        #         'light': ['nighttime']
        #     }
        #     for s in type_map[self.scene_type]:
        #         assert s in self.all_scene_map.keys(), f"scene_scope should be in {self.all_scene_map.keys()}"
        #         all_scene.extend(self.all_scene_map[s])
            
        #     unknow_train_path = os.path.join(self.data_root, 'training')
        #     unknow_val_path = os.path.join(self.data_root, 'validation')
        #     image_list_unknown_train = self.load_annotations(unknow_train_path, '.png', unknow_train_path, '_labelTrainIds.png', target_scene=all_scene)
        #     image_list_unknown_val = self.load_annotations(unknow_val_path, '.png', unknow_val_path, '_labelTrainIds.png', target_scene=all_scene)
        #     self.img_infos += image_list_unknown_train + image_list_unknown_val
        print('Found %d images in from %s' % (len(self.img_infos), self.data_root))
            


    # 读取JSON文件
    def load_dicts(self):
        with open("/root/tqdm/dataset/ORFD/english_scene_dict.json", 'r', encoding='utf-8') as f:
            english_dict = json.load(f)
        
        with open("/root/tqdm/dataset/ORFD/map2scene.json", 'r', encoding='utf-8') as f:
            map_dict = json.load(f)
        
        return english_dict, map_dict

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split=None, target_scene=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        if target_scene is None:
            target_scene = self.all_scene
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            all_scene = []
            self.dataset_mode = os.path.split(img_dir)[-1]
            for scene in os.listdir(img_dir):
                sub_img_dir = os.path.join(img_dir, scene, 'image_data')
                if scene[1:].replace("_","-") in target_scene:
                    all_scene.append(scene)
                    for img in mmcv.scandir(sub_img_dir, img_suffix, recursive=True):
                        img_info = dict(filename=os.path.join(self.dataset_mode, scene, 'image_data', img))
                        if ann_dir is not None:
                            seg_map = img.replace(img_suffix, seg_map_suffix)
                            img_info['ann'] = dict(seg_map=os.path.join(self.dataset_mode, scene, 'gt_image', seg_map))
                            img_infos.append(img_info)
                            # break
            img_infos = sorted(img_infos, key=lambda x: x['filename'])
        # img_infos = img_infos[0:50]
        print_log(f'Loaded {self.dataset_mode} dataset {len(img_infos)} images from {all_scene} in {self.scene_scope}', logger=get_root_logger())
        return img_infos
    
    def pretty_print_conf_mat(self, conf_mat: np.ndarray, class_names=None, float_fmt="{:.0f}"):
        """使用 PrettyTable 打印混淆矩阵。

        Args:
            conf_mat (np.ndarray): 形状 (C,C)
            class_names (List[str]|None): 类别名称；为 None 时使用索引 0..C-1
            float_fmt (str): 单元格数字格式
        """
        if isinstance(class_names, tuple):
            class_names = list(class_names)
        if conf_mat.ndim != 2 or conf_mat.shape[0] != conf_mat.shape[1]:
            print("[Warn] 非方阵，无法格式化：", conf_mat.shape)
            print(conf_mat)
            return
        C = conf_mat.shape[0]
        if class_names is None or len(class_names) != C:
            class_names = [f"C{i}" for i in range(C)]
        if PrettyTable is None:
            print("PrettyTable 未安装，使用原始 numpy 输出。可: pip install prettytable")
            print(conf_mat)
            return
        table = PrettyTable()
        table.field_names = ["Actual\\Pred"] + class_names + ["RowSum", "Recall"]
        col_sums = conf_mat.sum(axis=0) + 1e-12
        row_sums = conf_mat.sum(axis=1) + 1e-12
        for i in range(C):
            row_vals = []
            for j in range(C):
                row_vals.append(float_fmt.format(conf_mat[i, j]))
            row_sum = row_sums[i]
            recall = conf_mat[i, i] / row_sum if row_sum > 0 else 0.0
            table.add_row([class_names[i]] + row_vals + [float_fmt.format(row_sum), f"{recall*100:.2f}%"])
        # 添加列统计 (Precision)
        prec_row = ["Precision"]
        for j in range(C):
            prec = conf_mat[j, j] / col_sums[j] if col_sums[j] > 0 else 0.0
            prec_row.append(f"{prec*100:.2f}%")
        prec_row += [float_fmt.format(conf_mat.sum()), "-"]
        table.add_row(prec_row)
        # print_log("\nConfusion Matrix (rows=Actual, cols=Pred):")
        print_log(table)
        mprec, mrecall, mf1, miou, fwIoU, prec_road, rec_road, f1_road, iou_road = getScores_self(conf_mat)
        print_log(f"Overall: mPrec {mprec*100:.2f}%, mRecall {mrecall*100:.2f}%, mF1 {mf1*100:.2f}%, mIoU {miou*100:.2f}%, fwIoU {fwIoU*100:.2f}%")
        if len(class_names) == 2:
            print_log(f"Class {class_names[1]}: Prec {prec_road*100:.2f}%, Recall {rec_road*100:.2f}%, F1 {f1_road*100:.2f}%, IoU {iou_road*100:.2f}%")
        try:
            metrics_result = {
                'mPrec': mprec, 'mRecall': mrecall, 'mF1': mf1, 'mIoU': miou, 'fwIoU': fwIoU,
                'prec_road': prec_road, 'rec_road': rec_road, 'f1_road': f1_road, 'iou_road': iou_road,
            }
            swanlab.log(metrics_result)
        except Exception as e:
            print_log(f"[Warn] Swanlab log error: {e}")
            pass

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 save_dir=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'mrIoU']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                # 从数据集中获取标签信息
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics, file_stats = eval_metrics_per(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        print_log(f'file_stats: {len(file_stats)} items')
        file_stats_pd = pd.DataFrame.from_dict(file_stats).T
        
        # 修正后的正确代码:
        def get_attribute(scene_name, attr_key):
            """
            根据场景名称，从 self.scene_attr 中获取指定的属性值。
            
            Args:
                scene_name (str): 场景名称 (e.g., 'y0602_1330').
                attr_key (str): 要获取的属性键 (e.g., 'weather', 'road', 'light').

            Returns:
                str: 属性值，如果找不到则返回 'unknown'.
            """
            # 1. 从场景名中移除首字母前缀 (e.g., 'y0602_1330' -> '0602_1330')
            scene_key = scene_name[1:].replace('_', '-')
            # 2. 安全地获取该场景的属性字典
            scene_attributes = self.scene_attr.get(scene_key, {})
            # 3. 从属性字典中获取指定键的值
            if attr_key == 'road':
                # print(f'scene key: {scene_key}, scene_attributes: {scene_attributes}, road type: {scene_attributes.get(attr_key, "unknown")}')
                road_type = scene_attributes.get(attr_key, 'unknown')
                # return road_type.split('_')[0]  # 只返回 'paved' 或 'unpaved'
                return road_type
            else: 
                return scene_attributes.get(attr_key, 'unknown')

        # 为 weather, road, light 分别创建新列
        for key in ['weather', 'road', 'light']:
            file_stats_pd[key] = file_stats_pd['scene'].apply(lambda x: get_attribute(x, key))

        # 确定 CSV 保存目录：优先使用 save_dir，否则使用 ./csv_result
        if save_dir is not None:
            csv_save_dir = save_dir
            os.makedirs(csv_save_dir, exist_ok=True)
        else:
            csv_save_dir = './csv_result'
            os.makedirs(csv_save_dir, exist_ok=True)
        
        csv_path = osp.join(csv_save_dir, f'{self.dataset_mode}_eval_file_stats_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        file_stats_pd.to_csv(csv_path)
        print_log(f'file_stats saved to {csv_path}')
        
        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        class_num = len(class_names)
        for scene, group in file_stats_pd.groupby('scene'):
            print(self.scene_attr.get(scene[1:].replace('_', '-'), {}))
            conf_mat = np.zeros((class_num, class_num), dtype=np.int64)
            for i in range(class_num):
                class_intersect = group[f'class{i}_intersect'].sum()
                class_area_label = group[f'class{i}_label'].sum()
                class_pred_label = group[f'class{i}_pred_label'].sum()
                conf_mat[i, i] = class_intersect  # TP
                conf_mat[i, 1-i] = class_area_label - class_intersect  # FP
            print_log(f'Confusion Matrix for scene {scene}:')
            self.pretty_print_conf_mat(conf_mat, class_names=class_names)

        for key in [
            ['weather', 'light'],
            ['light', 'weather'],
            ]:
            for value, group in file_stats_pd.groupby(key[0]):
                conf_mat = np.zeros((class_num, class_num), dtype=np.int64)
                for i in range(class_num):
                    class_intersect = group[f'class{i}_intersect'].sum()
                    class_area_label = group[f'class{i}_label'].sum()
                    class_pred_label = group[f'class{i}_pred_label'].sum()
                    conf_mat[i, i] = class_intersect  # TP
                    conf_mat[i, 1-i] = class_area_label - class_intersect  # FP
                print_log(f'Confusion Matrix for {key[0]} {value}:')
                self.pretty_print_conf_mat(conf_mat, class_names=class_names)
                for subvalue, subgroup in group.groupby(key[1]):
                    sub_conf_mat = np.zeros((class_num, class_num), dtype=np.int64)
                    for i in range(class_num):
                        class_intersect = subgroup[f'class{i}_intersect'].sum()
                        class_area_label = subgroup[f'class{i}_label'].sum()
                        class_pred_label = subgroup[f'class{i}_pred_label'].sum()
                        sub_conf_mat[i, i] = class_intersect  # TP
                        sub_conf_mat[i, 1-i] = class_area_label - class_intersect  # FP
                    # 打印语句应在 for i 循环外部
                    print_log(f'Confusion Matrix for {key[0]} {value} {key[1]} {subvalue}:')
                    self.pretty_print_conf_mat(sub_conf_mat, class_names=class_names)

        conf_mat = np.zeros((class_num, class_num), dtype=np.int64)
        for i in range(class_num):
            class_intersect = file_stats_pd[f'class{i}_intersect'].sum()
            class_area_label = file_stats_pd[f'class{i}_label'].sum()
            class_pred_label = file_stats_pd[f'class{i}_pred_label'].sum()
            conf_mat[i, i] = class_intersect  # TP
            conf_mat[i, 1-i] = class_area_label - class_intersect  # FP
        print_log(f'Confusion Matrix for all:')
        self.pretty_print_conf_mat(conf_mat, class_names=class_names)

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'metrics.txt'), 'w') as w:
                w.write('per class results:')
                w.write('\n' + class_table_data.get_string())
                w.write('\n' + 'Summary:')
                w.write('\n' + summary_table_data.get_string())
            w.close()


        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        sne_info = img_info['filename'].replace("image_data", "surface_normal_d2net_v3")
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info, sne=sne_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        sne_info = img_info['filename'].replace("image_data", "surface_normal_d2net_v3")
        results = dict(img_info=img_info, sne=sne_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
