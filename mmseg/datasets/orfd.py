import os
import os.path as osp
import numpy as np
import pandas as pd
import warnings
from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics, eval_metrics_per
from prettytable import PrettyTable
from collections import OrderedDict

@DATASETS.register_module()
class ORFDDataset(CustomDataset):
    CLASSES = ('notraversable', 'traversable')
    PALETTE = [[0,0,0],[0,128,0]]

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        self.all_scene_map = {
            'val':{
                'snow':[
                    'x2021_0223_1856',
                ],
                'sun':[

                ],
                'rain':[
                    'y0609_1947',
                    'x2021_0222_1745'
                    ],
                'fog':[

                ]
            },
            'test':{
                'snow':[
                    'x2021_0223_1756',
                ],
                'sun':[
                    'y0602_1330',
                    'y0613_1220',
                    'y0613_1242',
                ],
                'rain':[
                    'y0609_1947',
                    'y0609_1923',
                    'y2021_0228_1802'
                    ],
                'fog':[
                    
                ]
            },
            'train':{
                'snow':[
                    'x2021_0223_1856',
                    'x2021_0222_1810',
                    'x2021_0223_1310',
                    'x2021_0223_1316',
                    'y2021_0228_1807'
                ],
                'sun':[
                    'c2021_0228_1819',
                    'x2021_0222_1720',
                    'y0602_1228',
                    'y0602_1235',
                    'y0609_1959_2',
                    'y0613_1238',
                    'y0613_1248',
                    'y0613_1252',
                    'y0613_1304',
                    'y0613_1509',
                ],
                'rain':[
                    'x0613_1627',
                    'y0609_1639',
                    'y0609_1750',
                    'y0609_1954',
                    'y0613_1632',
                    'y0616_1750',
                    'y0616_1950'
                    ],
                'fog':[
                    
                ]
            }
        }
        self.all_scene = []
        self.weather = kwargs['weather']
        for weather in self.weather:
            self.all_scene += self.all_scene_map['train'][weather]
            self.all_scene += self.all_scene_map['val'][weather]
            self.all_scene += self.all_scene_map['test'][weather]

        
        super(ORFDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)

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
                         split):
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
            for scene in self.all_scene:
                sub_img_dir = os.path.join(img_dir, scene, 'image_data')
                if os.path.exists(sub_img_dir):
                    for img in mmcv.scandir(sub_img_dir, img_suffix, recursive=True):
                        img_info = dict(filename=os.path.join(scene, 'image_data', img))
                        if ann_dir is not None:
                            seg_map = img.replace(img_suffix, seg_map_suffix)
                            img_info['ann'] = dict(seg_map=os.path.join(scene, 'gt_image', seg_map))
                            img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images from {self.weather}', logger=get_root_logger())
        return img_infos
    
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

        file_stats_pd = pd.DataFrame(file_stats).T
        os.makedirs('./csv_result', exist_ok=True)
        file_stats_pd.to_csv(osp.join('./csv_result', 'eval_file_stats.csv'))
        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

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
