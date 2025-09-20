# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GooseDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('notraversable', 'traversable')
    # CLASSES = ('Vehicle-accessible areas', 'High-risk terrain blocks')
    PALETTE = [[0,0,0],[0,128,0]]

    def __init__(self, **kwargs):
        super(GooseDataset, self).__init__(
            img_suffix='_vis.png',
            seg_map_suffix='_labelids.png',
            # reduce_zero_label=True, ###
            **kwargs)