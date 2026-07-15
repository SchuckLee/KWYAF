# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DsecDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    # METAINFO = dict(        # 19分类
    #     classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    #              'traffic light', 'traffic sign', 'vegetation', 'terrain',
    #              'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    #              'motorcycle', 'bicycle'),
    #     palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    #              [190, 153, 153], [153, 153, 153], [250, 170,
    #                                                 30], [220, 220, 0],
    #              [107, 142, 35], [152, 251, 152], [70, 130, 180],
    #              [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    #              [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    METAINFO = dict(        # 11分类
        classes=('sky', 'building', 'fence', 'person', 'pole', 'road',
                 'sidewalk', 'vegetation', 'car', 'wall','traffic sign'
                 ),
        palette=[[70, 130, 180], [70, 70, 70], [190, 153, 153], [220, 20, 60],  # cityscapes中，road的train_Id也是0，而且cityscapes.py里也 没有设置过 reduce_zero_label
                 [153, 153, 153], [128, 64, 128], [244, 35, 232], [107, 142, 35],   # dsec的19类别和cityscpaes是一模一样的，应该也不用设置了。
                 [0, 0, 142], [102, 102, 156], [220, 220, 0],
                 ])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 # reduce_zero_label:bool=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
