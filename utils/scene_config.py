"""
公共场景配置文件

定义数据集的 unknown 场景和异常场景，供多个脚本统一引用。
修改此文件后，所有引用的脚本将自动使用新的配置。

使用方法:
    from utils.scene_config import DATASET_UNKNOWN_SCENES, ABNORMAL_SCENES
"""

# 数据集 unknown 场景配置
# - unknown 场景用于评估模型在未见场景上的泛化能力
# - 这些场景在训练集中未出现，仅用于测试
DATASET_UNKNOWN_SCENES = {
    'road3d': [
        '2021-0403-1744', '0602-1107',
        # '2021-0222-1743', '2021-0403-1858',
        '2021-0223-1857', '2021-0403-1736'
    ],
    'orfd': [
        '0609-1923', '2021-0223-1756'
    ],
    'orfd2road': [
        # '1309',
        '0609-1924', '0609-1923',
        '2021-0403-1736', '2021-0223-1857'
    ],
}

# 异常场景配置
# - 这些场景可能存在标注问题或极端条件
# - 可选择在评估时排除
ABNORMAL_SCENES = [
    '2021-0223-1454', '2021-0223-1770', '2021-0403-1858', '2021-0222-1744-2'
]


def get_unknown_scenes(dataset_name: str) -> list:
    """获取指定数据集的 unknown 场景列表

    Args:
        dataset_name: 数据集名称 ('road3d', 'orfd', 'orfd2road')

    Returns:
        unknown 场景列表，如果数据集不存在则返回空列表
    """
    return DATASET_UNKNOWN_SCENES.get(dataset_name, [])


def is_unknown_scene(scene_name: str, dataset_name: str) -> bool:
    """判断场景是否为 unknown 场景

    Args:
        scene_name: 场景名称
        dataset_name: 数据集名称

    Returns:
        是否为 unknown 场景
    """
    unknown_scenes = get_unknown_scenes(dataset_name)
    return any(unknown in scene_name for unknown in unknown_scenes)


def is_abnormal_scene(scene_name: str) -> bool:
    """判断场景是否为异常场景

    Args:
        scene_name: 场景名称

    Returns:
        是否为异常场景
    """
    return any(abnormal in scene_name for abnormal in ABNORMAL_SCENES)
