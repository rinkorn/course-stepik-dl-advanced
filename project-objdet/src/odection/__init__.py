"""Object Detection Pytorch Docstring"""

__version__ = "1.0.0"

from .dataset_coco import CocoDataset, collate_fn
from .loss import Loss
from .model_evaluate import evaluate
from .ssd import SSD, ResNet
from .transform import SSDTransformer
from .utils import Encoder, coco_classes, colors, generate_dboxes

__all__ = [
    "coco_classes",
    "CocoDataset",
    "collate_fn",
    "colors",
    "Encoder",
    "evaluate",
    "generate_dboxes",
    "Loss",
    "ResNet",
    "SSD",
    "SSDTransformer",
]
