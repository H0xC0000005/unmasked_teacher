
from dataclasses import dataclass

import cv2

from single_modality.action_detection.alphaction.dataset.transforms import video_transforms as video_trans

cv2.setNumThreads(0)


@dataclass
class TransformsCfg:
    MIN_SIZE_TRAIN: int = 256
    MAX_SIZE_TRAIN: int = 464
    MIN_SIZE_TEST: int = 256
    MAX_SIZE_TEST: int = 464
    PIXEL_MEAN = [122.7717, 115.9465, 102.9801]
    PIXEL_STD = [57.375, 57.375, 57.375]
    TO_BGR: bool = False

    FRAME_NUM: int = 8  # 8
    FRAME_SAMPLE_RATE: int = 8  # 8

    COLOR_JITTER: bool = True
    HUE_JITTER: float = 20.0
    SAT_JITTER: float = 0.1
    VAL_JITTER: float = 0.1


def build_transforms(cfg=TransformsCfg(), is_train=True, sparse=False):
    # build transforms for training of testing
    if is_train:
        min_size = cfg.MIN_SIZE_TRAIN
        max_size = cfg.MAX_SIZE_TRAIN
        color_jitter = cfg.COLOR_JITTER
        flip_prob = 0.5
    else:
        min_size = cfg.MIN_SIZE_TEST
        max_size = cfg.MAX_SIZE_TEST
        color_jitter = False
        flip_prob = 0

    frame_num = cfg.FRAME_NUM
    sample_rate = cfg.FRAME_SAMPLE_RATE
    if sparse:
        print("Use sparse sampling")
        frame_span = 300
    else:
        print("Use dense sampling")
        frame_span = frame_num * sample_rate
    print(f"Frame span: {frame_span}")

    if color_jitter:
        color_transform = video_trans.ColorJitter(
            cfg.HUE_JITTER, cfg.SAT_JITTER, cfg.VAL_JITTER
        )
    else:
        color_transform = video_trans.Identity()

    to_bgr = cfg.TO_BGR
    normalize_transform = video_trans.Normalize(
        mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD, to_bgr=to_bgr
    )

    transform = video_trans.Compose(
        [
            video_trans.TemporalCrop(frame_num, sample_rate, sparse=sparse),
            video_trans.Resize(min_size, max_size),
            video_trans.RandomClip(is_train),
            color_transform,
            video_trans.RandomHorizontalFlip(flip_prob),
            video_trans.ToTensor(),
            normalize_transform,
        ]
    )

    print(f"Transform:\n{transform}")

    return frame_span, transform
