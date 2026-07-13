"""
ThyFormer — Central Configuration
All hyperparameters, paths, and experiment settings in one place.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    data_root: str = "/home/jiban/Documents/TI-RACS/stanford_dataset/images"
    train_csv: str = "/home/jiban/Documents/TI-RACS/stanford_dataset/train.csv"
    val_csv: str = "/home/jiban/Documents/TI-RACS/stanford_dataset/val.csv"
    test_csv: str = "/home/jiban/Documents/TI-RACS/stanford_dataset/test.csv"
    medsam_masks_dir: str = "/home/jiban/Documents/TI-RACS/stanford_dataset/med_sam"
    image_size: int = 720
    in_channels: int = 3
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: ["T1", "T2", "T3", "T4", "T5"])
    random_seed: int = 42


@dataclass
class AugmentationConfig:
    random_flip_p: float = 0.5
    random_rotate_limit: int = 15
    elastic_alpha: float = 120.0
    elastic_sigma: float = 6.0
    elastic_p: float = 0.3
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    speckle_noise_var: float = 0.05
    speckle_noise_p: float = 0.4
    brightness_limit: float = 0.15
    contrast_limit: float = 0.15
    mixup_alpha: float = 0.2
    mixup_p: float = 0.3


@dataclass
class ModelConfig:
    backbone: str = "swin_tiny_patch4_window7_224"
    img_size: int = 720  # must match DataConfig.image_size; timm rebuilds window masks for it
    pretrained: bool = True
    stem_kernel_size: int = 3
    stem_out_channels: int = 96  # must match Swin-T embedding dim
    stem_patch_size: int = 4
    eca_reduction_ratio: int = 16
    eca_echo_bins: int = 3  # hypo / iso / hyper
    cls_dropout: float = 0.2
    fpn_out_channels: int = 128
    seg_num_classes: int = 1
    freeze_stages: int = 0
    drop_path_rate: float = 0.1


@dataclass
class LossConfig:
    alpha: float = 1.0  # CE weight
    beta: float = 0.5  # Dice weight
    gamma: float = 0.3  # Boundary weight
    boundary_warmup_epochs: int = 2
    label_smoothing: float = 0.1
    use_class_weights: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 2  # 720x720 has ~10x the tokens of 224; 8GB GPU limit
    num_workers: int = 4
    pin_memory: bool = True
    fp16: bool = True  # master switch for mixed-precision (AMP)
    # AMP dtype: "bf16" has the same exponent range as fp32 (overflow-safe, no
    # loss scaling needed) — recommended at 720px. "fp16" overflows at >65504
    # and was causing the loss → inf → nan divergence around epoch 3.
    amp_dtype: str = "bf16"
    gradient_clip_val: float = 1.0
    optimizer: str = "adamw"
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    checkpoint_dir: str = "artifacts/v2_thyformer_v2_720/checkpoints"
    save_top_k: int = 3
    log_dir: str = "artifacts/v2_thyformer_v2_720/logs"
    log_interval: int = 10
    use_wandb: bool = False
    project_name: str = "thyformer"
    experiment_name: str = "thyformer_v2"
    seed: int = 42


@dataclass
class EvaluationConfig:
    primary_metric: str = "val_auc"
    # Root folder for evaluation runs. thyformer_evaluate.py writes a
    # timestamped subfolder here with metrics, figures (ROC/PR/confusion
    # matrix/per-class bars), per-sample predictions, GradCAM heatmaps,
    # DeLong/kappa results, and the full run configuration.
    output_dir: str = "artifacts/v2_thyformer_v2_720/evaluation"
    num_gradcam_samples: int = 50


@dataclass
class ThyFormerConfig:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def get_config() -> ThyFormerConfig:
    return ThyFormerConfig()
