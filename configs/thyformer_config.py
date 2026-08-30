"""
ThyFormer — Central Configuration
All hyperparameters, paths, and experiment settings in one place.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    data_root: str = "/home/jiban/Documents/TI-RACS/stanford_dataset/images"
    train_csv: str = "resplit_stanford_dataset/train.csv"
    val_csv: str = "resplit_stanford_dataset/val.csv"
    test_csv: str = "resplit_stanford_dataset/test.csv"
    medsam_masks_dir: str = "/home/jiban/Documents/TI-RACS/stanford_dataset/med_sam"
    image_size: int = 384
    in_channels: int = 3
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: ["T1", "T2", "T3", "T4", "T5"])
    random_seed: int = 42
    # Source frames are 802x1054. A plain Resize(size, size) squashes them
    # horizontally by 1.3x, which destroys the taller-than-wide ratio — itself a
    # scored ACR TI-RADS shape criterion. "letterbox" rescales the long side and
    # pads the short one with black (the natural US background), preserving it.
    # "stretch" restores the legacy anisotropic behaviour for ablation.
    resize_mode: str = "letterbox"  # letterbox | stretch
    # Training sampler. The nodule occupies a median 7.1% of the frame, and the
    # frame-weighted sampler draws len(ds)/5 frames per class per epoch: TR-1's
    # 52 frames (ONE clip) were being shown 45x each while TR-4's 58 clips got
    # 0.5x. "clip" balances over clips first, then draws a frame uniformly inside
    # the chosen clip, so no single clip can dominate an epoch.
    sampler: str = "clip"  # clip | frame | none
    # Hard ceiling on any single clip's share of a training epoch. Class
    # balancing alone cannot rescue a class backed by ONE clip: TR-1 has a single
    # clip in the whole dataset, so an exactly class-balanced sampler still hands
    # that one nodule 1/5 of every epoch. This caps it and redistributes the
    # surplus over the other clips. 134 train clips means uniform is ~0.75%, so
    # 5% still permits ~7x oversampling of the rare grades.
    max_clip_share: float = 0.05
    # Nodule ROI cropping. Polygon masks exist for every frame, so the crop is
    # free supervision-side information (it uses no label).
    #   off       — full frame, as before
    #   crop      — crop to the nodule bbox padded by roi_pad, then resize
    roi_crop: str = "crop"  # off | crop
    roi_pad: float = 1.6
    # Auxiliary ACR TI-RADS descriptor supervision (see LossConfig.delta).
    #   "metadata" — the radiologist's five ACR descriptors from metadata_csv.
    #                Present for all 192 clips with no missing values, and its
    #                ti_rads_level agrees with the training labels on every clip.
    #   "derived"  — intensity-derived echogenicity only, for when metadata is
    #                unavailable (weaker: a heuristic, and only 1 of 5 findings).
    #   "off"      — no auxiliary supervision; the ECA echo head goes back to
    #                receiving no gradient at all.
    echo_source: str = "metadata"  # metadata | derived | off
    metadata_csv: str = "/home/jiban/Documents/TI-RACS/stanford_dataset/metadata.csv"
    # Only used by the "derived" fallback: ratio of mean intensity inside the
    # nodule polygon to a surrounding parenchyma ring, binned hypo/iso/hyper.
    echo_iso_band: Tuple[float, float] = (0.90, 1.10)
    echo_ring_px: int = 40


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
    mixup_min_batch_size: int = 8  # mixup off below this batch size


@dataclass
class ModelConfig:
    backbone: str = "swin_tiny_patch4_window7_224"
    img_size: int = 384  # must match DataConfig.image_size; timm rebuilds window masks for it
    pretrained: bool = True
    stem_kernel_size: int = 3
    stem_out_channels: int = 96  # must match Swin-T embedding dim
    stem_patch_size: int = 4
    eca_reduction_ratio: int = 16
    # Echogenicity bins. 4 matches the ACR point scale in metadata.csv
    # (anechoic / hyper-iso / hypo / very hypo); the legacy 3 was the
    # hypo-iso-hyper split of the intensity-derived fallback.
    eca_echo_bins: int = 4
    # Auxiliary head predicting the five ACR TI-RADS descriptors from the stage-4
    # features. The grade is a deterministic function of these five, so this
    # supervises the rubric the label is computed from rather than only its
    # output — the strongest signal available for 134 training clips. Bins are the
    # observed level counts for composition/echogenicity/shape/margin/foci.
    descriptor_head: bool = True
    descriptor_bins: Tuple[int, ...] = (3, 4, 2, 3, 5)
    # ACR point value carried by each descriptor class, in the class order the
    # descriptor head emits. metadata.csv stores the ACR point values directly, so
    # these are exactly its observed levels — verify with load_descriptors().
    descriptor_point_values: Tuple[Tuple[int, ...], ...] = (
        (0, 1, 2),  # composition
        (0, 1, 2, 3),  # echogenicity
        (0, 3),  # shape
        (0, 2, 3),  # margin
        (0, 1, 2, 3, 4),  # echogenic foci
    )
    # ECA gating form. The legacy "sigmoid" gate sits at 0.500 at init (zero bias,
    # Xavier weights), so it halved every token before the *pretrained* Swin saw
    # it — a measured 0.5005 output/input norm ratio, i.e. the backbone spends its
    # frozen epochs undoing an input-scale shift. "residual_tanh" computes
    # 1 + tanh(fc2(h)) with fc2 zero-initialised: exactly identity at init, still
    # bounded in [0,2] so it can both suppress and amplify. Same state-dict
    # shapes, so old checkpoints still load either way.
    eca_gate: str = "residual_tanh"  # residual_tanh | sigmoid
    # Classification head. "corn" swaps the 5-way softmax for K-1 cumulative
    # binary logits (Cao et al., CORN). TI-RADS is ordinal and the model already
    # leads on the ordinal metrics (grade MAE 0.592 vs Swin's 0.674) while losing
    # on macro AUC, so this targets the axis where the signal actually is.
    #   "concept_bottleneck" routes the grade through the ACR point table via the
    #   five predicted descriptors, with NO direct feature->grade path. The
    #   concept predictions are then faithful by construction and a clinician can
    #   overwrite one and read off the new grade (ACRScoringLayer.intervene) —
    #   the intervenability result that a parallel aux head cannot support.
    head_type: str = "softmax"  # softmax | corn | concept_bottleneck
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
    # Echogenicity auxiliary CE weight. The ECA head's `echo_proj` output was
    # returned by forward() but never entered the loss, so its grad was literally
    # None — the "clinically interpretable echogenicity bins" were frozen random
    # numbers. delta > 0 supervises it against the pseudo-label derived from the
    # nodule polygon (DataConfig.echo_iso_band), turning it into a real auxiliary
    # task with a real ablation. 0.0 reproduces the legacy (dead) behaviour.
    delta: float = 0.2
    boundary_warmup_epochs: int = 2
    label_smoothing: float = 0.1
    # NOTE: only applied when the sampler is not already balancing classes —
    # stacking inverse-frequency weights on top of a balanced sampler
    # double-corrects the imbalance. See build_loss / build_class_weights.
    use_class_weights: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 16  # 720x720 has ~10x the tokens of 224; 8GB GPU limit
    accum_steps: int = 1  # effective batch = batch_size * accum_steps (2x8=16)
    num_workers: int = 8
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
    # Decaying LayerNorm gains and biases toward zero shrinks a transformer's
    # normalisation statistics for no regularisation benefit; the standard recipe
    # excludes every 1-D parameter from weight decay.
    wd_exclude_norm_bias: bool = True
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    freeze_backbone_epochs: int = 2  # epochs the pretrained Swin stages stay frozen
    min_lr: float = 1e-6
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"  # also ranks checkpoints
    early_stopping_mode: str = "min"
    selection_smoothing_epochs: int = 3  # trailing-mean window; 1 = off
    checkpoint_dir: str = (
        "/home/jiban/Documents/TI-RACS/artifacts/v4_720_crop_thyformer_test/checkpoints"
    )
    save_top_k: int = 3
    log_dir: str = "/home/jiban/Documents/TI-RACS/artifacts/v4_720_crop_thyformer_test/logs"
    log_interval: int = 10
    use_wandb: bool = False
    project_name: str = "thyformer"
    experiment_name: str = "thyformer_v2"
    seed: int = 42


@dataclass
class EvaluationConfig:
    primary_metric: str = "val_auc"
    output_dir: str = "/home/jiban/Documents/TI-RACS/artifacts/v4_720_crop_thyformer_test"
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
