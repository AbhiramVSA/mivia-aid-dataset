from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PathsConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    train_csv: Path = Path("Train_GT.csv")
    val_csv: Path = Path("MIVIA-AID-Dataset/Val_GT.csv")
    train_videos_dir: Path = Path("train-002/train")
    val_videos_dir: Path = Path("val")
    checkpoints_dir: Path = Path("artifacts/checkpoints")
    features_dir: Path = Path("artifacts/features")
    logs_dir: Path = Path("artifacts/logs")

    def resolve(self) -> "PathsConfig":
        return PathsConfig(
            project_root=self.project_root,
            train_csv=(self.project_root / self.train_csv).resolve(),
            val_csv=(self.project_root / self.val_csv).resolve(),
            train_videos_dir=(self.project_root / self.train_videos_dir).resolve(),
            val_videos_dir=(self.project_root / self.val_videos_dir).resolve(),
            checkpoints_dir=(self.project_root / self.checkpoints_dir).resolve(),
            features_dir=(self.project_root / self.features_dir).resolve(),
            logs_dir=(self.project_root / self.logs_dir).resolve(),
        )


@dataclass(slots=True)
class VideoSamplingConfig:
    sample_fps: int = 8
    clip_num_frames: int = 16
    clip_stride_frames: int = 4
    resize_short_side: int = 256
    crop_size: int = 224

    @property
    def clip_span_seconds(self) -> float:
        return self.clip_num_frames / float(self.sample_fps)

    @property
    def clip_stride_seconds(self) -> float:
        return self.clip_stride_frames / float(self.sample_fps)


@dataclass(slots=True)
class Stage1TrainConfig:
    batch_size: int = 2
    num_epochs: int = 10
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.05
    num_workers: int = 4
    grad_accum_steps: int = 1
    amp: bool = True
    max_steps_per_sample: int = 8
    window_stride_steps: int = 4


@dataclass(slots=True)
class Stage2TrainConfig:
    batch_size: int = 1
    num_epochs: int = 20
    warmup_epochs: int = 2
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.05
    num_workers: int = 2
    grad_accum_steps: int = 1
    amp: bool = True
    lambda_video_values: tuple[float, ...] = (0.25, 0.5, 1.0)
    max_steps_per_sample: int = 16
    window_stride_steps: int = 8
    onset_sigma_seconds: float = 1.0
    target_mode: str = "cumulative"
    monotonic_loss_weight: float = 0.1
    temporal_aux_loss_weight: float = 0.2
    temporal_distance_bin_edges_s: tuple[float, float, float] = (-5.0, 0.0, 5.0)
    hard_negative_multiplier: float = 2.0


@dataclass(slots=True)
class PostprocessConfig:
    median_kernel_size: int = 3
    tau_empty_grid: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
    tau_start_grid: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    tau_keep_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    tau_video_grid: tuple[float, ...] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)
    consecutive_hits_grid: tuple[int, ...] = (1, 2, 3)
    default_tau_empty: float = 0.4
    default_tau_start: float = 0.5
    default_tau_keep: float = 0.3
    default_tau_video: float = 0.0
    default_min_consecutive_steps: int = 1
    prediction_mode: str = "cumulative"


@dataclass(slots=True)
class ModelConfig:
    backbone_name: str = "MCG-NJU/videomae-base"
    hidden_size: int = 768
    temporal_model: str = "transformer"
    temporal_channels: tuple[int, int, int] = (512, 512, 256)
    dropout: float = 0.1
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ffn_dim: int = 2048


@dataclass(slots=True)
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    video: VideoSamplingConfig = field(default_factory=VideoSamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    stage1: Stage1TrainConfig = field(default_factory=Stage1TrainConfig)
    stage2: Stage2TrainConfig = field(default_factory=Stage2TrainConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    def resolved(self) -> "ExperimentConfig":
        return ExperimentConfig(
            paths=self.paths.resolve(),
            video=self.video,
            model=self.model,
            stage1=self.stage1,
            stage2=self.stage2,
            postprocess=self.postprocess,
        )
