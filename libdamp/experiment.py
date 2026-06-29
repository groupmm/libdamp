"""Base class for an experiment that can be run with `run.py`"""

from datetime import datetime
from pathlib import Path

import gin
import lightning as L
import mlflow
import soundfile as sf


@gin.configurable
class Experiment(L.LightningModule):
    """Base class for an experiment that can be run with `run.py`.

    Defaults for all properties are set in `configs/base.gin`.
    """

    def __init__(
        self,
        accelerator=gin.REQUIRED,
        accumulate_grad_batches=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        devices=gin.REQUIRED,
        early_stopping=gin.REQUIRED,
        early_stopping_patience=gin.REQUIRED,
        enable_tensorboard=gin.REQUIRED,
        enable_mlflow=gin.REQUIRED,
        enable_audio_logging=gin.REQUIRED,
        audio_log_batch_idx=gin.REQUIRED,
        audio_log_item_idx=gin.REQUIRED,
        fast_dev_run=gin.REQUIRED,
        full_initial_checkpoint=gin.REQUIRED,
        gradient_clip_val=gin.REQUIRED,
        limit_val_batches=gin.REQUIRED,
        load_checkpoint_weights=gin.REQUIRED,
        max_epochs=gin.REQUIRED,
        mlflow_tracking_uri=gin.REQUIRED,
        name=gin.REQUIRED,
        profiler=gin.REQUIRED,
        save_best_n_ckpts=gin.REQUIRED,
        save_last_epoch_ckpt=gin.REQUIRED,
        save_path=gin.REQUIRED,
        shuffle_data=gin.REQUIRED,
        val_check_interval=gin.REQUIRED,
    ):
        """Initialize experiment with training configuration parameters.

        All parameters are typically specified via gin configuration files rather than
        directly. The `gin.REQUIRED` marker indicates that a value must be provided
        either through a config file or programmatically before use.

        Parameters
        ----------
        accelerator : str
            PyTorch Lightning accelerator type (e.g., 'gpu', 'cpu')
        accumulate_grad_batches : int
            Number of batches to accumulate gradients over before updating weights
        batch_size : int
            Batch size for training and validation
        devices : int or list
            Number of devices or list of device indices to use
        early_stopping : bool
            Whether to use early stopping during training
        early_stopping_patience : int
            Number of validation checks with no improvement before stopping
        enable_tensorboard : bool
            Whether to log metrics to TensorBoard
        enable_mlflow : bool
            Whether to log metrics to MLflow
        enable_audio_logging : bool
            Whether to log audio predictions and references
        audio_log_batch_idx : int
            Batch index to log audio from (e.g., 0 for first batch)
        audio_log_item_idx : int
            Item index within the batch to log audio from (e.g., 0 for first item)
        fast_dev_run : bool or int
            Quickly run through training loop for debugging (number of batches if int)
        full_initial_checkpoint : str or None
            Path to checkpoint to load full model from (weights and architecture)
        gradient_clip_val : float or None
            Gradient clipping magnitude
        limit_val_batches : float or int
            Limit validation batches (fraction of 0-1 or number of batches)
        load_checkpoint_weights : str or None
            Path to checkpoint to load only weights from
        max_epochs : int
            Maximum number of training epochs
        mlflow_tracking_uri : str
            MLflow tracking URI for logging metrics, parameters and artifacts
        name : str
            Experiment name (can include '{date}' placeholder)
        profiler : str or None
            PyTorch Lightning profiler to use
        save_best_n_ckpts : int
            Number of best checkpoints to keep
        save_last_epoch_ckpt : bool
            Whether to save the last epoch checkpoint
        save_path : str
            Directory to save experiment in
        shuffle_data : bool
            Whether to shuffle training data
        val_check_interval : float or int
            How often to validate during training epoch
        """
        super().__init__()

        self.accelerator = accelerator
        self.accumulate_grad_batches = accumulate_grad_batches
        self.batch_size = batch_size
        self.devices = devices
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.enable_audio_logging = enable_audio_logging
        self.audio_log_batch_idx = audio_log_batch_idx
        self.audio_log_item_idx = audio_log_item_idx
        self.enable_tensorboard = enable_tensorboard
        self.enable_mlflow = enable_mlflow
        self.fast_dev_run = fast_dev_run
        self.full_initial_checkpoint = full_initial_checkpoint
        self.gradient_clip_val = gradient_clip_val
        self.limit_val_batches = limit_val_batches
        self.load_checkpoint_weights = load_checkpoint_weights
        self.max_epochs = max_epochs
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.name = name
        self.profiler = profiler
        self.save_best_n_ckpts = save_best_n_ckpts
        self.save_last_epoch_ckpt = save_last_epoch_ckpt
        self.save_path = Path(save_path)
        self.shuffle_data = shuffle_data
        self.val_check_interval = val_check_interval

        self._audio_logging_not_enabled_warning_shown = False
        self._audio_logging_reference_saved = False

    @property
    def name(self):
        """Replace '{date}' in name with the current date and time"""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value.replace("{date}", datetime.now().strftime("%Y%m%d_%H%M%S"))

    def log_audio(self, name: str, prediction, reference=None, current_batch_idx=None, split="val"):
        """
        Helper function to log audio prediction and reference (if provided) after each training, validation or testing epoch.
        The item to log is specified by `self.audio_log_item_idx` and `self.audio_log_batch_idx`.
        Audio files are saved to `self.save_path / self.name / "audio"` and optionally logged to MLflow if `self.enable_mlflow` is True.
        """

        if not getattr(self, "enable_audio_logging", False):
            if not self._audio_logging_not_enabled_warning_shown:
                print("Warning: Attempted to log audio but `enable_audio_logging` is False. Set `enable_audio_logging=True` to enable.")
                self._audio_logging_not_enabled_warning_shown = True
            return

        if current_batch_idx == self.audio_log_batch_idx:
            fs = getattr(self, "fs", None)
            if fs is None:
                raise ValueError("Warning: Could not log audio: No sample rate provided and `self.fs` is not defined.")

            if not self._audio_logging_reference_saved:
                fpath = self.save_path / self.name / "audio" / f"{name}_{split}_gt.wav"
                fpath.parent.mkdir(parents=True, exist_ok=True)
                if reference is not None:
                    sf.write(fpath, reference[self.audio_log_item_idx].cpu().numpy(), int(fs))
                    if self.enable_mlflow:
                        mlflow.log_artifact(fpath, artifact_path="audio")
                self._audio_logging_reference_saved = True

            fpath = self.save_path / self.name / "audio" / f"e{self.current_epoch:02d}_{split}_pred_{name}.wav"
            sf.write(fpath, prediction[self.audio_log_item_idx].cpu().numpy(), int(fs))
            if self.enable_mlflow:
                mlflow.log_artifact(fpath, artifact_path="audio")
