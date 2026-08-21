#!/usr/bin/env python

"""Run an experiment with libdamp."""

import argparse
import sys
from pathlib import Path

import gin
import lightning.pytorch as pl
import mlflow
import torch

# make experiments directly discoverable from gin imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import libdamp  # noqa: F401 - imported for the side effect of registering gin-configurable classes

# Registered here so gin configs can reference `@PyTorchProfiler()` directly,
# e.g. to set `sort_by_key` for profiling output
gin.external_configurable(pl.profilers.PyTorchProfiler, name="PyTorchProfiler")


@gin.configurable("libdamp")
def libdamp_entry_point(experiment=gin.REQUIRED, train_dataset=gin.REQUIRED, val_dataset=None, test_dataset=None):
    """Entrance function for specifying an experiment and dataset for training in the gin config

    Registered under the gin name "libdamp" (rather than this function's own Python name) so that
    config files can use the natural-looking binding syntax:
    ```
    libdamp.experiment = @MyModel
    libdamp.train_dataset = @MyTrainDataset
    libdamp.val_dataset = @MyValDataset
    libdamp.test_dataset = @MyTestDataset
    ```

    Validation and test datasets are optional.
    """
    return experiment, train_dataset, val_dataset, test_dataset


def handle_cmd_args():
    """Register and parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog="libdamp", description="Framework for experiments with Differentiable Digital Signal Processing (DDSP) in PyTorch"
    )

    parser.add_argument("-c", "--config", type=str, dest="configs", nargs="+", help="specify one or multiple gin config file(s)")
    parser.add_argument(
        "--config-path", type=str, dest="config_paths", nargs="+", help="specify additional path(s) to search for included gin config files"
    )
    parser.add_argument("--seed", type=int, dest="fixed_seed", default=0, help="specify a fixed seed to create reproducible results (default: 0)")

    return parser.parse_args()


def run():
    """Main entry point to the program"""
    args = handle_cmd_args()

    # load gin configs
    if args.config_paths is not None:
        for path in args.config_paths:
            gin.add_config_file_search_path(path)

    with gin.unlock_config():
        for config in args.configs:
            gin.parse_config_file(config, skip_unknown=False)

    gin.finalize()

    if args.fixed_seed is not None:
        pl.seed_everything(args.fixed_seed, workers=True)

    # get experiment and datasets as configured in the gin file
    experiment, train_dataset, val_dataset, test_dataset = libdamp_entry_point()

    print(f"Running experiment '{experiment.name}'")

    # initialize data loaders
    train_dl = torch.utils.data.DataLoader(
        train_dataset, num_workers=16, persistent_workers=True, batch_size=experiment.batch_size, shuffle=experiment.shuffle_data
    )
    val_dl = None
    if val_dataset is not None:
        val_dl = torch.utils.data.DataLoader(val_dataset, num_workers=16, persistent_workers=True, batch_size=experiment.batch_size, shuffle=False)

    loggers = []
    callbacks = []
    if not experiment.fast_dev_run:
        # create results folder
        (experiment.save_path / experiment.name).mkdir(parents=True, exist_ok=True)

        # save operative config
        with open(experiment.save_path / experiment.name / "operative_config.gin", "w", encoding="utf-8") as file:
            file.write(gin.operative_config_str())

        # prepare logging and Lightning callbacks
        loggers.append(pl.loggers.CSVLogger(save_dir=experiment.save_path, version="csv_log", name=experiment.name, flush_logs_every_n_steps=1000))
        if experiment.enable_tensorboard:
            loggers.append(pl.loggers.TensorBoardLogger(save_dir=experiment.save_path, name=experiment.name, version="tensorboard_log"))
        if experiment.enable_mlflow:
            # Save gin operative configuration as hyperparameters in MLFlow.
            operative_config = gin.config._OPERATIVE_CONFIG
            data = {}
            for (scope, name), values in operative_config.items():
                for k, v in values.items():
                    data[f"{scope}.{name}.{k}"] = v

            mlflow.set_tracking_uri(experiment.mlflow_tracking_uri)
            mlflow.pytorch.autolog(log_every_n_epoch=1, log_every_n_step=100, log_models=False, checkpoint=False)
            mlflow.set_experiment("libdamp")
            mlflow.set_tag("mlflow.runName", experiment.name)
            mlflow.log_params(data)

        if experiment.save_best_n_ckpts > 0:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=experiment.save_path / experiment.name / "checkpoints",
                    filename="ckpt_best-{epoch}-{step}-{val_loss:.2f}",
                    monitor="val_loss",
                    save_last=False,
                    save_top_k=experiment.save_best_n_ckpts,
                )
            )
        if experiment.save_last_epoch_ckpt:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=experiment.save_path / experiment.name / "checkpoints",
                    filename="ckpt_last-{epoch}-{step}",
                    save_last=True,
                    every_n_epochs=1,
                    save_on_train_epoch_end=True,
                )
            )
        if experiment.early_stopping:
            callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", patience=experiment.early_stopping_patience))

    # load weights from a previous model/checkpoint
    if experiment.load_checkpoint_weights is not None:
        print(f"Loading weights from '{experiment.load_checkpoint_weights}'...")
        prev = torch.load(experiment.load_checkpoint_weights, map_location=experiment.device)
        experiment.load_state_dict(prev["state_dict"])

    # run training
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=experiment.accelerator,
        devices=experiment.devices,
        max_epochs=experiment.max_epochs,
        val_check_interval=min(len(train_dl), experiment.val_check_interval),
        limit_val_batches=experiment.limit_val_batches,
        accumulate_grad_batches=experiment.accumulate_grad_batches,
        gradient_clip_val=experiment.gradient_clip_val,
        fast_dev_run=experiment.fast_dev_run,
        profiler=experiment.profiler,
    )
    trainer.fit(experiment, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=experiment.full_initial_checkpoint)

    # run optional test
    if test_dataset is not None and not experiment.fast_dev_run:
        test_dl = torch.utils.data.DataLoader(test_dataset, num_workers=4, persistent_workers=True, batch_size=experiment.batch_size, shuffle=False)

        trainer.test(experiment, dataloaders=test_dl, ckpt_path="best")
        trainer.test(experiment, dataloaders=test_dl, ckpt_path="last")


if __name__ == "__main__":
    run()
