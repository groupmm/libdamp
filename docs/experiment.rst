Running experiments
====================

The :class:`~libdamp.experiment.Experiment` base class and the ``scripts/run.py`` entry point
together provide a complete, configuration-driven training loop for models built from
:mod:`libdamp.generators` and :mod:`libdamp.processors`, so that individual experiments only need
to define the model itself, not any training boilerplate.

The Experiment class
---------------------

:class:`~libdamp.experiment.Experiment` is a `Lightning <https://lightning.ai/>`_
``LightningModule`` subclass that defines the training/validation/test step structure common to
all ``libdamp`` experiments. To define a new experiment, subclass it and implement the usual
Lightning hooks (``forward()``, ``training_step()``, ``validation_step()``, ``configure_optimizers()``,
...), composing the model out of :mod:`libdamp.generators`, :mod:`libdamp.processors`, and
``libdamp.models``::

    import gin
    import libdamp

    @gin.configurable
    class MyExperiment(libdamp.Experiment):
        def __init__(self, fs, **kwargs):
            super().__init__(**kwargs)
            self.fs = fs
            self.osc = libdamp.generators.HarmonicOsc(N=512, fs=fs)
            self.envelope = libdamp.processors.Envelope()
            self.loss_fn = libdamp.RMSLoss()

        def forward(self, f0, amplitudes, gain):
            self.osc.update(f0=f0, a=amplitudes)
            x = self.osc.generate()
            self.envelope.update(g=gain)
            return self.envelope.process(x)

        def training_step(self, batch, batch_idx):
            y, f0, amplitudes, gain = batch
            y_hat = self(f0, amplitudes, gain)
            loss = self.loss_fn(y, y_hat)
            self.log("train_loss", loss)
            return loss

All constructor parameters of :class:`~libdamp.experiment.Experiment` itself (batch size, number
of epochs, checkpointing, logging, ...) are marked ``gin.REQUIRED`` and are meant to be supplied
through a `gin <https://github.com/google/gin-config>`_ configuration file rather than hardcoded,
so the same experiment class can be reused across many training runs that only differ in
configuration. :meth:`~libdamp.experiment.Experiment.log_audio` is a ready-made helper for logging
example audio (predictions and, once, the reference) to disk and optionally MLflow after each
epoch.

Running an experiment with run.py
----------------------------------

``scripts/run.py`` is the command-line entry point that turns an
:class:`~libdamp.experiment.Experiment` subclass and a gin config into a full training run. The
gin config selects which experiment and datasets to use via the ``libdamp()`` binding::

    libdamp.experiment = @MyExperiment()
    libdamp.train_dataset = @MyTrainDataset()
    libdamp.val_dataset = @MyValDataset()    # optional
    libdamp.test_dataset = @MyTestDataset()  # optional

    MyExperiment.fs = 16000.0
    # ... plus all the gin.REQUIRED Experiment parameters (batch_size, max_epochs, save_path, ...)

and is then run with:

.. code-block:: console

    python scripts/run.py --config path/to/config.gin --seed 0

Command-line options:

- ``-c, --config`` (required): one or more gin config files, merged in order.
- ``--config-path``: additional directories to search for gin files included from a config (via
  gin's ``include`` statement).
- ``--seed``: fixed random seed for reproducible runs (default: ``0``).

Given the resolved configuration, ``run.py`` then takes care of the rest of the training
boilerplate so individual experiments don't have to:

- builds the train/validation ``DataLoader``\\ s from the configured datasets;
- sets up logging (CSV always, plus TensorBoard and/or MLflow if enabled) and writes the
  fully-resolved ("operative") gin config alongside the run for reproducibility;
- configures checkpointing (best-``n`` and/or last-epoch) and early stopping based on
  ``val_loss``, and optionally resumes from a previous checkpoint's weights;
- runs training via a Lightning ``Trainer`` configured from the experiment's parameters
  (accelerator, devices, gradient accumulation/clipping, validation interval, ``fast_dev_run``
  for quick smoke tests, ...);
- runs a final test pass with both the best and the last checkpoint, if a test dataset was given.

API reference
--------------

.. automodule:: libdamp.experiment
   :members:
   :show-inheritance:
