libdamp
=======

Framework for experiments with differentiable audio and music processing (DDSP) in PyTorch.

Goals
-----

Differentiable digital signal processing (DDSP) replaces classic, non-differentiable audio
synthesis and processing algorithms with PyTorch implementations that can be optimized end-to-end
with gradient descent, typically driven by a neural network that predicts their control
parameters. ``libdamp`` collects such differentiable building blocks — oscillators, noise
generators, filters, and envelopes — along with the supporting datasets, losses, and training
infrastructure needed to combine them into trainable audio synthesis and processing models.

The library grew out of research on differentiable wind instrument modeling (see the citation in
the `project README <https://github.com/groupmm/libdamp>`_), but its building blocks are
intentionally generic and reusable for other DDSP-style audio and music processing tasks.

Architecture
------------

``libdamp`` is organized around two complementary kinds of stateful audio building blocks, plus
the supporting code needed to train models built from them:

- :mod:`libdamp.generators` produce an audio signal from scratch, given control parameters such as
  frequency or amplitude trajectories (e.g. :class:`~libdamp.generators.SinusoidalOsc`,
  :class:`~libdamp.generators.HarmonicOsc`, :class:`~libdamp.generators.TableOsc`).
- :mod:`libdamp.processors` transform an existing audio signal according to control parameters
  (e.g. :class:`~libdamp.processors.Envelope`, :class:`~libdamp.processors.LowPassFilter`).

Both follow the same lifecycle: ``update()`` sets new control parameters, ``generate()`` /
``process()`` produces the next chunk of audio from them, and ``clear()`` resets internal state.
This shared, stateful design lets generators and processors be called repeatedly with successive
chunks of control parameters while producing a continuous output signal, which is what makes them
suitable for block-wise, streaming-friendly training and inference.

The remaining subpackages support building and training models out of these blocks:

- ``libdamp.helpers`` provides the underlying differentiable DSP math that generators and
  processors are built on — filter design, interpolation, windowing, and frequency conversions.
- ``libdamp.datasets`` wraps various audio data sources for use as training data.
- ``libdamp.losses`` defines training objectives for comparing two audio signals
  (e.g. :class:`~libdamp.losses.rms.RMSLoss`, :class:`~libdamp.losses.mss.MSSLoss`).
- ``libdamp.models`` provides generic neural network backbones (e.g.
  :class:`~libdamp.models.lstm.BiLSTM`) that are typically used to predict the control parameters
  fed into generators and processors.
- :class:`~libdamp.experiment.Experiment` ties all of the above together into a trainable
  `Lightning <https://lightning.ai/>`_ module, configured via `gin <https://github.com/google/gin-config>`_,
  and is run via the ``scripts/run.py`` command-line entry point — see :doc:`experiment` for the
  full training workflow.

.. toctree::
   :maxdepth: 2
   :caption: Features

   experiment

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   generators
   processors
   datasets
   helpers
   losses
   models

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
