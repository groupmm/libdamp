libdamp
=======

Framework for experiments with **D**\ ifferentiable **A**\ udio and **M**\ usic **P**\ rocessing in PyTorch.

Goals
-----

Differentiable digital signal processing (DDSP) [1]_ is an umbrella term for including *classical* DSP building blocks (such as filters, oscillators, and signal transforms) in deep learning pipelines.
``libdamp`` offers a collection of DSP building blocks written in `PyTorch <https://pytorch.org/>`__, so that they can used within its automatic differentiation framework.
In addition to the differentiable building blocks — oscillators, noise
generators, filters, envelopes, transforms, etc. — ``libdamp`` offers the infrastructure for running experiments, including configuration with `gin-config <https://github.com/google/gin-config>`__ datasets, losses, and training with `Lightning <https://lightning.ai/>`__.

The library grew out of research on differentiable wind instrument modeling using pulsetable synthesis, but its building blocks are intentionally generic and reusable for other audio and music processing tasks and DDSP-related research.

Publications
------------
The experiments provided in this repository accompany the following publications:

1. Simon Schwär, Christian Dittmar, Stefan Balke, and Meinard Müller. *Differentiable Pulsetable Synthesis for Wind Instrument Modeling.* In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP): 14792–14796, Barcelona, Spain, 2026.

If you use ``libdamp`` in your research, please consider citing one of the above works.

In addition, we provide code and experiment setups accompanying the following talks and workshop contributions:

1. Manuel Peters and Simon Schwär. *Differentiable Wind Instrument Synthesis with Pulsetables and Beyond.*  Workshop on Audio & Music Signal Processing (pre-ICASSP), Barcelona, Spain, 2026.
2. Simon Schwär, Christian Dittmar, Stefan Balke, Meinard Müller. *Comparing Differentiable Implementations of Classical Sound Synthesis Methods.* DAGA, Dresden, 2026.

Architecture
------------

``libdamp`` is organized around two complementary kinds of stateful audio building blocks, plus
the supporting code needed to train models built from them:

- :mod:`libdamp.generators` produce a new audio signal from given control parameters such as frequency or amplitude trajectories (e.g. :class:`~libdamp.generators.SinusoidalOsc`, :class:`~libdamp.generators.HarmonicOsc`, :class:`~libdamp.generators.TableOsc`).
- :mod:`libdamp.processors` modify an existing audio signal according to control parameters (e.g. :class:`~libdamp.processors.GainEnvelope`, :class:`~libdamp.processors.ButterworthLowPassFilter`).

Both provide a similar interface: ``update()`` sets new control parameters, ``generate()`` (for :mod:`libdamp.generators`) or
``process()`` (for :mod:`libdamp.processors`) produces the next chunk of audio, and ``clear()`` resets internal state.
This shared, stateful design lets generators and processors be called repeatedly without updating the parameters, e.g. if the frame rate of a real-time system is separate from the update rate of the parameter estimation.

The remaining subpackages support building and training models out of these blocks:

- ``libdamp.helpers`` provides the underlying functionalities that generators and processors are built on — filter design, interpolation, windowing, and frequency conversions.
- ``libdamp.datasets`` wraps various data sources for use as training data.
- ``libdamp.losses`` defines training objectives for comparing two audio signals (:class:`~libdamp.losses.rms.RMSLoss`, :class:`~libdamp.losses.mss.MSSLoss`).
- ``libdamp.models`` provides generic neural network building blocks (e.g. :class:`~libdamp.models.lstm.BiLSTM`) that are used to predict the control parameters within experiments.
- :class:`~libdamp.experiment.Experiment` ties all of the above together into a trainable `Lightning <https://lightning.ai/>`_ module, configured via `gin-config <https://github.com/google/gin-config>`_, and is run via the ``scripts/run.py`` command-line entry point — see :doc:`experiment` for the full training workflow.

References
----------

.. [1] Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, and Adam Roberts, "DDSP: Differentiable
   Digital Signal Processing", International Conference on Learning Representations (ICLR), 2020.


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
