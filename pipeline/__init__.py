"""Airline passenger re-accommodation pipeline.

Typical usage::

    from pipeline import run_pipeline

    assignments, unbooked = run_pipeline(
        pnr="pnr.csv",
        cancelled="target.csv",
        available="available.csv",
    )

For finer control, import the config dataclasses::

    from pipeline import run_pipeline, QUBOWeights, PreprocessingConfig

See ``run_pipeline`` docstring for all available parameters.
"""

import logging

from .config import MultiLegConfig, PreprocessingConfig, QUBOWeights
from .runner import ReaccommodationPipeline, run_pipeline
from .types import BatchStrategy, CandidateFilterLevel

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "run_pipeline",
    "ReaccommodationPipeline",
    "PreprocessingConfig",
    "MultiLegConfig",
    "QUBOWeights",
    "BatchStrategy",
    "CandidateFilterLevel",
]
