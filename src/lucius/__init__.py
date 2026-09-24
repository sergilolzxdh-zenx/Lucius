"""Lucius: a continual-learning Blender GUI agent.

The package is organised along the learning lifecycle:

    recorder -> trajectory -> segmentation -> intent -> skills -> memory
    -> retrieval -> planner -> executor -> evaluation -> practice -> dataset -> training

Every subsystem persists through :mod:`lucius.storage` and announces what it did on
:mod:`lucius.events`, so a failure in one stage never destroys the raw evidence of another.
"""

__version__ = "0.1.0"
