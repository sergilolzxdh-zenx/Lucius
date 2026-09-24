"""Training preparation (sections 44-48).

What is implemented: converting exported datasets into imitation-learning formats (behaviour
cloning pairs and chat-style SFT samples) and an evidence-based advisor that says *whether*
training is worth considering. What is an interface only: the trainers themselves. No trainer
backend ships with Lucius; ``TrainerRegistry`` is empty until one is registered, and nothing
here launches training on its own.
"""

from lucius.training.advisor import TrainingAdvisor, TrainingSignal
from lucius.training.formats import behaviour_cloning_pairs, sft_samples, write_training_files
from lucius.training.interfaces import TrainerBackend, TrainerRegistry, TrainingJobSpec, TrainingStrategy

__all__ = ["TrainerBackend", "TrainerRegistry", "TrainingAdvisor", "TrainingJobSpec", "TrainingSignal",
           "TrainingStrategy", "behaviour_cloning_pairs", "sft_samples", "write_training_files"]
