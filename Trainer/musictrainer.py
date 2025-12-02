import numpy as np
from . import Trainer
from Models.MLPMusicGen import MLPMusicGen
from .losses import *


class MusicTrainer(Trainer):
    """
    Trainer specialized for MLPMusicGen.
    Uses sequences of integer notes and trains on (context_length -> next_note) pairs.
    """

    def __init__(self, model: MLPMusicGen, epochs: int = 10):
        # Use CrossEntropyLoss by default for classification
        super().__init__(model=model,
                         optimizer=None,              
                         loss_fn=CrossEntropyLoss(),
                         epochs=epochs)
        
        
        
        