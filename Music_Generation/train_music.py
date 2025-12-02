from Trainer import Trainer
from .music_dataset import MusicDataset
from Models import MLPNetwork
from Trainer.losses import LinearLoss
from Trainer.optimizer import SGD


class MusicTrainer():
    def __init__(self,songs,window_size = 8,hidden_layer=[64,64],epochs = 50,batch_size=32,loss_fn=None,optimizer=SGD,learning_rate = 0.01):
        
        self.dataset = MusicDataset(songs,window_size=window_size)
        
        self.model = MLPNetwork(
            input_dim =window_size,
            n_neurons=hidden_layer,
            output_size = 1,
            classification="none")
      
        self.optimizer = optimizer if optimizer is not None else SGD(learning_rate=0.01)
        self.loss_fn   = loss_fn if loss_fn is not None else LinearLoss()
        
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            epochs=epochs
        )
        
    def fit(self):
        self.trainer.fit(self.dataset.x,self.dataset.y,batch_size=self.batch_size)
        