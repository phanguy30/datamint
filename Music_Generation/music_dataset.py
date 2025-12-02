import random
from .midi_to_dataset import MidiDatasetLoader
import numpy as np

class MusicDataset(MidiDatasetLoader):
    def __init__(self,folder_path,window_size=16,shuffle=True,seed=42):
        """
        Args:
            folder_path (path): folder of midi files
            window_size (int, optional): length of input sequence. Defaults to 16.
            shuffle (bool, optional): if the data should be shuffled or not. Defaults to True.
            seed (int, optional): set seed for reproducibility. Defaults to 42.
        """
        
        
        self.window_size= window_size
        self.shuffle = shuffle
        self.seed= seed
        
        super().__init__(folder_path)
        
        self.x,self.y = self._build_sequences(self.songs,window_size)
        self.encoded_x,self.encoded_y= self._convert_to_one_hot_encoder(self.x,self.y)
        
        
        
        if shuffle:
            random.seed(seed)
            indices = list(range(len(x)))
            random.shuffle(indices)
            x =[x[i] for i in indices]
            y =[x[i] for i in indices]
            
        self.x = x
        self.y = y
            
        
        def _build_sequences(self, songs,window_size):
            x,y = [],[]
            
            for song in songs:
                # assert that the window_size has to larger than the song
                
                
                for i in range(len(song) -window_size):
                    seq = song[i:i+window_size]
                    target = song[i+window_size]
                    x.append(seq)
                    y.append(target)
                    
            return x,y
        
        def _convert_to_one_hot_encoder(self, x,y):
            encoded_x=[]
            encoded_y = []
            
            
            for section in x:
                for note in section:
                    I = list(np.eye(128))
                    encoded_x.append(I[note])
            
            for section_out in y:
                for note_out in section_out:
                    I = list(np.eye(128))
                    encoded_y.append(I[note_out])
                    
            return encoded_x,encoded_y
            