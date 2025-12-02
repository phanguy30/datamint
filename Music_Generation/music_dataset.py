import random
from .midi_to_dataset import MidiDatasetLoader
import numpy as np

class MusicDataset(MidiDatasetLoader):
    def __init__(self,folder_path,window_size=10,shuffle=True,seed=42):
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
        
        self.encoded_songs= self._convert_to_one_hot_encoder(self.songs)
        
        self.out_x,self.out_y = self._build_sequences(self.encoded_songs,window_size)
        
        
        
        
        # if shuffle:
        #     random.seed(seed)
        #     indices = list(range(len(self.x)))
        #     random.shuffle(indices)
        #     x =[self.x[i] for i in indices]
        #     y =[self.x[i] for i in indices]
            
        # self.x = x
        # self.y = y
    
    
    def _convert_to_one_hot_encoder(self, songs):
        encoded_songs = []
        I = np.eye(128)
        for song in songs:
            for note in song:
                encoded_songs.append(list(I[note]))
            
                
                
        return encoded_songs
    
    def _build_sequences(self, notes, window_size):

        out_x = []
        out_y = []

        for i in range(len(notes) - window_size):
            seq = notes[i:i + window_size]
            target = notes[i + window_size]
            
            # flattens it
            rez_x = [v for vec in seq for v in vec]

            # converts to integer
            target_idx = int(np.argmax(target))

            out_x.append(rez_x)
            out_y.append(target_idx)

        return out_x, out_y

    