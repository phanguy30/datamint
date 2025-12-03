import random
from midi_to_dataset import MidiDatasetLoader

class MusicDataset(MidiDatasetLoader):
    def __init__(self,folder_path,window_size=16,shuffle=True,seed=42,train_ratio = 0.8):
        """
        Args:
            songs (list): list of song,each song is a list of ints
            window_size (int, optional): length of input sequence. Defaults to 16.
            shuffle (bool, optional): if the data should be shuffled or not. Defaults to True.
            seed (int, optional): set seed for reproducibility. Defaults to 42.
        """
        
        
        self.window_size= window_size
        self.shuffle = shuffle
        self.seed= seed
        
        super().__init__(folder_path)
        
        x,y = self._build_sequences(self.songs,window_size)
        
        
        
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
        
        
            
            