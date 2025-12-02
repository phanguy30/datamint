from Models.MLPMusicGen import MLPMusicGen
from Trainer.musictrainer import MusicTrainer
from Music_Generation.music_dataset import MusicDataset

# example 

model = MLPMusicGen()

trainer = MusicTrainer(model)


#"C:\Users\preet\Documents\DATA533\test_folder"

test = MusicDataset('C:/Users/preet/Documents/DATA533/test_folder')

trainer.fit(test.encoded_x,test.encoded_y)

