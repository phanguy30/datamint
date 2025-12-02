from Models.MLPMusicGen import MLPMusicGen
from Trainer.musictrainer import MusicTrainer
from Music_Generation.music_dataset import MusicDataset

# example 

model = MLPMusicGen()

trainer = MusicTrainer(model)


#"C:\Users\preet\Documents\DATA533\test_folder"

test = MusicDataset('C:/Users/preet/Documents/DATA533/datamint/test_folder')
print(len(test.out_x[0][0]))
print(len(test.out_y))

# flattened = [[x for vec in sublist for x in vec] for sublist in test.out_x]


# flattened_y = [[x for vec in sublist for x in vec] for sublist in test.out_y]
# # print(len(flattened[0]))

# print(len(test.out[1]))



