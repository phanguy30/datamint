import numpy as np
from mido import MidiFile, MidiTrack, Message
import pygame
import pygame


class PlaySong:
    def __init__(self,notes,name_of_file):
        
        
        self.notes = notes
        self.name_of_file =name_of_file
        self.generate_midi(self.notes,self.name_of_file)
        self.play_midi(self.name_of_file)
        
        
        
         
    def generate_midi(self,notes, name_of_file):


        new_mid = MidiFile()
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)

        for note in notes:
            new_track.append(Message('note_on', note=note, velocity=64, time=128))
        new_mid.save(name_of_file)



    def play_midi(self,name_of_file):
        pygame.mixer.init()
        pygame.mixer.music.load(name_of_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
         
         
          
    
    
    
    
    