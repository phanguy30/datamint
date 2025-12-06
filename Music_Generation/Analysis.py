import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

#Twinkle Twinkle Little Star MIDI note values
test_data = [60, 60, 67, 67, 69, 69, 67,65, 65, 64, 64, 62, 62, 60,
    67, 67, 65, 65, 64, 64, 62,67, 67, 65, 65, 64, 64, 62,60, 60, 67, 67, 69, 69, 67,
    65, 65, 64, 64, 62, 62, 60, 67, 67, 74, 74, 76, 76, 74,72, 72, 71, 71, 69, 69, 67,
    74, 74, 72, 72, 71, 71, 69, 74, 74, 72, 72, 71, 71, 69,67, 67, 74, 74, 76, 76, 74,
    72, 72, 71, 71, 69, 69, 67,74, 74, 72, 72, 71, 71, 69,74, 74, 72, 72, 71, 71, 69,
    67, 67, 74, 74, 76, 76, 74, 72, 72, 71, 71, 69, 69, 67]

class MusicAnalysis:

    char_notes = pd.read_csv('Music_Generation/Note.csv')

    def __init__(self, data):
        """
        Initialize the MusicAnalysis with a DataFrame containing music data and ensures
        data is in list format.
        """
        self.data = data
    
    def count_notes(self):
        """
        Count the occurrences of each note in the dataset and pair them 
        with the corresponding note names in the Note file.
        """
        note_counts = pd.Series(self.data).value_counts().sort_index()
        note_counts.index.name = 'int'
        note_counts = note_counts.reset_index(name='count')
        merged = pd.merge(self.char_notes, note_counts, on='int', how='left').fillna(0)
        merged_counts = merged[['note', 'count']].query('count > 0')
        print(merged_counts)
        return merged_counts

    def riffs(self):
        """
        Identify and count repeated sequences of notes (riffs) in the dataset.
        """
        patterns = [tuple(self.data[i:i+3]) for i in range(len(self.data)-2)]
        pattern_counts = Counter(patterns)
        max_patern = max(pattern_counts, key=pattern_counts.get)
        named_pattern = [self.char_notes.loc[self.char_notes['int'] == note, 'note'].values[0] for note in max_patern]
        print(f"Most common riff: {'-'.join(named_pattern)} with count {pattern_counts[max_patern]}")
        return pattern_counts
    
    def pitch(self):
        """
        Calculate the average note value in the dataset, and print the 2 note
        characters on either side of the average value.
        """
        avg = round(np.mean(self.data), 3)
        lo = int(np.floor(avg))
        hi = int(np.ceil(avg))
        lo_note = self.char_notes.loc[self.char_notes['int'] == lo, 'note'].iat[0]
        hi_note = self.char_notes.loc[self.char_notes['int'] == hi, 'note'].iat[0]
        print(f"Average note value is {avg} which is between {lo_note} and {hi_note}")

    def plot_music(self):
        """
        Plot a bar chart of reversed note values (127 - note) 
        in the order they appear in the sequence.
        """
        data = self.data
        x = range(len(data))
        plt.figure(figsize=(12, 4))
        plt.bar(x, data)
        plt.xticks([0, len(data)-1], ["Beginning", "End"])
        plt.yticks([0, 127], ["Low Pitch", "High Pitch"])
        plt.xlabel("Note Position")
        plt.ylabel("Pitch")
        plt.title("Pitch Plot of Song")
        plt.tight_layout()
        plt.show()

    def counts_plot(self):
        """
        Plot a bar chart of note counts from the merged DataFrame.
        """
        merged_counts = self.count_notes()
        plt.figure(figsize=(12, 6))
        plt.bar(merged_counts['note'], merged_counts['count'])
        plt.xticks(rotation=90)
        plt.xlabel("Note")
        plt.ylabel("Count")
        plt.title("Note Counts")
        plt.tight_layout()
        plt.show()

music = MusicAnalysis(test_data)
music.count_notes()
music.riffs()
music.pitch()
music.plot_music()
music.counts_plot()