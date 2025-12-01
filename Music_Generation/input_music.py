from mido import MidiFile, MidiTrack

def get_midi_file_notes(filename):
    """Returns notes based on the midi file
    """
    notes = []
    for msg in  MidiFile(filename):
        if msg.type == 'note_on':
            notes.append(msg.note)
    return notes