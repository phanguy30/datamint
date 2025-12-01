from mido import MidiFile, MidiTrack

def get_midi_file_notes(filename):
    """Returns the sequence of notes played in the midi file
    There are 128 possible notes on a MIDI device, and they are numbered 0 to 127.
    The middle C is note number 60. Larger numbers indiciate higher pitch notes,
    and lower numbers indicate lower pitch notes.

    You can read more about the midi representation below, but it is not
    necessary for this assignment.
    http://midi.teragonaudio.com/tech/midispec/noteon.htm
    """
    notes = []
    for msg in  MidiFile(filename):
        if msg.type == 'note_on':
            notes.append(msg.note)
    return notes