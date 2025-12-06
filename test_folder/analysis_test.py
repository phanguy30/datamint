import unittest
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from Music_Generation.Analysis import MusicAnalysis

class TestMusicAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_data = [60, 62, 64, 60, 62, 64, 65, 67, 69, 60, 62, 64]
        self.music_analysis = MusicAnalysis(self.test_data)
    
    def tearDown(self):
        self.test_data = None
        self.music_analysis = None

    def test_count_notes(self): 
        merged_counts = self.music_analysis.count_notes()
        self.assertIsInstance(merged_counts, pd.DataFrame)
        self.assertIn('note', merged_counts.columns)
        self.assertIn('count', merged_counts.columns)
        self.assertEqual(merged_counts['count'].sum(), len(self.test_data))

        note_map = {'60': 3,'62': 3,'64': 3,'65': 1,'67': 1,'69': 1}

        for note_int, expected_count in note_map.items():
            name = self.music_analysis.char_notes.loc[self.music_analysis.char_notes['int'] == int(note_int), 'note'].iat[0]
            actual = merged_counts.loc[merged_counts['note'] == name, 'count'].iat[0]
            self.assertEqual(actual, expected_count)

    def test_riffs(self):
        pattern_counts = self.music_analysis.riffs()
        expected_pattern = (60, 62, 64)
        self.assertIn(expected_pattern, pattern_counts)
        self.assertEqual(pattern_counts[expected_pattern], 3)

    def test_pitch(self):
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        self.music_analysis.pitch()
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("Average note value is", output)
        self.assertIn("which is between", output)
        self.assertIn("E4", output)  # 64 corresponds to E4

    def test_plot_music_runs(self):
        try:
            self.music_analysis.plot_music()
        except Exception as e:
            self.fail(f"plot_music() raised an exception: {e}")
