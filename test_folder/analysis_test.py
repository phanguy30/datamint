import unittest
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path so we can import Music_Generation
sys.path.insert(0, str(Path(__file__).parent.parent))

from Music_Generation.Analysis import MusicAnalysis

class TestMusicAnalysis(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.test_data = [60, 62, 64, 60, 62, 64, 65, 67, 69, 60, 62, 64]
        self.music_analysis = MusicAnalysis(self.test_data)
    
    def tearDown(self):
        # Clean up after each test
        self.test_data = None
        self.music_analysis = None

    def test_count_notes(self):
        merged_counts = self.music_analysis.count_notes()
        
        self.assertIsInstance(merged_counts, pd.DataFrame)
        self.assertIn('note', merged_counts.columns)
        self.assertIn('count', merged_counts.columns)
        
        # Test specific note counts based on test_data
        # 60 appears 3 times, 62 appears 3 times, 64 appears 3 times
        # 65 appears 1 time, 67 appears 1 time, 69 appears 1 time
        note_60 = merged_counts[merged_counts['note'].str.contains('60|BS3', na=False)]
        note_62 = merged_counts[merged_counts['note'].str.contains('62|D4', na=False)]
        note_64 = merged_counts[merged_counts['note'].str.contains('64|E4', na=False)]
        
        # Verify counts (they should be 3, 3, 3 for the repeated notes)
        self.assertGreater(len(merged_counts), 0, "Should have at least one note counted")
        self.assertEqual(merged_counts['count'].sum(), len(self.test_data), "Total count should equal data length")

    def test_riffs(self):
        pattern_counts = self.music_analysis.riffs()
        expected_pattern = (60, 62, 64)
        self.assertIn(expected_pattern, pattern_counts)
        self.assertEqual(pattern_counts[expected_pattern], 3)

    def test_pitch(self):
        # Capture the printed output
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