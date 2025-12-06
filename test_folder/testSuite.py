import unittest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path so we can import Music_Generation
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_folder.analysis_test import TestMusicAnalysis

## Implement a test suite
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMusicAnalysis))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())