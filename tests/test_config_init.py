import unittest
import pytest
from panobbgo.config import Config

@pytest.mark.skip(reason="Skipping all Dask/testing_mode tests - deferred to future work")
class TestConfigInit(unittest.TestCase):
    def setUp(self):
        # Reset Config singleton for each test
        Config._instance = None
    
    def test_singleton_basic(self):
        c1 = Config(testing_mode=False)
        c2 = Config()
        self.assertIs(c1, c2)
        self.assertFalse(c1.testing_mode)
    
    def test_testing_mode_init(self):
        c = Config(testing_mode=True)
        self.assertTrue(c.testing_mode)
        # Verify dashboard default
        self.assertEqual(c.dask_dashboard_address, ":0")
        
    def test_reinit_with_testing_mode(self):
        # First init as normal
        c1 = Config(testing_mode=False)
        self.assertFalse(c1.testing_mode)
        self.assertEqual(c1.dask_dashboard_address, ":8787")
        
        # Re-init with testing mode
        c2 = Config(testing_mode=True)
        self.assertIs(c1, c2)
        self.assertTrue(c2.testing_mode)
        self.assertEqual(c2.dask_dashboard_address, ":0")

    def test_parse_args_overrides(self):
        # This mocks command line args if parse_args=True, but difficult to test 
        # without mocking sys.argv. We focus on init params.
        pass

if __name__ == '__main__':
    unittest.main()