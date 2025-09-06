import unittest
import yaml
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestConfigLoading(unittest.TestCase):
    
    def setUp(self):
        """Set up the test configuration file path."""
        self.config_path = Path(__file__).parent.parent / "config.yaml"
    
    def test_config_file_exists(self):
        """Test that the config.yaml file exists."""
        self.assertTrue(self.config_path.exists(), "config.yaml file should exist")
    
    def test_config_loads_successfully(self):
        """Test that the YAML configuration loads without errors."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.assertIsInstance(config, dict, "Config should be a dictionary")
        except Exception as e:
            self.fail(f"Config loading failed with error: {e}")
    
    def test_config_contains_required_sections(self):
        """Test that all required configuration sections are present."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = [
            'debug', 'models', 'resize', 'masking', 'plot_kpts',
            'matcher_kwargs', 'ransac_kwargs', 'seq', 'covisibility_threshold',
            'min_shared_points', 'min_track_len', 'max_reproj_error',
            'min_parallax', 'min_pairs_for_submap', 'min_matches_for_pose',
            'min_inliers_for_pose', 'thresholds_r', 'thresholds_t',
            'pair_images_strategy', 'random_subset', 'random_subset_size',
            'min_distance_bw_frames'
        ]
        
        for section in required_sections:
            self.assertIn(section, config, f"Config should contain '{section}' section")
    
    def test_config_data_types(self):
        """Test that configuration values have the expected data types."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test specific data types
        self.assertIsInstance(config['debug']['enabled'], bool)
        self.assertIsInstance(config['debug']['activated_flags'], list)
        self.assertIsInstance(config['models'], list)
        self.assertIsInstance(config['masking'], bool)
        self.assertIsInstance(config['plot_kpts'], bool)
        self.assertIsInstance(config['matcher_kwargs'], dict)
        self.assertIsInstance(config['ransac_kwargs'], dict)
        self.assertIsInstance(config['seq'], str)
        self.assertIsInstance(config['thresholds_r'], list)
        self.assertIsInstance(config['thresholds_t'], list)
        
        # Test that thresholds contain numbers
        for threshold in config['thresholds_r']:
            self.assertIsInstance(threshold, (int, float))
        for threshold in config['thresholds_t']:
            self.assertIsInstance(threshold, (int, float))

if __name__ == '__main__':
    unittest.main()