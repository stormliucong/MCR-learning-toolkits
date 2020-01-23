import unittest

class TestUtils(unittest.TestCase):
    def test_process_config(self):
        from utils.config import process_config
        config = process_config('None')
        self.assertTrue(config.dictionary.concept2id_dir == 'data/concept2id_dictionary')
        
    def test_get_config_from_json(self):
        from utils.config import get_config_from_json
        json_file = '/home/cl3720/2019-concept2vec/deep-learning-skeleton/./experiments/configs.json'
        config, _ = get_config_from_json(json_file)
        self.assertTrue('data' in config)

    def test_create_dirs(self):
        import os
        from utils.dirs import create_dirs
        dirs = os.path.join(os.path.abspath('.'),'tests/test_dir')
        create_dirs(dirs)
        self.assertTrue(os.path.exists(dirs))
        dirs = [os.path.join(os.path.abspath('.'),'tests/test1_dir'),os.path.join(os.path.abspath('.'),'tests/test2_dir')]
        create_dirs(dirs)
        for dir_ in dirs:
            self.assertTrue(os.path.exists(dir_))
    
    
