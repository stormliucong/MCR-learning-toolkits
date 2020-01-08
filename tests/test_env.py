import os
import unittest
import sys

class TestEnv(unittest.TestCase):
    def test_ssh(self):
        pwd = os.getcwd()
        self.assertEqual(pwd, '/home/cl3720/2019-concept2vec/deep-learning-skeleton')

    def test_py(self):
        pyversion = sys.version
        self.assertTrue(pyversion.startswith('3.5'))
    
    def test_keras_version(self):
        pass

    def test_gpu_info(self):
        pass

if __name__ == "__main__":
    unittest.main()