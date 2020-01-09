import os
import unittest
import sys

class TestEnv(unittest.TestCase):
    def test_ssh(self):
        # pwd = os.getcwd()
        # self.assertEqual(pwd, '/home/cl3720/2019-concept2vec/deep-learning-skeleton')
        pass

    def test_py(self):
        self.assertTrue(sys.version.startswith('3.5'))

    def test_tf(self):
        import tensorflow as tf
        self.assertTrue(tf.__version__ == '1.12.0')
        self.assertTrue(tf.test.is_gpu_available())
    
    def test_keras_version(self):
        import keras as k
        self.assertTrue(k.__version__ == '2.2.4')


if __name__ == "__main__":
    unittest.main()