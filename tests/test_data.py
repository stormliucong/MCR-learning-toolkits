# Test the data in the data directoray is correct.
import unittest

class TestData(unittest.TestCase):
    def test_traininig_sample_format(self):
        pass

    def test_validation_sample_format(self):
        pass
    
    def test_training_sample_size(self):
        pass
    
    def test_validation_sample_size(self):
        pass

    def test_dictionary(self):
        #load dictionary
        #self.assertTrue(concept2idx[12345],234)
        pass

    def test_get_unique_conceptset(self):
        from utils.data_preprocessing import get_unique_conceptset
        import os
        unique_concept_set = get_unique_conceptset(os.path.join(os.path.abspath('.'),'/data/drug_condition_pair.csv'))
        print(len(unique_concept_set))
        self.assertTrue(len(unique_concept_set),)


if __name__ == "__main__":
    unittest.main()