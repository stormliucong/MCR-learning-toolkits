# data loader related functions.
import os


def generate_pairs(directory,batch_size,concept_dictionary):
    '''
    Generate pair of concept for batch trainining
    -- directory: directory contain 
    -- batch size: number of pairs in each batch
    -- concept_dictionary: conceptid to emb id mapping
    -- return : a batch of traininig samples
    '''
    pass

def load_test_data(directory):
    '''
    load all concept for test
    '''  

def load_dictionary(csv):
    '''
    load dictionary from csv
    -- csv: a csv contain dictionary
    -- return: concept2id and id2concept
    '''
    pass

def load_weight_matric(npy):
    '''
    load weight matrix from npy
    -- npy: a npy file contain weight matrix
    -- return: list of wieghted matrix
    '''
    pass


