import numpy as numpy
from scipy.spatial.distance import cosine

def cosine_sim(vec_a,vec_b):
    """
    -- vec_a, vec_b : vectors to calculate cosine similarity between them
    """
    return (1 - cosine(vec_a,vec_b))

def evaluate_structural_info():
    pass

def evaluate_cooccur_info():
    pass

def evaluate_phe_sim():
    """use phe_pairs to evaluate representations"""
    pass