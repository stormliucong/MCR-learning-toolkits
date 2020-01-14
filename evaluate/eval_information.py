import numpy as numpy
from scipy.spatial.distance import cosine

def consine_sim(a,b):
    return (1 - cosine(a,b))

def evaluate_structural_info():
    pass

def evaluate_cooccur_info():
    pass
