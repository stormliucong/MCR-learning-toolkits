# calculate cosine similarity
from scipy.spatial.distance import cosine

def consine_distance(a,b):
    return 1-consine(a,b)