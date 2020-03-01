import numpy as numpy
import random
import networkx as nx
from scipy.spatial.distance import cosine

def cosine_sim(vec_a,vec_b):
    """
    -- vec_a, vec_b : vectors to calculate cosine similarity between them
    """
    return (1 - cosine(vec_a,vec_b))

class StructuralEval():
    """Class for structural information evaluation"""
    def __init__(self, config):
        self.G = None
        self.concept2id = load_dictionary(config.dict_dir)
        self.source_list = list(self.concept2id.keys())
    
    def getDistPair(self, distance, num_pairs):
        dist_pairs = []
        
        while (len(dist_pairs) < num_pairs):
            source_node = random.choice(self.source_list)
            target_path = set(nx.single_source_shortest_path_length(self.G, source_node, cutoff=distance).items())
            connecting_path = set(nx.single_source_shortest_path_length(self.G, source_node, cutoff=distance-1).items())
            target_pairs = list(target_path.difference(connecting_path))
        
            if len(target_pairs) > 0:
                for pair in target_pairs:
                    if len(dist_pairs) < num_pairs:
                        dist_pairs.append((source_node, pair[0]))
            else:
                continue
        
            dist_pairs = list(set(dist_pairs))
        
        return dist_pairs

        def buildGraph(self, edgelist_dir):
            """build graph using edgelist"""
            edgelist = readEdgelist(edgelist_dir)
            self.G = nx.Graph()
            self.G.add_edges_from(edgelist)