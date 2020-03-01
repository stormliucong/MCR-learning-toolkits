import numpy as np
import random
import networkx as nx
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cosine
from collections import OrderedDict

def cosine_sim(vec_a,vec_b):
    """
    -- vec_a, vec_b : vectors to calculate cosine similarity between them
    """
    return (1 - cosine(vec_a,vec_b))

class StructuralEval():
    """Class for structural information evaluation"""
    def __init__(self, config):
        self.G = None
        self.dist_pairs  = OrderedDict()
        self.sims = OrderedDict()
        self.concept2id = load_dictionary(config.dict_dir)
        self.enhanced_emb = load_emb_matrix(config.enhanced_emb_dir)
        self.n2v_emb = load_emb_matrix(config.n2v_emb_dir)
        self.glove_emb = load_emb_matrix(config.glove_emb_dir)
        self.source_list = list(self.concept2id.keys())
        self.buildGraph(config.edgelist_dir)
    
    def getDistPair(self, distance, num_pairs):
        # too many similr pairs check it !
        # two nodes both must be included in source_list !
        dist_pairs = [] 
        
        while (len(dist_pairs) < num_pairs):
            source_node = random.choice(self.source_list)
            try:
                target_path = set(nx.single_source_shortest_path_length(self.G, source_node, cutoff=distance).items())
                connecting_path = set(nx.single_source_shortest_path_length(self.G, source_node, cutoff=distance-1).items())
            except:
                continue
            target_pairs = list(target_path.difference(connecting_path))
        
            if len(target_pairs) > 0:
                for pair in target_pairs:
                    if len(dist_pairs) < num_pairs:
                        dist_pairs.append((source_node, pair[0]))
            else:
                continue
            dist_pairs = list(set(dist_pairs))
        return dist_pairs
    
    def updateDistPairs(self, num_pairs):
        """"""
        self.dist_pairs["dist2_pairs"] = self.getDistPair(2, num_pairs)
        print("pairs for distance 2 have been updated")
        self.dist_pairs["dist3_pairs"] = self.getDistPair(3, num_pairs)
        print("pairs for distance 3 have been updated")
        self.dist_pairs["dist4_pairs"] = self.getDistPair(4, num_pairs)
        print("pairs for distance 4 have been updated")
        self.dist_pairs["dist5_pairs"] = self.getDistPair(5, num_pairs)
        print("pairs for distance 5 have been updated")
        self.dist_pairs["dist6_pairs"] = self.getDistPair(6, num_pairs)
        print("pairs for distance 6 have been updated")
    
    def updateSims(self):
        enhanced_sims = []
        n2v_sims = []
        glove_sims = []

        for dpairs in list(self.dist_pairs.keys()):
            enhanced_sims.append(computeSims(self.dist_pairs[dpairs], self.enhanced_emb, self.concept2id))
            n2v_sims.append(computeSims(self.dist_pairs[dpairs], self.n2v_emb, self.concept2id))
            glove_sims.append(computeSims(self.dist_pairs[dpairs], self.glove_emb, self.concept2id))

        self.sims["enhanced_sims"] = enhanced_sims
        self.sims["n2v_sims"] = n2v_sims
        self.sims["glove_sims"] = glove_sims

    def buildGraph(self, edgelist_dir):
        """build graph using edgelist"""
        edgelist = readEdgelist(edgelist_dir)
        self.G = nx.Graph()
        self.G.add_edges_from(edgelist)

    def plotSims(self):
        """plot similarity results"""
        pass

def readEdgelist(edgelist_dir):
    """read raw edgelist file"""
    f = open(edgelist_dir, "r")
    file = f.readlines()
    edgelist = []
    
    for line in file:
        line = line.strip("\n")
        linelist = line.split(" ")
        tform = (linelist[0], linelist[1])
        edgelist.append(tform)
    
    return edgelist

def load_dictionary(pklfile):
    f = open(pklfile, "rb")
    dict_load = pickle.load(f)
    return dict_load

def computeSims(pairs, vector_matrix, concept2id):
    sim_list = []
    print("start computing cosine similarities in the pair list")
    for i in tqdm(range(len(pairs))):
        id_pairs = (concept2id[pairs[i][0]], concept2id[pairs[i][1]])
        try:
            sim = 1 - (cosine(vector_matrix[id_pairs[0]], vector_matrix[id_pairs[1]]))
            sim_list.append(sim)
        except:
            pass    
    return sim_list

def load_emb_matrix(npydir):
    return np.load(npydir)