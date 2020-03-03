import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.spatial.distance import cosine
import pickle
import itertools
import json
from tqdm import tqdm
from dotmap import DotMap
import matplotlib.pyplot as plt

class PhenotypingEval():
    """Class for phenotyping task evaluation"""
    def __init__(self, json_params):
        self.config = setConfig(json_params)
        self.condition_pairs = None
        self.drug_pairs = None
        self.cross_pairs = None
        self.total_pairs = None
        self.enhanced_sims = dict()
        self.n2v_sims = dict()
        self.glove_sims = dict()
        self.condition_phe = OrderedDict()  
        self.drug_phe = OrderedDict() 
        self.total_phe = OrderedDict() 
        self.enhanced_emb = load_emb_matrix(self.config.results.enhanced_emb)
        self.n2v_emb = load_emb_matrix(self.config.results.n2v_emb)
        self.glove_emb = load_emb_matrix(self.config.results.glove_emb)
        self.concept2id = load_dictionary(self.config.data.concept2id)

    def setPheDict(self):
        phe_data = pd.DataFrame.from_csv(self.config.phe_data_dir, sep="\t")
        unique_phenotype = list(set(phe_data["phenotype"]))
        condition_data = phe_data.loc[phe_data["standard_domain"] == ("Condition")]
        drug_data = phe_data.loc[phe_data["standard_domain"] == ("Drug")]

        for phenotype in unique_phenotype:
            condition_phe_concepts = list(condition_data[condition_data["phenotype"] == phenotype]["standard_concept_id"])
            drug_phe_concepts = list(drug_data[drug_data["phenotype"] == phenotype]["standard_concept_id"])
            condition_phe_list = getIntersections(
                list(map(str, map(int, condition_phe_concepts))), set(self.concept2id.keys()))
            drug_phe_list = getIntersections(
                list(map(str, map(int, drug_phe_concepts))), set(self.concept2id.keys()))
            self.condition_phe.update({phenotype : condition_phe_list})
            self.drug_phe.update({phenotype : drug_phe_list})
            self.total_phe.update({phenotype : condition_phe_list + drug_phe_list})
    
    def setPhePairs(self):
        """get all the possible combinations between concepts in each phenotyping algorithm"""
        self.condition_pairs = getPairsfromDict(self.condition_phe, self.condition_phe)
        self.drug_pairs = getPairsfromDict(self.drug_phe, self.drug_phe)
        self.cross_pairs = getPairsfromDict(self.condition_phe, self.drug_phe)

    def computeRandomSims(self):
        # use random pairs instead of negative pairs
        # for each bin in histogram
        pass

    def updateSims(self, dict_to_be_updated):
        phenotypes = list(self.condition_pairs.keys())

        condition_sims = dict()
        drug_sims = dict()
        cross_sims = dict()
        total_sims = dict()
        for phe in phenotypes:
            condition_sims.update({phe : computeSims(self.condition_pairs[phe], self.enhanced_emb, self.concept2id)})
            drug_sims.update({phe : computeSims(self.drug_pairs[phe], self.enhanced_emb, self.concept2id)})
            cross_sims.update({phe : computeSims(self.cross_pairs[phe], self.enhanced_emb, self.concept2id)})

        total_sims.update(condition_sims)
        total_sims.update(drug_sims)
        total_sims.update(cross_sims)

        dict_to_be_updated.update({"condition_sims" : condition_sims, "drug_sims" : drug_sims,
        "cross_sims" : cross_sims,"total_sims" : total_sims})

    def updateSimsforEmb(self):
        self.updateSims(self.enhanced_sims)
        self.updateSims(self.n2v_sims)
        self.updateSims(self.glove_sims)

    def plotSimHist(self):
        """plot results to multiple histograms"""
        for i in range(1, 17):
            plt.subplot(4, 4, i)
        pass

    def visualizeTSNE(self):
        """visualize results using t-SNE"""
        pass
        
# package-wide functions

def computeSims(pairs, vector_matrix, concept2id):
    sim_list = []
    print("start computing cosine similarities in the pair list")
    if len(pairs) > 0:
        for i in tqdm(range(len(pairs))):
            id_pairs = (concept2id[pairs[i][0]], concept2id[pairs[i][1]])
            try:
                sim = 1 - (cosine(vector_matrix[id_pairs[0]], vector_matrix[id_pairs[1]]))
                sim_list.append(sim)
            except:
                pass    
    return sim_list

def load_dictionary(pklfile):
    f = open(pklfile, "rb")
    dict_load = pickle.load(f)
    return dict_load

def load_emb_matrix(npydir):
    return np.load(npydir)

def getIntersections(concept_list, concept_range_set):
    concept_set = set(concept_list)
    intersections = list(concept_set.intersection(concept_range_set))
    return intersections  

def setConfig(json_file):
    """
    Get the config from a json file
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)
    return config      

def getPairsfromDict(phedict1, phedict2):
    phenotypes = list(phedict1.keys())
    pairs_dict = OrderedDict()
    
    if phedict1 == phedict2:
        for phe in phenotypes:
            concepts = phedict1[phe]
            if len(concepts) > 1:
                combinations = list(itertools.combinations(concepts, 2))
            else:
                combinations = []
            pairs_dict[phe] = combinations
        
    else:
        for phe in phenotypes:
            concepts1 = phedict1[phe]
            concepts2 = phedict2[phe]
            if len(concepts1) > 1 and len(concepts2) > 1:
                combinations = list(itertools.product(concepts1, concepts2))
            else:
                combinations = []
            pairs_dict[phe] = combinations
            
    return pairs_dict