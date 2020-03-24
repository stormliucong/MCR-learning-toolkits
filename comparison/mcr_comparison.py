import pandas as pd
import os
from collections import OrderedDict
from scipy.spatial.distance import cosine
import pickle
import itertools
import json
from tqdm import tqdm
import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
import gensim
import random

class MCREvaluation():
    """Class for phenotyping task evaluation"""
    def __init__(self, json_dir):
        self.config = setConfig(json_dir)
        self.condition_phe = OrderedDict()  
        self.drug_phe = OrderedDict() 
        self.n2v_hierarchical = None # loaded model
        self.n2v_full = None # loaded model

        self.n2v_phedict = OrderedDict()
        self.glove_phedict = OrderedDict()
        self.glove_concept2id = OrderedDict()
        self.glove_emb_matrix = OrderedDict()
        
        self.glove_sims = OrderedDict() # dict for saving computed sims
        self.n2v_sims = OrderedDict() # dict for saving computed sims
        self.glove_rsims = OrderedDict() # dict for saving computed random sims
        self.n2v_rsims = OrderedDict() # dict for saving computed random sims
        
        self.avg_sims = OrderedDict()
        self.avg_rsims = OrderedDict()
        self.avg_total = OrderedDict()
        
        self.avgsim_df = OrderedDict()
        self.avgsim_total = None


    def setPheDict(self):
        phe_data = pd.DataFrame.from_csv(self.config.data.phe_data, sep="\t")
        unique_phenotype = list(set(phe_data["phenotype"]))
        condition_data = phe_data.loc[phe_data["standard_domain"] == ("Condition")]
        drug_data = phe_data.loc[phe_data["standard_domain"] == ("Drug")]

        for phenotype in unique_phenotype:
            condition_phe_list = dblToStr(list(condition_data[condition_data["phenotype"] == phenotype]["standard_concept_id"]))
            drug_phe_list = dblToStr(list(drug_data[drug_data["phenotype"] == phenotype]["standard_concept_id"]))
            condition_phe_list = list(set(condition_phe_list))
            drug_phe_list = list(set(drug_phe_list))
            self.condition_phe.update({phenotype : condition_phe_list})
            self.drug_phe.update({phenotype : drug_phe_list})
        
    def load_n2vemb(self):
        print("load n2v results")
        self.n2v_hierarchical = gensim.models.KeyedVectors.load_word2vec_format(self.config.results.n2v_hierarchical)
        self.n2v_full = gensim.models.KeyedVectors.load_word2vec_format(self.config.results.n2v_hierarchical)
        print("n2v results have been loaded")
    
    def load_glove_concept2id(self):
        concept2id_5yrs = load_dictionary(self.config.data.concept2id_5yrs)
        concept2id_visit = load_dictionary(self.config.data.concept2id_visit)
        self.glove_concept2id["concept2id_5yrs"] = concept2id_5yrs
        self.glove_concept2id["concept2id_visit"] = concept2id_visit
        print("concept2id for glove have been loaded")
        
    def load_glove_emb_matrix(self):
        self.glove_emb_matrix["glove_5yrs"] = load_emb_matrix(self.config.results.glove_5yrs)
        self.glove_emb_matrix["glove_visit"] = load_emb_matrix(self.config.results.glove_visit)
        
    def setPhePairs_glove(self):
        self.load_glove_concept2id()
        glove_5yrs_all_concepts = set(self.glove_concept2id["concept2id_5yrs"].keys())
        glove_visit_all_concepts = set(self.glove_concept2id["concept2id_visit"].keys())
        unique_phe = list(self.condition_phe.keys())
        glove_5yrs_phedict = {}
        glove_visit_phedict = {}
        
        print("updating concepts for each phenotyping algorithm")
        for phe in unique_phe:
            yrs_intersection = glove_5yrs_all_concepts.intersection(set(self.condition_phe[phe]))
            visit_intersection = glove_visit_all_concepts.intersection(set(self.condition_phe[phe]))
            
            yrs_combinations = list(itertools.combinations(list(yrs_intersection), 2))
            visit_combinations = list(itertools.combinations(list(visit_intersection), 2))
            
            glove_5yrs_phedict.update({phe : yrs_combinations})
            glove_visit_phedict.update({phe : visit_combinations})
            
        self.glove_phedict.update({"glove_5yrs" : glove_5yrs_phedict})
        self.glove_phedict.update({"glove_visit" : glove_visit_phedict})
        
    def setPhePairs_n2v(self):
        self.load_n2vemb()
        n2v_h_all_concepts = set(self.n2v_hierarchical.vocab)
        n2v_f_all_concepts = set(self.n2v_full.vocab)
        unique_phe = list(self.condition_phe.keys())
        n2v_h_phedict = {}
        n2v_f_phedict = {}
        
        print("updating concepts for each phenotyping algorithm")
        for phe in unique_phe:
            f_intersection = n2v_f_all_concepts.intersection(set(self.condition_phe[phe]))
            h_intersection = n2v_h_all_concepts.intersection(set(self.condition_phe[phe]))
            
            f_combinations = list(itertools.combinations(list(f_intersection), 2))
            h_combinations = list(itertools.combinations(list(h_intersection), 2))
            
            n2v_f_phedict.update({phe : f_combinations})
            n2v_h_phedict.update({phe : h_combinations})
        
        self.n2v_phedict.update({"n2v_hierarchical" : n2v_h_phedict})
        self.n2v_phedict.update({"n2v_full" : n2v_f_phedict})
        
    def computeSim_n2v(self):
        unique_phe = list(self.condition_phe.keys())
        sims_f = OrderedDict()
        sims_h = OrderedDict()
        
        print("computing cosine similarities of n2v_full")
        for phe in unique_phe:
            sims = computeSims_n2v(self.n2v_phedict["n2v_full"][phe], self.n2v_full)
            sims_f.update({phe : sims})
            
        print("computing cosine similarities of n2v_hierarchical")
        for phe in unique_phe:
            sims = computeSims_n2v(self.n2v_phedict["n2v_hierarchical"][phe], self.n2v_hierarchical)
            sims_h.update({phe : sims})
            
        self.n2v_sims["sims_full"] = sims_f
        self.n2v_sims["sims_hierarchical"] = sims_h
            
    def getRandomSim_n2v(self):
        unique_phe = list(self.condition_phe.keys())
        n2v_full_rsim = OrderedDict()
        n2v_hierarchical_rsim = OrderedDict()
        
        print("computing cosine similarities for random pairs of n2v_full")
        for phe in unique_phe:
            pair_num = len(self.n2v_sims["sims_full"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_n2v(pair_num, self.n2v_full)
            else:
                random_sims = []
            n2v_full_rsim.update({phe : random_sims})
            
        print("computing cosine similarities for random pairs of n2v_hierarchical")
        for phe in unique_phe:
            pair_num = len(self.n2v_sims["sims_hierarchical"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_n2v(pair_num, self.n2v_hierarchical)
            else:
                random_sims = []
            n2v_hierarchical_rsim.update({phe : random_sims})
        
        self.n2v_rsims["rsims_full"] = n2v_full_rsim
        self.n2v_rsims["rsims_hierarchical"] = n2v_hierarchical_rsim
        
    def computeSim_glove(self):
        unique_phe = list(self.condition_phe.keys())
        sims_5yrs = OrderedDict()
        sims_visit = OrderedDict()
        
        print("computing cosine similarities of glove_5yrs")
        for phe in unique_phe:
            sims = computeSims(self.glove_phedict["glove_5yrs"][phe], self.glove_emb_matrix["glove_5yrs"], 
                               self.glove_concept2id["concept2id_5yrs"])
            sims_5yrs.update({phe : sims})
            
        print("computing cosine similarities of glove_visit")
        for phe in unique_phe:
            sims = computeSims(self.glove_phedict["glove_visit"][phe], self.glove_emb_matrix["glove_visit"], 
                               self.glove_concept2id["concept2id_visit"])
            sims_visit.update({phe : sims})
            
        self.glove_sims["sims_5yrs"] = sims_5yrs
        self.glove_sims["sims_visit"] = sims_visit
        
    def getRandomSim_glove(self):
        unique_phe = list(self.condition_phe.keys())
        glove_5yrs_rsim = OrderedDict()
        glove_visit_rsim = OrderedDict()
        
        print("computing cosine similarities for random pairs of glove_5yrs")
        for phe in unique_phe:
            pair_num = len(self.glove_sims["sims_5yrs"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_glove(pair_num, self.glove_emb_matrix["glove_5yrs"])
            else:
                random_sims = []
            glove_5yrs_rsim.update({phe : random_sims})
            
        print("computing cosine similarities for random pairs of glove_visit")
        for phe in unique_phe:
            pair_num = len(self.glove_sims["sims_visit"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_glove(pair_num, self.glove_emb_matrix["glove_visit"])
            else:
                random_sims = []
            glove_visit_rsim.update({phe : random_sims})
        
        self.glove_rsims["rsims_5yrs"] = glove_5yrs_rsim
        self.glove_rsims["rsims_visit"] = glove_visit_rsim        
        
    def getAvgSim(self):
        unique_phe = list(self.condition_phe.keys())
        
        self.avg_sims["avgsim_n2v_full"] = computeAvgSim(self.n2v_sims["sims_full"])
        self.avg_sims["avgsim_n2v_hierarchical"] = computeAvgSim(self.n2v_sims["sims_hierarchical"])
        self.avg_sims["avgsim_glove_5yrs"] = computeAvgSim(self.glove_sims["sims_5yrs"])
        self.avg_sims["avgsim_glove_visit"] = computeAvgSim(self.glove_sims["sims_visit"])
        
        self.avg_rsims["avgrsim_n2v_full"] = computeAvgSim(self.n2v_rsims["rsims_full"])
        self.avg_rsims["avgrsim_n2v_hierarchical"] = computeAvgSim(self.n2v_rsims["rsims_hierarchical"])
        self.avg_rsims["avgrsim_glove_5yrs"] = computeAvgSim(self.glove_rsims["rsims_5yrs"])
        self.avg_rsims["avgrsim_glove_visit"] = computeAvgSim(self.glove_rsims["rsims_visit"])

        self.avg_total["avgsim_n2v_full_total"] = computeAvgSim_total(self.n2v_sims["sims_full"])
        self.avg_total["avgsim_n2v_hierarchical_total"] = computeAvgSim_total(self.n2v_sims["sims_hierarchical"])
        self.avg_total["avgsim_glove_5yrs_total"] = computeAvgSim_total(self.glove_sims["sims_5yrs"])
        self.avg_total["avgsim_glove_visit_total"] = computeAvgSim_total(self.glove_sims["sims_visit"])
        
        self.avg_total["avgrsim_n2v_full_total"] = computeAvgSim_total(self.n2v_rsims["rsims_full"])
        self.avg_total["avgrsim_n2v_hierarchical_total"] = computeAvgSim_total(self.n2v_rsims["rsims_hierarchical"])
        self.avg_total["avgrsim_glove_5yrs_total"] = computeAvgSim_total(self.glove_rsims["rsims_5yrs"])
        self.avg_total["avgrsim_glove_visit_total"] = computeAvgSim_total(self.glove_rsims["rsims_visit"])
        
        self.avg_sims["phenotyping_algorithm"] = unique_phe
        self.avg_rsims["phenotyping_algorithm"] = unique_phe
        
        avgsim_df = pd.DataFrame(self.avg_sims)
        avgsim_df = pd.DataFrame.transpose(avgsim_df.set_index("phenotyping_algorithm"))
        avgrsim_df = pd.DataFrame(self.avg_rsims)
        avgrsim_df = pd.DataFrame.transpose(avgrsim_df.set_index("phenotyping_algorithm"))
        
        self.avgsim_df["sims"] = avgsim_df
        self.avgsim_df["rsims"] = avgrsim_df
        self.avgsim_total = pd.DataFrame(self.avg_total)
    
    def saveResults(self):
        self.avgsim_df["sims"].to_csv(os.path.join(self.config.save_dir, "avgsim.csv"))
        self.avgsim_df["rsims"].to_csv(os.path.join(self.config.save_dir, "avgrsim.csv"))
        self.avgsim_total.to_csv(os.path.join(self.config.save_dir, "avgsim_total.csv"))


def dblToStr(dbl_list):
    int_list = list(map(int, dbl_list))
    str_list = list(map(str, int_list))
    return str_list

def setConfig(json_file):
    """
    Get the config from a json file
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        json_body = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = DotMap(json_body)
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

def load_dictionary(pklfile):
    f = open(pklfile, "rb")
    dict_load = pickle.load(f)
    return dict_load

def load_emb_matrix(npydir):
    return np.load(npydir)

def computeSims_n2v(pair_list, n2v_model):
    sim_list = []
    
    if len(pair_list) > 0:
        for i in tqdm(range(len(pair_list))):
            try:
                sim = 1 - (cosine(n2v_model[pair_list[i][0]], n2v_model[pair_list[i][1]]))
                sim_list.append(sim)
            except:
                continue
    
    return sim_list

def computeSims(pair_list, vector_matrix, concept2id):
    sim_list = []

    if len(pair_list) > 0:
        for i in tqdm(range(len(pair_list))):
            try:
                id_pairs = (concept2id[pair_list[i][0]], concept2id[pair_list[i][1]])
                sim = 1 - (cosine(vector_matrix[id_pairs[0]], vector_matrix[id_pairs[1]]))
                sim_list.append(sim)
            except:
                continue
    return sim_list

def generateRandomSim_glove(nums, vector_matrix):
    random_sims = []
    for i in range(nums):
        random_pair = random.sample(range(1,vector_matrix.shape[0]), 2)
        sim = 1 - (cosine(vector_matrix[random_pair[0]], vector_matrix[random_pair[1]]))
        random_sims.append(sim)
    
    return random_sims

def generateRandomSim_n2v(nums, n2v_model):
    concepts = list(n2v_model.vocab)
    random_sims = []
    for i in range(nums):
        random_pair = random.sample(concepts, 2)
        sim = 1 - (cosine(n2v_model[random_pair[0]], n2v_model[random_pair[1]]))
        random_sims.append(sim)
    
    return random_sims

def computeAvgSim_total(sim_dict):
    phe_list = list(sim_dict.keys())
    total_sims = []
    for phe in phe_list:
        total_sims.extend(sim_dict[phe])
    return np.average(total_sims)
    
        
def computeAvgSim(sim_dict):
    phe_list = list(sim_dict.keys())
    sim_list = []
    
    for phe in phe_list:
        if len(sim_dict[phe]) > 0:
            avg_sim = np.average(sim_dict[phe])
        else:
            avg_sim = 0
        
        sim_list.append(avg_sim)
    
    return sim_list