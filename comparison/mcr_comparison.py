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
        """set dictionaries that have condition concepts and drug concepts"""
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
        glove_5yrs_phedict_c = OrderedDict()
        glove_visit_phedict_c = OrderedDict()
        glove_5yrs_phedict_d = OrderedDict()
        glove_visit_phedict_d = OrderedDict()
        glove_5yrs_phedict_cd = OrderedDict()
        glove_visit_phedict_cd = OrderedDict()
        
        print("updating concepts for each phenotyping algorithm")
        for phe in unique_phe:
            yrs_intersection_c = glove_5yrs_all_concepts.intersection(set(self.condition_phe[phe]))
            visit_intersection_c = glove_visit_all_concepts.intersection(set(self.condition_phe[phe]))
            yrs_intersection_d = glove_5yrs_all_concepts.intersection(set(self.drug_phe[phe]))
            visit_intersection_d = glove_visit_all_concepts.intersection(set(self.drug_phe[phe]))
            
            yrs_combinations_c = list(itertools.combinations(list(yrs_intersection_c), 2))
            visit_combinations_c = list(itertools.combinations(list(visit_intersection_c), 2))
            yrs_combinations_d = list(itertools.combinations(list(yrs_intersection_d), 2))
            visit_combinations_d = list(itertools.combinations(list(visit_intersection_d), 2))
            yrs_combinations_cd = list(itertools.product(list(yrs_intersection_c), list(yrs_intersection_d)))
            visit_combinations_cd = list(itertools.product(list(visit_intersection_c), list(visit_intersection_d)))
            
            glove_5yrs_phedict_c.update({phe : yrs_combinations_c})
            glove_visit_phedict_c.update({phe : visit_combinations_c})
            glove_5yrs_phedict_d.update({phe : yrs_combinations_d})
            glove_visit_phedict_d.update({phe : visit_combinations_d})
            glove_5yrs_phedict_cd.update({phe : yrs_combinations_cd})
            glove_visit_phedict_cd.update({phe : visit_combinations_cd})
            
        self.glove_phedict.update({"glove_5yrs_c" : glove_5yrs_phedict_c})
        self.glove_phedict.update({"glove_visit_c" : glove_visit_phedict_c})
        self.glove_phedict.update({"glove_5yrs_d" : glove_5yrs_phedict_d})
        self.glove_phedict.update({"glove_visit_d" : glove_visit_phedict_d})
        self.glove_phedict.update({"glove_5yrs_cd" : glove_5yrs_phedict_cd})
        self.glove_phedict.update({"glove_visit_cd" : glove_visit_phedict_cd})
        
    def setPhePairs_n2v(self):
        self.load_n2vemb()
        n2v_h_all_concepts = set(self.n2v_hierarchical.vocab)
        n2v_f_all_concepts = set(self.n2v_full.vocab)
        unique_phe = list(self.condition_phe.keys())
        n2v_h_phedict_c = OrderedDict()
        n2v_f_phedict_c = OrderedDict()
        n2v_h_phedict_d = OrderedDict()
        n2v_f_phedict_d = OrderedDict()
        n2v_h_phedict_cd = OrderedDict()
        n2v_f_phedict_cd = OrderedDict() ## here
        
        print("updating concepts of n2v_full for each phenotyping algorithm")
        for phe in unique_phe:
            f_intersection_c = n2v_f_all_concepts.intersection(set(self.condition_phe[phe]))
            f_intersection_d = n2v_f_all_concepts.intersection(set(self.drug_phe[phe]))
            
            f_combinations_c = list(itertools.combinations(list(f_intersection_c), 2))
            f_combinations_d = list(itertools.combinations(list(f_intersection_d), 2))
            f_combinations_cd = list(itertools.product(list(f_intersection_c), list(f_intersection_d)))
            
            n2v_f_phedict_c.update({phe : f_combinations_c})
            n2v_f_phedict_d.update({phe : f_combinations_d})
            n2v_f_phedict_cd.update({phe : f_combinations_cd})

        print("updating concepts of n2v_hierarchocal for each phenotyping algorithm")
        for phe in unique_phe:
            h_intersection_c = n2v_h_all_concepts.intersection(set(self.condition_phe[phe]))
            h_intersection_d = n2v_h_all_concepts.intersection(set(self.drug_phe[phe]))

            h_combinations_c = list(itertools.combinations(list(h_intersection_c), 2))
            h_combinations_d = list(itertools.combinations(list(h_intersection_d), 2))
            h_combinations_cd = list(itertools.product(list(h_intersection_c), list(h_intersection_d)))

            n2v_h_phedict_c.update({phe : h_combinations_c})
            n2v_h_phedict_d.update({phe : h_combinations_d})
            n2v_h_phedict_cd.update({phe : h_combinations_cd})
        
        self.n2v_phedict.update({"n2v_hierarchical_c" : n2v_h_phedict_c})
        self.n2v_phedict.update({"n2v_full_c" : n2v_f_phedict_c})
        self.n2v_phedict.update({"n2v_hierarchical_d" : n2v_h_phedict_d})
        self.n2v_phedict.update({"n2v_full_d" : n2v_f_phedict_d})
        self.n2v_phedict.update({"n2v_hierarchical_cd" : n2v_h_phedict_cd})
        self.n2v_phedict.update({"n2v_full_cd" : n2v_f_phedict_cd})
        
    def computeSim_n2v(self):
        unique_phe = list(self.condition_phe.keys())
        sims_f_c = OrderedDict()
        sims_h_c = OrderedDict()
        sims_f_d = OrderedDict()
        sims_h_d = OrderedDict()
        sims_f_cd = OrderedDict()
        sims_h_cd = OrderedDict()
        
        print("computing cosine similarities of n2v_full_c")
        for phe in unique_phe:
            sims = computeSims_n2v(self.n2v_phedict["n2v_full_c"][phe], self.n2v_full)
            sims_f_c.update({phe : sims})

        print("computing cosine similarities of n2v_full_d")
        for phe in unique_phe:
            sims = computeSims_n2v(self.n2v_phedict["n2v_full_d"][phe], self.n2v_full)
            sims_f_d.update({phe : sims})

        print("computing cosine similarities of n2v_full_cd")
        for phe in unique_phe:
            sims = computeSims_n2v(self.n2v_phedict["n2v_full_cd"][phe], self.n2v_full)
            sims_f_cd.update({phe : sims})
            
        print("computing cosine similarities of n2v_hierarchical_c")
        for phe in unique_phe:
            sims = computeSims_n2v(self.n2v_phedict["n2v_hierarchical_c"][phe], self.n2v_hierarchical)
            sims_h_c.update({phe : sims})

        print("computing cosine similarities of n2v_hierarchical_d")
        for phe in unique_phe:
            sims = computeSims_n2v(self.n2v_phedict["n2v_hierarchical_d"][phe], self.n2v_hierarchical)
            sims_h_d.update({phe : sims})

        print("computing cosine similarities of n2v_hierarchical_cd")
        for phe in unique_phe:
            sims = computeSims_n2v(self.n2v_phedict["n2v_hierarchical_cd"][phe], self.n2v_hierarchical)
            sims_h_cd.update({phe : sims})
            
        self.n2v_sims["sims_full_c"] = sims_f_c
        self.n2v_sims["sims_hierarchical_c"] = sims_h_c
        self.n2v_sims["sims_full_d"] = sims_f_d
        self.n2v_sims["sims_hierarchical_d"] = sims_h_d
        self.n2v_sims["sims_full_cd"] = sims_f_cd
        self.n2v_sims["sims_hierarchical_cd"] = sims_h_cd
            
    def getRandomSim_n2v(self):
        unique_phe = list(self.condition_phe.keys())
        n2v_full_rsim_c = OrderedDict()
        n2v_hierarchical_rsim_c = OrderedDict()
        n2v_full_rsim_d = OrderedDict()
        n2v_hierarchical_rsim_d = OrderedDict()
        n2v_full_rsim_cd = OrderedDict()
        n2v_hierarchical_rsim_cd = OrderedDict()
        
        print("computing cosine similarities for random pairs of n2v_full_c")
        for phe in unique_phe:
            pair_num = len(self.n2v_sims["sims_full_c"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_n2v(pair_num, self.n2v_full)
            else:
                random_sims = []
            n2v_full_rsim_c.update({phe : random_sims})
            
        print("computing cosine similarities for random pairs of n2v_hierarchical_c")
        for phe in unique_phe:
            pair_num = len(self.n2v_sims["sims_hierarchical_c"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_n2v(pair_num, self.n2v_hierarchical)
            else:
                random_sims = []
            n2v_hierarchical_rsim_c.update({phe : random_sims})

        print("computing cosine similarities for random pairs of n2v_full_d")
        for phe in unique_phe:
            pair_num = len(self.n2v_sims["sims_full_d"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_n2v(pair_num, self.n2v_full)
            else:
                random_sims = []
            n2v_full_rsim_d.update({phe : random_sims})
            
        print("computing cosine similarities for random pairs of n2v_hierarchical_d")
        for phe in unique_phe:
            pair_num = len(self.n2v_sims["sims_hierarchical_d"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_n2v(pair_num, self.n2v_hierarchical)
            else:
                random_sims = []
            n2v_hierarchical_rsim_d.update({phe : random_sims})
        
        print("computing cosine similarities for random pairs of n2v_full_cd")
        for phe in unique_phe:
            pair_num = len(self.n2v_sims["sims_full_cd"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_n2v(pair_num, self.n2v_full)
            else:
                random_sims = []
            n2v_full_rsim_cd.update({phe : random_sims})
            
        print("computing cosine similarities for random pairs of n2v_hierarchical_cd")
        for phe in unique_phe:
            pair_num = len(self.n2v_sims["sims_hierarchical_cd"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_n2v(pair_num, self.n2v_hierarchical)
            else:
                random_sims = []
            n2v_hierarchical_rsim_cd.update({phe : random_sims})

        self.n2v_rsims["rsims_full_c"] = n2v_full_rsim_c
        self.n2v_rsims["rsims_hierarchical_c"] = n2v_hierarchical_rsim_c
        self.n2v_rsims["rsims_full_d"] = n2v_full_rsim_d
        self.n2v_rsims["rsims_hierarchical_d"] = n2v_hierarchical_rsim_d
        self.n2v_rsims["rsims_full_cd"] = n2v_full_rsim_cd
        self.n2v_rsims["rsims_hierarchical_cd"] = n2v_hierarchical_rsim_cd
        
    def computeSim_glove(self):
        unique_phe = list(self.condition_phe.keys())
        sims_5yrs_c = OrderedDict()
        sims_visit_c = OrderedDict()
        sims_5yrs_d = OrderedDict()
        sims_visit_d = OrderedDict()
        sims_5yrs_cd = OrderedDict()
        sims_visit_cd = OrderedDict()
        
        print("computing cosine similarities of glove_5yrs")
        for phe in unique_phe:
            sims_c = computeSims(self.glove_phedict["glove_5yrs_c"][phe], self.glove_emb_matrix["glove_5yrs"], 
                               self.glove_concept2id["concept2id_5yrs"])
            sims_d = computeSims(self.glove_phedict["glove_5yrs_d"][phe], self.glove_emb_matrix["glove_5yrs"], 
                               self.glove_concept2id["concept2id_5yrs"])
            sims_cd = computeSims(self.glove_phedict["glove_5yrs_cd"][phe], self.glove_emb_matrix["glove_5yrs"], 
                               self.glove_concept2id["concept2id_5yrs"])
            sims_5yrs_c.update({phe : sims_c})
            sims_5yrs_d.update({phe : sims_d})
            sims_5yrs_cd.update({phe : sims_cd})
            
        print("computing cosine similarities of glove_visit")
        for phe in unique_phe:
            sims_c = computeSims(self.glove_phedict["glove_visit_c"][phe], self.glove_emb_matrix["glove_visit"], 
                               self.glove_concept2id["concept2id_visit"])
            sims_d = computeSims(self.glove_phedict["glove_visit_d"][phe], self.glove_emb_matrix["glove_visit"], 
                               self.glove_concept2id["concept2id_visit"])
            sims_cd = computeSims(self.glove_phedict["glove_visit_cd"][phe], self.glove_emb_matrix["glove_visit"], 
                               self.glove_concept2id["concept2id_visit"])
            sims_visit_c.update({phe : sims_c})
            sims_visit_d.update({phe : sims_d})
            sims_visit_cd.update({phe : sims_cd})
            
        self.glove_sims["sims_5yrs_c"] = sims_5yrs_c
        self.glove_sims["sims_visit_c"] = sims_visit_c
        self.glove_sims["sims_5yrs_d"] = sims_5yrs_d
        self.glove_sims["sims_visit_d"] = sims_visit_d
        self.glove_sims["sims_5yrs_cd"] = sims_5yrs_cd
        self.glove_sims["sims_visit_cd"] = sims_visit_cd
        
    def getRandomSim_glove(self):
        unique_phe = list(self.condition_phe.keys())
        glove_5yrs_rsim_c = OrderedDict()
        glove_visit_rsim_c = OrderedDict()
        glove_5yrs_rsim_d = OrderedDict()
        glove_visit_rsim_d = OrderedDict()
        glove_5yrs_rsim_cd = OrderedDict()
        glove_visit_rsim_cd = OrderedDict() 
        
        print("computing cosine similarities for random pairs of glove_5yrs_c")
        for phe in unique_phe:
            pair_num = len(self.glove_sims["sims_5yrs_c"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_glove(pair_num, self.glove_emb_matrix["glove_5yrs"])
            else:
                random_sims = []
            glove_5yrs_rsim_c.update({phe : random_sims})
            
        print("computing cosine similarities for random pairs of glove_visit_c")
        for phe in unique_phe:
            pair_num = len(self.glove_sims["sims_visit_c"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_glove(pair_num, self.glove_emb_matrix["glove_visit"])
            else:
                random_sims = []
            glove_visit_rsim_c.update({phe : random_sims})

        print("computing cosine similarities for random pairs of glove_5yrs_d")
        for phe in unique_phe:
            pair_num = len(self.glove_sims["sims_5yrs_d"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_glove(pair_num, self.glove_emb_matrix["glove_5yrs"])
            else:
                random_sims = []
            glove_5yrs_rsim_d.update({phe : random_sims})
            
        print("computing cosine similarities for random pairs of glove_visit_d")
        for phe in unique_phe:
            pair_num = len(self.glove_sims["sims_visit_c"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_glove(pair_num, self.glove_emb_matrix["glove_visit"])
            else:
                random_sims = []
            glove_visit_rsim_d.update({phe : random_sims})
    
        print("computing cosine similarities for random pairs of glove_5yrs_cd")
        for phe in unique_phe:
            pair_num = len(self.glove_sims["sims_5yrs_cd"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_glove(pair_num, self.glove_emb_matrix["glove_5yrs"])
            else:
                random_sims = []
            glove_5yrs_rsim_cd.update({phe : random_sims})
            
        print("computing cosine similarities for random pairs of glove_visit_cd")
        for phe in unique_phe:
            pair_num = len(self.glove_sims["sims_visit_cd"][phe])
            if pair_num != 0:
                random_sims = generateRandomSim_glove(pair_num, self.glove_emb_matrix["glove_visit"])
            else:
                random_sims = []
            glove_visit_rsim_cd.update({phe : random_sims})
        
        self.glove_rsims["rsims_5yrs_c"] = glove_5yrs_rsim_c
        self.glove_rsims["rsims_visit_c"] = glove_visit_rsim_c    
        self.glove_rsims["rsims_5yrs_d"] = glove_5yrs_rsim_d
        self.glove_rsims["rsims_visit_d"] = glove_visit_rsim_d
        self.glove_rsims["rsims_5yrs_cd"] = glove_5yrs_rsim_cd
        self.glove_rsims["rsims_visit_cd"] = glove_visit_rsim_cd    
        
    def getAvgSim(self):
        unique_phe = list(self.condition_phe.keys())
        
        self.avg_sims["avgsim_n2v_full_c"] = computeAvgSim(self.n2v_sims["sims_full_c"])
        self.avg_sims["avgsim_n2v_hierarchical_c"] = computeAvgSim(self.n2v_sims["sims_hierarchical_c"])
        self.avg_sims["avgsim_n2v_full_d"] = computeAvgSim(self.n2v_sims["sims_full_d"])
        self.avg_sims["avgsim_n2v_hierarchical_d"] = computeAvgSim(self.n2v_sims["sims_hierarchical_d"])
        self.avg_sims["avgsim_n2v_full_cd"] = computeAvgSim(self.n2v_sims["sims_full_cd"])
        self.avg_sims["avgsim_n2v_hierarchical_cd"] = computeAvgSim(self.n2v_sims["sims_hierarchical_cd"])

        self.avg_sims["avgsim_glove_5yrs_c"] = computeAvgSim(self.glove_sims["sims_5yrs_c"])
        self.avg_sims["avgsim_glove_visit_c"] = computeAvgSim(self.glove_sims["sims_visit_c"])
        self.avg_sims["avgsim_glove_5yrs_d"] = computeAvgSim(self.glove_sims["sims_5yrs_d"])
        self.avg_sims["avgsim_glove_visit_d"] = computeAvgSim(self.glove_sims["sims_visit_d"])
        self.avg_sims["avgsim_glove_5yrs_cd"] = computeAvgSim(self.glove_sims["sims_5yrs_cd"])
        self.avg_sims["avgsim_glove_visit_cd"] = computeAvgSim(self.glove_sims["sims_visit_cd"])
        
        self.avg_rsims["avgrsim_n2v_full_c"] = computeAvgSim(self.n2v_rsims["rsims_full_c"])
        self.avg_rsims["avgrsim_n2v_hierarchical_c"] = computeAvgSim(self.n2v_rsims["rsims_hierarchical_c"])
        self.avg_rsims["avgrsim_n2v_full_d"] = computeAvgSim(self.n2v_rsims["rsims_full_d"])
        self.avg_rsims["avgrsim_n2v_hierarchical_d"] = computeAvgSim(self.n2v_rsims["rsims_hierarchical_d"])
        self.avg_rsims["avgrsim_n2v_full_cd"] = computeAvgSim(self.n2v_rsims["rsims_full_cd"])
        self.avg_rsims["avgrsim_n2v_hierarchical_cd"] = computeAvgSim(self.n2v_rsims["rsims_hierarchical_cd"])

        self.avg_rsims["avgrsim_glove_5yrs_c"] = computeAvgSim(self.glove_rsims["rsims_5yrs_c"])
        self.avg_rsims["avgrsim_glove_visit_c"] = computeAvgSim(self.glove_rsims["rsims_visit_c"])
        self.avg_rsims["avgrsim_glove_5yrs_d"] = computeAvgSim(self.glove_rsims["rsims_5yrs_d"])
        self.avg_rsims["avgrsim_glove_visit_d"] = computeAvgSim(self.glove_rsims["rsims_visit_d"])
        self.avg_rsims["avgrsim_glove_5yrs_cd"] = computeAvgSim(self.glove_rsims["rsims_5yrs_cd"])
        self.avg_rsims["avgrsim_glove_visit_cd"] = computeAvgSim(self.glove_rsims["rsims_visit_cd"])

        self.avg_total["avgsim_n2v_full_total_c"] = computeAvgSim_total(self.n2v_sims["sims_full_c"])
        self.avg_total["avgsim_n2v_hierarchical_total_c"] = computeAvgSim_total(self.n2v_sims["sims_hierarchical_c"])
        self.avg_total["avgsim_n2v_full_total_d"] = computeAvgSim_total(self.n2v_sims["sims_full_d"])
        self.avg_total["avgsim_n2v_hierarchical_total_d"] = computeAvgSim_total(self.n2v_sims["sims_hierarchical_d"])
        self.avg_total["avgsim_n2v_full_total_cd"] = computeAvgSim_total(self.n2v_sims["sims_full_cd"])
        self.avg_total["avgsim_n2v_hierarchical_total_cd"] = computeAvgSim_total(self.n2v_sims["sims_hierarchical_cd"])

        self.avg_total["avgsim_glove_5yrs_total_c"] = computeAvgSim_total(self.glove_sims["sims_5yrs_c"])
        self.avg_total["avgsim_glove_visit_total_c"] = computeAvgSim_total(self.glove_sims["sims_visit_c"])
        self.avg_total["avgsim_glove_5yrs_total_d"] = computeAvgSim_total(self.glove_sims["sims_5yrs_d"])
        self.avg_total["avgsim_glove_visit_total_d"] = computeAvgSim_total(self.glove_sims["sims_visit_d"])
        self.avg_total["avgsim_glove_5yrs_total_cd"] = computeAvgSim_total(self.glove_sims["sims_5yrs_cd"])
        self.avg_total["avgsim_glove_visit_total_cd"] = computeAvgSim_total(self.glove_sims["sims_visit_cd"])
        
        self.avg_total["avgrsim_n2v_full_total_c"] = computeAvgSim_total(self.n2v_rsims["rsims_full_c"])
        self.avg_total["avgrsim_n2v_hierarchical_total_c"] = computeAvgSim_total(self.n2v_rsims["rsims_hierarchical_c"])
        self.avg_total["avgrsim_n2v_full_total_d"] = computeAvgSim_total(self.n2v_rsims["rsims_full_d"])
        self.avg_total["avgrsim_n2v_hierarchical_total_d"] = computeAvgSim_total(self.n2v_rsims["rsims_hierarchical_d"])
        self.avg_total["avgrsim_n2v_full_total_cd"] = computeAvgSim_total(self.n2v_rsims["rsims_full_cd"])
        self.avg_total["avgrsim_n2v_hierarchical_total_cd"] = computeAvgSim_total(self.n2v_rsims["rsims_hierarchical_cd"])

        self.avg_total["avgrsim_glove_5yrs_total_c"] = computeAvgSim_total(self.glove_rsims["rsims_5yrs_c"])
        self.avg_total["avgrsim_glove_visit_total_c"] = computeAvgSim_total(self.glove_rsims["rsims_visit_c"])
        self.avg_total["avgrsim_glove_5yrs_total_d"] = computeAvgSim_total(self.glove_rsims["rsims_5yrs_d"])
        self.avg_total["avgrsim_glove_visit_total_d"] = computeAvgSim_total(self.glove_rsims["rsims_visit_d"])
        self.avg_total["avgrsim_glove_5yrs_total_cd"] = computeAvgSim_total(self.glove_rsims["rsims_5yrs_cd"])
        self.avg_total["avgrsim_glove_visit_total_cd"] = computeAvgSim_total(self.glove_rsims["rsims_visit_cd"])
        
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