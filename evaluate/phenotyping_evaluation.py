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
    def __init__(self, json_dir):
        self.config = setConfig(json_dir)
        self.condition_phe = OrderedDict()  
        self.drug_phe = OrderedDict() 
        self.total_phe = OrderedDict()
        self.condition_pairs = OrderedDict()
        self.drug_pairs = OrderedDict()
        self.cross_pairs = OrderedDict()
        self.total_pairs = OrderedDict()
        self.enhanced_sims = dict()
        self.n2v_sims = dict()
        self.glove_sims = dict()
        self.enhanced_baselines = dict()
        self.n2v_baselines = dict()
        self.glove_baselines = dict()
        self.enhanced_emb = load_emb_matrix(self.config.results.enhanced_emb)
        self.n2v_emb = load_emb_matrix(self.config.data.n2v_emb)
        self.glove_emb = load_emb_matrix(self.config.data.glove_emb)
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
        
        phenotypes = list(self.condition_pairs.keys())
        for phe in phenotypes:
            total_pairs = self.condition_pairs[phe] + self.drug_pairs[phe] + self.cross_pairs[phe]
            self.total_pairs.update({phe : total_pairs})

    def updateSims(self, updating_emb, dict_to_be_updated):
        phenotypes = list(self.condition_pairs.keys())

        condition_sims = OrderedDict()  
        drug_sims = OrderedDict()
        cross_sims = OrderedDict()
        total_sims = OrderedDict()
        for phe in phenotypes:
            condition_sims.update({phe : computeSims(self.condition_pairs[phe], updating_emb, self.concept2id)})
            drug_sims.update({phe : computeSims(self.drug_pairs[phe], updating_emb, self.concept2id)})
            cross_sims.update({phe : computeSims(self.cross_pairs[phe], updating_emb, self.concept2id)})

        total_sims.update(condition_sims)
        total_sims.update(drug_sims)
        total_sims.update(cross_sims)

        dict_to_be_updated.update({"condition_sims" : condition_sims, "drug_sims" : drug_sims,
        "cross_sims" : cross_sims,"total_sims" : total_sims})

    def updateSimsforEmb(self):
        self.updateSims(self.enhanced_emb, self.enhanced_sims)
        self.updateSims(self.n2v_emb, self.n2v_sims)
        self.updateSims(self.glove_emb, self.glove_sims)

    def genRandomBaselines(self, emb_matrix, updating_dict, num_sampling=100):
        num_condition_samples = int(np.ceil(computeMedianPairs(self.condition_pairs)))
        num_drug_samples = int(np.ceil(computeMedianPairs(self.drug_pairs)))
        num_cross_samples = int(np.ceil(computeMedianPairs(self.cross_pairs)))
        num_total_samples = int(np.ceil(computeMedianPairs(self.total_pairs)))
    
        condition_sample_medians = []
        drug_sample_medians = []
        cross_sample_medians = []
        total_sample_medians = []
    
        for i in tqdm(range(num_sampling)):
            condition_sample_medians.append(getRandomMedian(num_condition_samples, emb_matrix))
            drug_sample_medians.append(getRandomMedian(num_drug_samples, emb_matrix))
            cross_sample_medians.append(getRandomMedian(num_cross_samples, emb_matrix))
            total_sample_medians.append(getRandomMedian(num_total_samples, emb_matrix))
            print(i+1, "th median baseline computed")
        
        updating_dict.update({"condition_baselines" : condition_sample_medians})
        updating_dict.update({"drug_baselines" : drug_sample_medians})
        updating_dict.update({"cross_baselines" : cross_sample_medians})
        updating_dict.update({"total_baselines" : total_sample_medians})

    def plotSimHist(self, emb_type, fig_size=(16,3)):
        """plot results to multiple histograms"""
        # add random pairs sim 
        if emb_type == "enhanced":
            data_dict = self.enhanced_sims
            baselines = self.enhanced_baselines
        elif emb_type == "n2v":
            data_dict = self.n2v_sims
            baselines = self.n2v_baselines
        elif emb_type == "glove":
            data_dict = self.glove_sims
            baselines = self.glove_baselines
        else:
            print("No data")

        labels = list(data_dict["condition_sims"].keys())
    
        f = plt.figure(figsize=fig_size)
        ax = f.add_subplot(141)
        plt.title("Median sim of condition pairs")
        ax.bar(labels, list(data_dict["condition_sims"].values()))
        plt.xticks(rotation=90, fontsize=8)
        ax.axhline(np.median(baselines["condition_baselines"]), linewidth=0.5, color = "r", ls="-") # median of random baseline
        ax.axhline(np.percentile(baselines["condition_baselines"], 5), linewidth=0.5, color = "r", ls="--") # 0.95 of random baseline
        ax.axhline(np.percentile(baselines["condition_baselines"], 95), linewidth=0.5, color = "r", ls="--") # 0.05 of random baseline
        ax2 = f.add_subplot(142)
        plt.title("Median sim of drug pairs")
        ax2.bar(labels, list(data_dict["drug_sims"].values()))
        plt.xticks(rotation=90, fontsize=8)
        ax2.axhline(np.median(baselines["drug_baselines"]), linewidth=0.5, color = "r", ls="-") # median of random baseline
        ax2.axhline(np.percentile(baselines["drug_baselines"], 5), linewidth=0.5, color = "r", ls="--") # 0.95 of random baseline
        ax2.axhline(np.percentile(baselines["drug_baselines"], 95), linewidth=0.5, color = "r", ls="--") # 0.05 of random baseline
        ax3 = f.add_subplot(143)
        plt.title("Median sim of cross pairs")
        ax3.bar(labels, list(data_dict["cross_sims"].values()))
        plt.xticks(rotation=90, fontsize=8)
        ax3.axhline(np.median(baselines["cross_baselines"]), linewidth=0.5, color = "r", ls="-") # median of random baseline
        ax3.axhline(np.percentile(baselines["cross_baselines"], 5), linewidth=0.5, color = "r", ls="--") # 0.95 of random baseline
        ax3.axhline(np.percentile(baselines["cross_baselines"], 95), linewidth=0.5, color = "r", ls="--") # 0.05 of random baseline
        ax4 = f.add_subplot(144)
        plt.title("Median sim of total pairs")
        ax4.bar(labels, list(data_dict["total_sims"].values()))
        plt.xticks(rotation=90, fontsize=8)
        ax4.axhline(np.median(baselines["total_baselines"]), linewidth=0.5, color = "r", ls="-") # median of random baseline
        ax4.axhline(np.percentile(baselines["total_baselines"], 5), linewidth=0.5, color = "r", ls="--") # 0.95 of random baseline
        ax4.axhline(np.percentile(baselines["total_baselines"], 95), linewidth=0.5, color = "r", ls="--") # 0.05 of random baseline
        plt.show()

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
    return np.nanmedian(sim_list)

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

def getRandomMedian(num_samples, emb_matrix):
    total_len = emb_matrix.shape[0]
    sim_samples = []
    
    for i in range(num_samples):
        random_pair = np.random.choice(total_len, 2, replace=False)
        sim = 1 - cosine(emb_matrix[random_pair[0]], emb_matrix[random_pair[1]])
        sim_samples.append(sim)
        
    return np.nanmedian(sim_samples)

def computeMedianPairs(pairs_dict):
    phe_list = list(pairs_dict.keys())
    
    nums = []
    for phe in phe_list:
        nums.append(len(pairs_dict[phe]))
    
    return np.median(nums)

def computeMeanPairs(pairs_dict):
    phe_list = list(pairs_dict.keys())
    
    nums = 0
    for phe in phe_list:
        nums += len(pairs_dict[phe])
    
    return nums / len(phe_list)