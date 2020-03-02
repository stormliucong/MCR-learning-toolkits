import pandas as pd
from collections import OrderedDict
import pickle

class PhenotypingEval():
    """Class for phenotyping task evaluation"""
    def __init__(self, config):
        self.condition_pairs = None
        self.drug_pairs = None
        self.cross_pairs = None
        self.total_pairs = None
        self.condition_phe = OrderedDict()        
        self.drug_phe = OrderedDict() 
        self.total_phe = OrderedDict() 
        self.enhanced_emb = load_dictionary(config.enhanced_emb_dir)
        self.n2v_emb = load_dictionary(config.n2v_emb_dir)
        self.glove_emb = load_dictionary(config.glove_emb_dir)
        self.concept2id = load_dictionary(config.concept2id_dir)

    def setPheDict(self):
        phe_data = pd.DataFrame.from_csv(config.phe_data_dir, sep="\t")
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
    
    def getPhePairs(self):
        pass

    def getNegativePairs(self):
        pass

    def computeSims(self):
        pass

    def plotSimHist(self):
        pass
        
# package-wide functions

def load_dictionary(pklfile):
    f = open(pklfile, "rb")
    dict_load = pickle.load(f)
    return dict_load

def getIntersections(concept_list, concept_range_set):
    concept_set = set(concept_list)
    intersections = list(concept_set.intersection(concept_range_set))
    return intersections        