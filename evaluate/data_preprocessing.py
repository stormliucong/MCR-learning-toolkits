import numpy as np
import pandas as pd 
import os
import ossaudiodev
from utils.dictionary import load_dictionary
from tqdm import tqdm
import itertools


def split_to_quartiles(indexlist):
    resid = len(indexlist) % 10
    chunk = int((len(indexlist) - resid) / 10)
    
    splitted_list = []
    for i in range(10):
        if i == 9:
            group = indexlist[chunk * i : chunk * (i+1) + resid]
        else:
            group = indexlist[chunk * i : chunk * (i+1)]
        splitted_list.append(group)
        
    return splitted_list

def get_condition_rank_quartiles(freq_dir, concept2id_dir, savedir):
    concept2id = load_dictionary(concept2id_dir)
    concept_set = set(concept2id.keys())
    freq_df = pd.read_csv(freq_dir, escapechar='\\')

    valid_index = []
    for i in tqdm(range(len(freq_df))):
        check1 = len(concept_set.intersection([str(freq_df["concept_id_1"][i])])) == 1
        check2 = len(concept_set.intersection([str(freq_df["concept_id_2"][i])])) == 1
    
        if (check1 and check2 == True):
            valid_index.append(i)
    
    valid_df = freq_df.loc[valid_index]
    
    # sorting
    sorted_df = valid_df.sort_values(by=["count"], ascending=False)
    sorted_index = list(sorted_df.index)
    splitted_index = split_to_quartiles(sorted_index)

    for i in range(10):
        df_1 = list(valid_df["concept_id_1"][splitted_index[i]])
        df_2 = list(valid_df["concept_id_2"][splitted_index[i]])
        assert len(df_1) == len(df_2), "invalid pair exists"
    
        with open(os.path.join(savedir, "condition_rank", i, ".txt"), 'w') as f:
            for ind in range(len(df_1)):
                f.write("%s,%s\n" % (df_1[ind], df_2[ind]))

def get_drug_rank_quartiles():
    pass

def get_cd_rank_quartiles():
    pass

def get_total_rank_quartiles():
    pass

def get_phekb_dict(phekb_data_dir, concept2id):
    """return pheKB dictionary {phe_algorithm : list of concepts in the algorithm}"""
    phe_data = pd.DataFrame.from_csv(phekb_data_dir, sep="\t")
    phe_data["index"] = range(len(phe_data))

    phe_set = set(list(phe_data["phenotype"]))
    filtered_df = pd.concat([phe_data.loc[phe_data["standard_domain"] == ("Condition")],
                      phe_data.loc[phe_data["standard_domain"] == ("Drug")]], ignore_index=True)
    
    phe_dict = {}
    total_set = set(concept2id.keys())
    for phenotype in phe_set:
        concepts = list(filtered_df[filtered_df["phenotype"] == phenotype]["standard_concept_id"])
        concept_set = set(map(str, map(int, list(map(float, concepts)))))
        intersections = total_set.intersection(concept_set)
        concept_list = list(map(str, intersections))
        phe_dict.update({phenotype : list(concept_list)})
    
    return phe_dict

def get_phekb_pairs(phe_dict):
    """return positive pairs under a specific phe algorithm
    phe_algorithm : list of positive pairs"""

    phe_pairs = {}
    for phe in phe_dict.keys():
        concept_list = phe_dict[phe]
        combinations = itertools.combinations(concept_list, 2)
        phe_pairs[phe] = combinations
    
    return phe_pairs






