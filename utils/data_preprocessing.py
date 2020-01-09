# all functions used to preprocess the data.
import pandas as pd


def get_unique_conceptset(csv):
    pairs = pd.read_csv(csv)
    
    concept_list1 = list(map(str, list(pairs["concept_id_1"])))
    concept_list2 = list(map(str, list(pairs["concept_id_2"])))
    concept_set1 = set(concept_list1)
    concept_set2 = set(concept_list2)
    unique_concept_set = concept_set1.union(concept_set2)

    return unique_concept_set

def get_intersect_concepts(concept_set1, concept_set2):

    if type(concept_set1) != set:
        concept_set1 = set(concept_set1)
    if type(concept_set2) != set:
        concept_set2 = set(concept_set2)

    intersections = concept_set1.intersection(concept_set2)

    return intersections

def get_merged_concepts(concept_dict1, concept_dict2):
    '''
    -- concept_dict1: concept dictionary 1
    -- concept_doct2: concept dictionary 2
    -- return: combined concept set
    '''

    assert type(concept_dict1) == dict, "concept_dict1 is not a dictionary"
    assert type(concept_dict2) == dict, "concept_dict2 is not a dictionary"

    concept_ids_1 = set(list(concept_dict1.keys()))
    concept_ids_2 = set(list(concept_dict1.keys()))
    concept_ids_1.update(concept_ids_2)

    return concept_ids_1





