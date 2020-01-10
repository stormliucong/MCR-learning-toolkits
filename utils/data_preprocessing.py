# all functions used to preprocess the data.
import pandas as pd
import numpy as np
import gensim
from utils import dictionary
from tqdm import tqdm

def get_unique_conceptset(csv):
    '''
    read csv format pair data and output unique concept set
    -- csv: csv format concept pairs
    -- return: unique concept set in the concept pair data 
    '''

    pairs = pd.read_csv(csv)
    
    concept_list1 = list(map(str, list(pairs["concept_id_1"])))
    concept_list2 = list(map(str, list(pairs["concept_id_2"])))
    concept_set1 = set(concept_list1)
    concept_set2 = set(concept_list2)
    unique_concept_set = concept_set1.union(concept_set2)

    return unique_concept_set

def load_emb(csvemb):
    '''
    load csv format embedding
    -- csvemb: a csv file contain embedding results
    -- return: a dictionary of embeddings
    '''

    embeddings = pd.read_csv(csvemb)
    concept_ids = list(embeddings["standard_concept_id"])
    embs = list(embeddings["embedding"])
    assert len(concept_ids) == len(set(concept_ids)), "Duplicated concepts in the data"
    assert len(concept_ids) == len(embs), "The total number of concepts and embeddigns does not match"

    concept2emb = {}
    
    for i in range(len(concept_ids)):
        concept2emb[str(concept_ids)] = embs[i].split(",")

    return concept2emb

def get_intersect_concepts(csvpair, condition_emb, drug_emb):

    concept_from_pair = get_unique_conceptset(csvpair)
    concept_from_emb = set()

    condition2emb = load_emb(condition_emb)
    drug2emb = load_emb(drug_emb)

    concept_from_emb.update(list(condition2emb.keys()))
    concept_from_emb.update(list(drug2emb.keys()))

    intersection_concepts = concept_from_emb.intersection(concept_from_pair)

    return intersection_concepts
    
def save_embmatrix(embmatrix, name, savedir):
    np.save(savedir + "/%s.npy" % name)
    print("%s successfully saved in the savdir" % name)

def build_n2v_matrix(n2v_model_dir, concept2id_dir, savedir):
    '''
    Build weight matrix of n2v embedding
    -- n2v_model_dir: file directory of n2v embedding model
    -- concept2id_dir: file directory of concept2id dictionary
    -- return: embedding matrix indexed by concept2id dictionary array shaped (# of concepts, dim)
    '''

    n2v_model = gensim.models.KeyedVectors.load_word2vec_format(n2v_model_dir)
    concept2id = dictionary.load_dictionary(concept2id_dir)

    # construct initial embedding matrix
    matrix_n2v = np.zeros((len(concept2id), 128), dtype="float32")

    total_concepts = list(concept2id.keys())
    for i in tqdm(range(len(total_concepts))):
        current_concept = total_concepts[i]
        ind = concept2id[current_concept]-1
        matrix_n2v[ind,] = n2v_model[current_concept]
    
    save_embmatrix(matrix_n2v, "n2v_matrix", savedir)

def build_glove_matrix(condition_emb_dir, drug_emb_dir, concept2id_dir, savedir):

    concept2emb = {}
    concept2id = dictionary.load_dictionary(concept2id_dir)
    concept2emb.update(load_emb(condition_emb_dir))
    concept2emb.update(load_emb(drug_emb_dir))

    matrix_glove = np.zeros((len(concept2id), 128), dtype="float32")
    total_concepts = list(concept2id.keys())
    
    for i in tqdm(range(len(total_concepts))):
        current_concept = total_concepts[i]
        ind = concept2id[current_concept]-1
        matrix_glove[ind,] = concept2emb[current_concept]

    save_embmatrix(matrix_glove, "glove_matrix", savedir)

def generate_training_pairs(pairs_csv, concept2id, savedir):

    #filtering

    #encoding

    #make sgpair

    #filtering

    #save 

    pass

    






