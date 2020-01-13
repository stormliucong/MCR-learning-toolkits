# all functions used to preprocess the data.
import pandas as pd
import numpy as np
import gensim
import math
from utils import dictionary
from tqdm import tqdm
from keras.preprocessing.sequence import skipgrams

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
    np.save(embmatrix, savedir + "/%s.npy" % name)
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

def get_condition_concepts(csvpair, condition_emb):

    concept_from_pair = get_unique_conceptset(csvpair)
    concept_from_emb = set()

    condition2emb = load_emb(condition_emb)
    concept_from_emb.update(list(condition2emb.keys()))

    condition_concepts = concept_from_emb.intersection(concept_from_pair)

    return condition_concepts

def get_drug_concepts(csvpair, drug_emb):

    concept_from_pair = get_unique_conceptset(csvpair)
    concept_from_emb = set()

    drug2emb = load_emb(drug_emb)
    concept_from_emb.update(list(drug2emb.keys()))

    drug_concepts = concept_from_emb.intersection(concept_from_pair)

    return drug_concepts


def generate_training_pairs(pairs_csv, concept2id_dir, condition_emb, drug_emb, savedir):
    '''
    -- pairs_csv: directory of the pairs csv file
    -- concept2id: directory of the concept2id dictionary file
    -- savedir: directory in which you want to save training pairs
    -- return: 
    '''

    total_pairs = []
    pairs = pd.read_csv(pairs_csv)
    concept2id = dictionary.load_dictionary(concept2id_dir)
    condition_set = get_condition_concepts(pairs_csv, condition_emb)
    drug_set = get_drug_concepts(pairs_csv, drug_emb)
    
    for i in tqdm(range(pairs.shape[0])):
        #filtering and ordering
        concept_1 = pairs.loc[i]["concept_id_1"]
        concept_2 = pairs.loc[i]["concept_id_2"]

        test_1 = len(condition_set.intersection([concept_1]))
        test_2 = len(drug_set.intersection([concept_2]))

        if (test_1 + test_2 == 2):
            candidate_pair = [concept_1, concept_2]
        elif (test_1 + test_2 == 0):
            candidate_pair = [concept_2, concept_1]
        
        #encoding
        encoded_pair = [concept2id[candidate_pair[0]], concept2id[candidate_pair[1]]]

        #make sgpair
        sg = skipgrams(encoded_pair, len(concept2id), 0.5)
        positive = np.array(sg[0])[np.array(sg[1]) == 1]
        negative = np.array(sg[0])[np.array(sg[1]) == 0]

        positive_pair = positive[0].tolist()
        negative_pair = negative[0].tolist()
        positive_pair.append(1)
        negative_pair.append(0)

        total_pairs.append(positive_pair)
        total_pairs.append(negative_pair)
    
    with open(savedir + "/training_pairs.txt", "w") as f:
        for pair in total_pairs:
            f.write("%s, %s\n" % pair[0], pair[1])
    
def load_training_pairs(training_pairs_dir):
    with open(training_pairs_dir, "r") as f:
        body = f.read()
        raw_list = body.split("\n")

        training_pairs = []
        for pair in raw_list:
            training_pairs.append(pair.split(", "))
        
    return training_pairs
        
def split_into_batch(training_pairs_dir, num_lines, save_dir):
    '''
    split the entire training pairs file into the specified number of batches
    -- training_pairs_dir: 
    -- num_lines:
    -- save_dir:
    -- return: 
    '''

    training_pairs = load_training_pairs(training_pairs_dir)
    num_batch = math.ceil(len(training_pairs) / num_lines)

    for i in tqdm(range(num_batch)):
        if i != (num_batch-1):
            pairs = training_pairs[i * num_lines : (i+1) * num_lines]
        elif i == (num_batch-1):
            pairs = training_pairs[i * num_lines :]
        
        with open(save_dir + "/batch_pairs_%s.txt" % i, 'w') as f:
            for pair in pairs:
                content = str(pair[0]) + " " + str(pair[1]) + "\n"
                f.write(content)
        file_num = i

    print("total %s batch files have been saved" % file_num)





