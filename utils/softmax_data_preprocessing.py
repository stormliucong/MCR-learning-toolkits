import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as sparse
from src.data_loader import load_dictionary

def to_conceptlist(csr_matrix):
    """csr matrix to a list of patient record that consists of several concepts"""
    patient_list = []

    for i in tqdm(range(csr_matrix.shape[0])):
        dense_row = csr_matrix[i].todense()
        concept_record = list(dense_row.nonzero()[1])
        if len(concept_record) > 1:
            patient_list.append(concept_record)
    
    return patient_list

def to_csr(config):

    condition_df = raw_toDataFrame(config.data_dir)
    drug_df = raw_toDataFrame(config.data_dir)
    concat_df = pd.concat([condition_df, drug_df], ignore_index=True)
    
    patient_count = get_count(concat_df, "patient_id")
    concept_count = get_count(concat_df, "concept_id")
    unique_pid = patient_count.index
    unique_cid = concept_count.index
    patient2id = dict((uid, i) for (i, uid) in enumerate(unique_pid))
    concept2id = dict((cid, i) for (i, cid) in enumerate(unique_cid))

    pid = map(lambda x: patient2id[x], concat_df["patient_id"])
    cid = map(lambda x: concept2id[x], concat_df["concept_id"])
    patient_df = pd.DataFrame(data={'pid': list(pid), 'cid': list(cid)})

    return to_SparseMatrix(patient_df), concept2id

def raw_toDataFrame(data_dir):
    concept_record = []

    with open(data_dir, "r") as f:
        body = f.read()
    body = body.split("\n")
    concepts = body[2:-4]

    for i in tqdm(range(len(concepts))):
        raw_pair = concepts[i].split(" ")
        while ("" in raw_pair):
            raw_pair.remove("")
        concept_record.append(raw_pair)
    
    record_df = pd.DataFrame(concept_record)
    record_df = record_df[[0,1]]
    record_df = record_df.rename(columns={0 : "patient_id", 1 : "concept_id"})

    return record_df
    
def get_count(df, col_name):
    assert type(col_name) == str, "column name must be string type"
    count_groupbyid = df[[col_name]].groupby(df, as_index=False)
    count = count_groupbyid.size()
    return count

def to_SparseMatrix(data):
    """DataFrame to CSR"""
    n_users = data['pid'].max() + 1
    n_items = len(data["cid"].unique())

    rows, cols = data['pid'], data['cid']
    csr_matrix = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return csr_matrix