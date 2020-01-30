import pandas as pd
from tqdm import tqdm
from src.data_loader import load_dictionary

def to_CSR(config):

    concept2id = load_dictionary(config.dict_dir)

    condition_df = raw_toDataFrame(config.data_dir)
    drug_df = raw_toDataFrame(config.data_dir)
    concat_df = pd.concat([condition_df, drug_df], ignore_index=True)
    


    return 

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

