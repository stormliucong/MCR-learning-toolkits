# manage concept code, concept id, and id in dictionary
from keras.preprocessing import text
import pickle


def build_dictionary(concept_list, savedir):
    '''
    -- concept_list: a list that has concepts
    -- return: concept2id and id2concept dictionary pair
    '''

    if concept_list != list:
        concept_list = list(concept_list)

    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(concept_list)
    concept2id = tokenizer.word_index
    id2concept = {v:k for k, v in concept2id.items()}

    save_dictionary(concept2id, "concept2id", savedir)
    save_dictionary(id2concept, "id2concept", savedir)

def save_dictionary(dict, name, savedir):
    with open(savedir + "/%s.pkl" % name , "wb") as f:
        pickle.dump(dict, f)
    print("%s successfully saved in the savedir" %name)

def load_dictionary(dictdir):
    with open(dictdir, "rb") as f:
        mydict = pickle.load(f)
    return mydict