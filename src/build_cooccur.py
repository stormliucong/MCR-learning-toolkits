from scipy import sparse
import itertools
import numpy as np
import sys


def build_cooccur(vocab, corpus):
    """
    --vocab: dict form word2id
    --corpus: list of patient visits
    """
    vocab_size = len(vocab)
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)

    for visit in corpus:
        visit_encode = [vocab[concept] for concept in visit]
        permutations = itertools.permutations(visit_encode, 2)
        for p in permutations:
            cooccurrences[p[0], p[1]] += 1

    return cooccurrences
    
if __name__=='__main__':
    vocab = sys.argv[1]
    corpus = sys.argv[2]
    build_cooccur(vocab, corpus)

    # set the dir to save
