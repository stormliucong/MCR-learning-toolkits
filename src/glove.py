import numpy as np



def train_glove(vocab, cooccurrences, dim=128, iter=20, **kwargs):
    vocab_size = len(vocab)

    W = ((np.random.rand(vocab_size * 2, dim) - 0.5)
         / float(dim + 1))
    return pass