import numpy as np
from scipy import sparse
import itertools
import sys

class GloVe():
     self.comap = None

     def __init__(self, embedding_dim=128, max_vocab_size=100000, min_occurrences=1, 
     scaling_factor=0.75, cooccurr_ceil=100, batch_size=512, learning_rate=0.01):
          self.embedding_dim = embedding_dim
          self.max_vocab_size = max_vocab_size
          self.min_occurrences = min_occurrences
          self.scaling_factor = scaling_factor
          self.cooccurr_ceil = cooccurr_ceil
          self.batch_size = batch_size
          self.learning_rate = learning_rate
     
     def build_comap(self, vocab, corpus):
          vocab_size = len(vocab)
          self.comap = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)

          for visit in corpus:
               visit_encode = [vocab[concept] for concept in visit]
               permutations = itertools.permutations(visit_encode, 2)
                    for p in permutations:
                         self.comap[p[0], p[1]] += 1
          

     def build_model(self):
          pass