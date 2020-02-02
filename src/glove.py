import numpy as np

class GloVe():
     def __init__(self, embedding_dim, max_vocab_size=100000, min_occurrences=1, 
     scaling_factor=3/4, cooccurr_ceil=100, batch_size=512, learning_rate=0.05):
          self.embedding_dim = embedding_dim
          self.max_vocab_size = max_vocab_size
          self.min_occurrences = min_occurrences
          self.scaling_factor = scaling_factor
          self.cooccurr_ceil = cooccurr_ceil
          self.batch_size = batch_size
          self.learning_rate = learning_rate
     
     def build_comap(self, vocab, corpus):
          pass

     def build_model(self):
          pass