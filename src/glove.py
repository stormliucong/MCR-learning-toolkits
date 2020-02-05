import numpy as np
import tensorflow as tf
from scipy import sparse
import itertools
import sys

class GloVe():

     def __init__(self, embedding_dim=128, max_vocab_size=100000, min_occurrences=1, 
     scaling_factor=0.75, cooccurrence_ceil=100, batch_size=512, learning_rate=0.01):
          self.embedding_dim = embedding_dim
          self.max_vocab_size = max_vocab_size
          self.min_occurrences = min_occurrences
          self.scaling_factor = scaling_factor
          self.cooccurrence_ceil = cooccurrence_ceil
          self.batch_size = batch_size
          self.learning_rate = learning_rate
          self.vocab_size = 0
          self.comap = None
     
     def build_comap(self, vocab, corpus):
          self.vocab_size = len(vocab)
          self.comap = sparse.lil_matrix((self.vocab_size, self.vocab_size), dtype=np.float64)

          for visit in corpus:
               visit_encode = [vocab[concept] for concept in visit]
               permutations = itertools.permutations(visit_encode, 2)
               for p in permutations:
                    self.comap[p[0], p[1]] += 1

     def build_model(self):
          count_max = tf.constant([self.cooccurrence_ceil], dtype=tf.float64)
          scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float64)
          
          target_embeddings = tf.Variable(
               tf.random_uniform([self.vocab_size, self.embedding_dim], 1.0, -1.0),
               name="target_embeddings")
          context_embeddings = tf.Variable(
               tf.random_uniform([self.vocab_size, self.embedding_dim], 1.0, -1.0),
               name="context_embeddings")

          target_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
          name='target_biases')
          context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
          name="context_biases")

          pass