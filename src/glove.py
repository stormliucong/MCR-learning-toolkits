import numpy as np
import tensorflow as tf
from scipy import sparse
import itertools
import sys

class GloVe(tf.keras.Model):
     def __init__(self, embedding_dim=128, max_vocab_size=10000, min_occurrences=1, 
     scaling_factor=0.75, cooccurrence_ceil=100, batch_size=512, learning_rate=0.01):
          super(GloVe, self).__init__()
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

     def init_params(self):
          self.target_embeddings = tf.Variable(
               tf.random.uniform([self.vocab_size, self.embedding_dim], 1.0, -1.0),
               name="target_embeddings")
          self.context_embeddings = tf.Variable(
               tf.random.uniform([self.vocab_size, self.embedding_dim], 1.0, -1.0),
               name="context_embeddings")
          self.target_biases = tf.Variable(tf.random.uniform([self.vocab_size], 1.0, -1.0),
          name='target_biases')
          self.context_biases = tf.Variable(tf.random.uniform([self.vocab_size], 1.0, -1.0),
          name="context_biases")

    def compute_cost(self, x):
        """x = [target_ind, context_ind, co_occurrence_count]"""
        target_emb = tf.nn.embedding_lookup([self.target_embeddings], x[0])
        context_emb = tf.nn.embedding_lookup([self.context_embeddings], x[1])
        target_bias = tf.nn.embedding_lookup([self.target_biases], x[0])
        context_bias = tf.nn.embedding_lookup([self.context_biases], x[1])

        weight = tf.math.minimum(1.0, 
        tf.cast(tf.math.pow(
             tf.math.truediv(x[2], self.max_vocab_size), 
             self.scaling_factor),
             dtype=tf.float32))
        emb_product = tf.math.reduce_sum(tf.multiply(target_emb, context_emb), axis=0)
        log_cooccurrence = tf.math.log(tf.cast(x[2], dtype=tf.float32))

        distance_cost = tf.math.square(
               tf.math.add_n([emb_product, target_bias, context_bias, 
               tf.math.negative(log_cooccurrence)]))
          
        cost = tf.multiply(weight, distance_cost)

        return cost


     def compute_gradients(self, x):
          with tf.GradientTape() as tape:
               loss = model.compute_cost(x)
     return loss, tape.gradient(loss, self.trainable_variables)

     def get_batches(self):


     def train_GloVe(self, num_epochs, save_dir):

          for epoch in range(num_epochs):
               pass
     pass