import numpy as np
import tensorflow as tf
from tqdm import tqdm
import itertools
import random
import sys
import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class GloVe(tf.keras.Model):
     def __init__(self, embedding_dim=128, max_vocab_size=100, 
     scaling_factor=0.75, batch_size=512, learning_rate=0.01):
          super(GloVe, self).__init__()
          self.embedding_dim = embedding_dim
          self.max_vocab_size = max_vocab_size
          self.scaling_factor = scaling_factor
          self.batch_size = batch_size
          self.vocab_size = 0
          self.concept2id = None
          self.comap = None
          self.comatrix = None
          self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
          self.epoch_loss_avg = []

     def build_dict(self, corpus):
          tokenizer = tf.keras.preprocessing.text.Tokenizer()
          tokenizer.fit_on_texts(corpus)
          self.concept2id = tokenizer.word_index
          self.concept2id.update({"0" : 0})

     def save_dict(self, save_dir):
          with open(save_dir + "/concept2id.pkl", "wb") as f:
               pickle.dump(self.concept2id, f)
          print("concept2id successfully saved in the savedir")
     
     def fit_to_corpus(self, corpus):
          self.comap = defaultdict(float)
          self.comatrix = np.zeros((len(self.concept2id), len(self.concept2id)), dtype=np.float64)
          concept2id = self.concept2id

          for i in tqdm(range(len(corpus))):
               patient = corpus[i]
               for p in patient:
                    for k in patient:
                         if p != k:
                              self.comap[(p, k)] += 1
        
          for pair, count in self.comap.items():
               self.comatrix[concept2id[pair[0]], concept2id[pair[1]]] = count

     def init_params(self):
          with tf.device("/cpu:0"):
               """must be implemented with cpu-only env since this is sparse updating"""
               self.target_embeddings = tf.Variable(
                    tf.random.uniform([self.vocab_size, self.embedding_dim], 0.1, -0.1),
                    name="target_embeddings")
               self.context_embeddings = tf.Variable(
                    tf.random.uniform([self.vocab_size, self.embedding_dim], 0.1, -0.1),
                    name="context_embeddings")
               self.target_biases = tf.Variable(tf.random.uniform([self.vocab_size], 0.1, -0.1),
               name='target_biases')
               self.context_biases = tf.Variable(tf.random.uniform([self.vocab_size], 0.1, -0.1),
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
          emb_product = tf.math.reduce_sum(tf.math.multiply(target_emb, context_emb), axis=1)
          log_cooccurrence = tf.math.log(tf.cast(x[2], dtype=tf.float32))

          distance_cost = tf.math.square(
               tf.math.add_n([emb_product, target_bias, context_bias, 
               tf.math.negative(log_cooccurrence)]))
               
          batch_cost = tf.math.reduce_sum(tf.multiply(weight, distance_cost))
          
          return batch_cost

     def compute_gradients(self, x):
          with tf.GradientTape() as tape:
               cost = self.compute_cost(x)
          return cost, tape.gradient(cost, self.trainable_variables)

     def prepare_batch(self):

          i_ids = []
          j_ids = []
          co_occurs = []

          for i in range(self.comatrix.shape[0]):
               for j in range(self.comatrix.shape[0]):
                    if i == j: continue
                    i_ids.append(i)
                    j_ids.append(j)
                    co_occurs.append(self.comatrix[i, j])
     
          assert len(i_ids) == len(j_ids), "The length of the data are not the same"
          assert len(i_ids) == len(co_occurs), "The length of the data are not the same"
          return i_ids, j_ids, co_occurs

     def get_embeddings(self):
          self.embeddings = self.target_embeddings + self.context_embeddings

     def generate_tsne(self, size=(10, 10)):
          tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
          low_dim_embs = tsne.fit_transform(self.embeddings[:, :])
          labels = list(range(self.embeddings.shape[0]))
        
          return plot_with_labels(low_dim_embs, labels, size)

     def train_GloVe(self, num_epochs, save_dir, saving_term):
          i_ids, j_ids, co_occurs = self.prepare_batch()
          total_batch = int(np.ceil(len(i_ids) / self.batch_size))
          cost_avg = tf.keras.metrics.Mean()

          for epoch in range(num_epochs):

               progbar = tf.keras.utils.Progbar(len(i_ids))

               for i in range(total_batch):
                    i_batch = i_ids[i * self.batch_size : (i+1) * self.batch_size]
                    j_batch = j_ids[i * self.batch_size : (i+1) * self.batch_size]
                    co_occurs_batch = co_occurs[i * self.batch_size : (i+1) * self.batch_size]
                    cost, gradients = self.compute_gradients([i_batch, j_batch, co_occurs_batch])
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    cost_avg(cost) 
                    progbar.add(self.batch_size)
                    print("Step {}: Loss: {:.4f}".format(self.optimizer.iterations.numpy(), cost))

               if (epoch % 1) == 0: 
                    avg_loss = cost_avg.result()
                    print("Epoch {}: Loss: {:.4f}".format(epoch, avg_loss))
                    self.epoch_loss_avg.append(avg_loss)
                    
               if (epoch % saving_term) == 0:
                    self.save_weights(os.path.join(save_dir, 
                    "e{:03d}_loss{:.4f}.ckpt".format(epoch, avg_loss)))

def plot_with_labels(low_dim_embs, labels, size):
     assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
     figure = plt.figure(figsize=size)  # in inches
     for i, label in enumerate(labels):
          x, y = low_dim_embs[i, :]
          plt.scatter(x, y)
          plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')