import tensorflow as tf
import random
import numpy as np
import os
from utils.data_preprocessing import get_ids
from utils.data_loader import load_dict, load_emb_matrix

class EnhancingNet(tf.keras.Model):
    def __init__(self, config):
        super(EnhancingNet, self).__init__()
        self.config = config
        self.InputNet = None
        self.ContextNet = None
        self.optimizer = tf.keras.optimizers.Adadelta()
        self.epoch_loss_avg = []
        self.n2v_emb = load_emb_matrix()
        self.glove_emb = load_emb_matrix()
        self.concept2id = #load_dict(config.dir)
        self.num_gpus = self.config.training_settings.num_gpus
        self.max_len = None
        self.build_InputNet()
        self.build_ContextNet()

    def encode(self, ids):
        """make concatenated vector using n2v_emb and glove_emb"""
        n2v_embs = tf.nn.embedding_lookup(self.n2v_emb, ids)
        glove_embs = tf.nn.embedding_lookup(self.glove_emb, ids)
        return tf.concat([n2v_embs, glove_embs], axis=1)

    def build_InputNet(self):
        """build input network"""
        inputs = tf.keras.layers.Input(shape=(256,))
        h1 = tf.keras.layers.Dense(196, use_bias=False, name="mlp11")(inputs)
        h1 = tf.keras.layers.BatchNormalization(name="batchnorm11")(h1)
        h1 = tf.keras.layers.PReLU(name="prelu11")(h1)
        h2 = tf.keras.layers.Dense(128, use_bias=False,name="mlp12")(h1)
        h2 = tf.keras.layers.BatchNormalization(name="batchnorm12")(h2)
        outputs = tf.keras.layers.PReLU(name="prelu12")(h2)

        self.InputNet = tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_ContextNet(self):
        """build context network"""

        inputs = tf.keras.layers.Input(shape=(256,))
        h1 = tf.keras.layers.Dense(196, use_bias=False, name="mlp21")(inputs)
        h1 = tf.keras.layers.BatchNormalization(name="batchnorm21")(h1)
        h1 = tf.keras.layers.PReLU(name="prelu21")(h1)
        h2 = tf.keras.layers.Dense(128, name="mlp22")(h1)
        h2 = tf.keras.layers.BatchNormalization(name="batchnorm22")(h2)
        outputs = tf.keras.layers.PReLU(name="prelu22")(MLP_layer_2_2)

        self.ContextNet = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def get_enhanced_rep(self):
        """Intended to use after loading trained weights"""
        self.enhanced_rep = self.InputNet(self.encode(list(range(len(model.concept2id)))))

    def compute_X(self, batch_size):
        self.get_enhanced_rep()
        self.X = tf.reshape(tf.tile(self.enhanced_rep, [batch_size, 1]), 
        [batch_size, len(self.concept2id), 128]) # batch_size * total_concepts * emb_dim

    def compute_v(self, x_batch):
        flatten_batch = tf.reshape(x_batch, [-1])
        self.v = tf.reshape(self.InputNet(self.encode(flatten_batch)), 
        [len(x_batch), self.max_len, 128]) # batch_size * max_len * emb_dim
        # add self.max_len

@tf.function
def compute_loss(model, x_batch):
    """
    --model: Enhancing model
    --x_batch: designated size of x
    --k: total number of concepts
    """
    p_vec, i_vec, j_vec = padMatrix(x_batch) 
    model.compute_normat(x_batch[0])
    matmul_vX = tf.matmul(model.v, tf.transpose(model.X, [0,2,1])) # n * l * k matrix
    denom_mat = tf.reduce_sum(matmul_vX, axis=-1) # n * l matrix

    nom_ids = tf.transpose([p_vec, i_vec, j_vec]) # length = n * l(l-1)
    denom_ids = tf.transpose([p_vec, i_vec]) # length = n * l(l-1)

    noms = tf.exp(tf.gather_nd(matmul_vX, nom_ids))
    denoms = tf.exp(tf.gather_nd(denom_mat, denom_ids))

    batch_loss = tf.math.reduce_sum(-tf.math.log(noms / denoms), axis=0) / x_batch[0]
    # batch training : take average

    return batch_loss

@tf.function
def compute_gradients(model, x_batch):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x_batch)
        
    return loss, tape.gradient(loss, model.trainable_variables)
