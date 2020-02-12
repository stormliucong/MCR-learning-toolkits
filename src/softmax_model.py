import tensorflow as tf
import random
import numpy as np
import os
from src.data_loader import load_dictionary, load_emb_matrix

class EnhancingNet(tf.keras.Model):
    def __init__(self, config):
        super(EnhancingNet, self).__init__()
        self.config = config
        self.InputNet = None
        self.ContextNet = None
        self.optimizer = tf.keras.optimizers.Adadelta() # set hparams
        self.epoch_loss_avg = []
        self.n2v_emb = load_emb_matrix(self.config.npydir) # set config.npydir
        self.glove_emb = load_emb_matrix(self.config.npydir) # set config.npydir
        self.concept2id = load_dictionary(config.dictdir) # set config.dir
        self.num_gpus = self.config.training_settings.num_gpus 
        self.max_len = config.max_len # set config.max_len
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
        outputs = tf.keras.layers.PReLU(name="prelu22")(h2)

        self.ContextNet = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def get_enhanced_rep(self):
        """Intended to use after loading trained weights"""
        self.enhanced_rep = self.InputNet(self.encode(list(range(len(self.concept2id)))))

    def get_context_rep(self):
        """Intended to use after loading trained weights"""
        self.context_rep = self.ContextNet(self.encode(list(range(len(self.concept2id)))))

    def compute_v(self, x_batch):
        flatten_batch = tf.reshape(x_batch, [-1])
        self.v = tf.reshape(self.InputNet(self.encode(flatten_batch)), 
        [len(x_batch), self.max_len, 128]) # batch_size * max_len * emb_dim
        # add self.max_len

    def save_embeddings(self, save_dir, epoch):
        self.get_enhanced_rep()
        self.get_context_rep()
        np.save(os.path.join(save_dir, "enhanced_rep_e{:03d}.npy".format(epoch)), self.enhanced_rep)
        np.save(os.path.join(save_dir, "context_rep_e{:03d}.npy".format(epoch)), self.context_rep)