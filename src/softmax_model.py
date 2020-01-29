import tensorflow as tf
import random
import numpy as np
import os
from utils.data_loader import load_dict, load_emb_matrix

class EnhancingNet(tf.keras.Model):
    def __init__(self, config):
        super(EnhancingNet, self).__init__()
        self.config = config
        self.InputNet = None
        self.ContextNet = None
        self.optimizer = tf.keras.optimizers.Adadelta()
        self.gradients = None
        self.epoch_loss_avg = []
        self.n2v_emb = #load_emb_matrix(config.dir)
        self.glove_emb = #load_emb_matrix(config.dir)
        self.concept2id = #load_dirc(config.dir)
        self.num_gpus = self.config.training_settings.num_gpus
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
        enhanced_rep = self.InputNet(range(len(model.concept2id)+1))

        return enhanced_rep

@tf.function
def compute_loss(model, x):
    i_vec, j_vec, n_vec = get_ids(x, len(model.concept2id))
    positive_product = tf.math.multiply(model.InputNet(model.encode(i_vec)), model.ContextNet(model.encode(j_vec)))
    negative_product = tf.math.multiply(model.InputNet(model.encode(i_vec)), model.ContextNet(model.encode(n_vec)))

    positive_noms = tf.math.reduce_sum(tf.split(tf.math.reduce_sum(positive_product, axis=1), len(x)), axis=1)
    negative_noms = tf.math.reduce_sum(tf.split(tf.math.reduce_sum(negative_product, axis=1), len(x)), axis=1)
    noms = tf.exp(positive_noms)
    denoms = tf.exp(tf.math.add(positive_noms, negative_noms))

    loss = tf.math.reduce_sum(-tf.math.log(noms / denoms), axis=0)

    return loss

@tf.function
def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        
    return loss, tape.gradient(loss, model.trainable_variables)

def model_train(model, data_dir, config):
    
    # train_data = load_data(dir)
    # need data_load functions
    for epoch in range(config.num_epochs):
        loss_avg = tf.keras.metrics.Mean()

        for x_train in train_dataset: 
            loss, gradients = compute_gradients(model, x_train)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            loss_avg(loss) # fix loss printing
            print("Step {}: Loss: {:.4f}".format(model.optimizer.iterations.numpy(), loss))
        
        if epoch % 1 == 0:
            avg_loss = loss_avg.result()
            model.epoch_loss_avg.append(avg_loss)
            model.save_weights(os.path.join(config.save_dir, "e{:03d}_loss{:.4f}.ckpt".format(epoch, avg_loss)))
            print("Epoch {}: Loss: {:.4f}".format(epoch, loss_avg.result()))