import tensorflow as tf
import random
import numpy as np
from utils.data_loader import load_dict, load_emb_matrix

class EnhancingNet_CE(tf.keras.Model):
    """Enhancing model using element-wise cross-entropy loss"""
    def __init__(self, config):
        super(EnhancingNet_CE, self).__init__()
        self.config = config
        self.TargetNet = None
        self.optimizer = tf.keras.optimizers.Adadelta()
        self.epoch_loss_avg = []
        self.n2v_emb = #load_emb_matrix(config.dir)
        self.glove_emb = #load_emb_matrix(config.dir)
        self.concept2id = #load_dirc(config.dir)
        self.num_gpus = self.config.training_settings.num_gpus
        self.build_TargetNet()

    def encode(self, ids):
        n2v_matrix = tf.nn.embedding_lookup(self.n2v_emb, ids)
        glove_matrix = tf.nn.embedding_lookup(self.glove_emb, ids)
        return tf.concat([n2v_matrix, glove_matrix], axis=1)

    def build_TargetNet(self):
        """build input network"""
        inputs = tf.keras.layers.Input(shape=(256,))
        MLP_layer_1 = tf.keras.layers.Dense(196, use_bias=False, name="mlp11")(inputs)
        MLP_layer_1 = tf.keras.layers.BatchNormalization(name="batchnorm11")(MLP_layer_1)
        MLP_layer_1 = tf.keras.layers.PReLU(name="prelu11")(MLP_layer_1)
        MLP_layer_2 = tf.keras.layers.Dense(128, use_bias=False,name="mlp12")(MLP_layer_1)
        MLP_layer_2 = tf.keras.layers.BatchNormalization(name="batchnorm12")(MLP_layer_2)
        outputs = tf.keras.layers.PReLU(name="prelu12")(MLP_layer_2)

        self.TargetNet = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_PatientNet(self):

        enhanced_rep = tf.keras.layers.Input(shape=(128,))
        outputs = tf.keras.layers.Dense(len(model.concept2id), name="mlp_patient")(enhanced_rep)

        self.PatientNet = tf.keras.Model(inputs=enhanced_rep, outputs=outputs)

@tf.function
def compute_loss(model, x):
    return 

@tf.function
def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        
    return loss, tape.gradient(loss, model.trainable_variables)
