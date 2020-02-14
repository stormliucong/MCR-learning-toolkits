import tensorflow as tf
import random
import numpy as np
import os
import random
from src.data_loader import load_dictionary, load_emb_matrix, load_train_data

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

    def compute_loss(self, x_batch, p_vec, i_vec, j_vec):
        self.get_context_rep()
        self.compute_v(x_batch)
        matmul_vX = tf.linalg.normalize(tf.matmul(self.v, tf.transpose(self.context_rep)), axis=-1, ord=1)[0] # n * l * k matrix
        denom_mat = tf.math.reduce_sum(matmul_vX, axis=-1) # n * l matrix

        nom_ids = tf.transpose([p_vec, i_vec, j_vec]) # length = n * l(l-1)
        denom_ids = tf.transpose([p_vec, i_vec]) # length = n * l(l-1)

        noms = tf.exp(tf.gather_nd(matmul_vX, nom_ids))
        denoms = tf.exp(tf.gather_nd(denom_mat, denom_ids))

        batch_loss = tf.math.reduce_sum(-tf.math.log(noms / denoms), axis=0) / len(x_batch)
        # batch training : take average
        return batch_loss

    def compute_gradients(self, x_batch, p_vec, i_vec, j_vec):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x_batch, p_vec, i_vec, j_vec)
        return loss, tape.gradient(loss, self.trainable_variables)
    
    def model_train(self, batch_size, num_epochs):
        train_data = load_train_data(self.config.train_data_dir) # load padded train data
        for epoch in range(num_epochs):
            total_batch = int(np.ceil(len(train_data) / batch_size))
            loss_avg = tf.keras.metrics.Mean()
            progbar = tf.keras.utils.Progbar(total_batch)
            shuffled_data = shuffle_data(train_data)

            for i in range(total_batch):
                x_batch = shuffled_data[i * batch_size : (i+1) * batch_size]
                p_vec, i_vec, j_vec = padMatrix(x_batch)
            
                loss, gradients = self.compute_gradients(x_batch, p_vec, i_vec, j_vec)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
                loss_avg(loss) 
                progbar.add(1)
                print("Step {}: Loss: {:.4f}".format(self.optimizer.iterations.numpy(), loss))
        
            if epoch % 1 == 0:
                avg_loss = loss_avg.result()
                self.epoch_loss_avg.append(avg_loss)
                print("Epoch {}: Loss: {:.4f}".format(epoch, loss_avg.result()))


    def save_embeddings(self, save_dir, epoch):
        self.get_enhanced_rep()
        self.get_context_rep()
        np.save(os.path.join(save_dir, "enhanced_rep_e{:03d}.npy".format(epoch)), self.enhanced_rep)
        np.save(os.path.join(save_dir, "context_rep_e{:03d}.npy".format(epoch)), self.context_rep)

def shuffle_data(data):
    train_data_shuffled = []
    shuffle_index = list(range(len(data)))
    random.shuffle(shuffle_index)
    for ind in shuffle_index:
        train_data_shuffled.append(data[ind])
    return train_data_shuffled

def get_permutations(idx, seq, i_vec, j_vec, p_vec):
    for first in seq:
        for second in seq:
            if first == 0:
                continue
            if second == 0:
                continue
            if first == second: 
                continue
            i_vec.extend(np.where(seq == first)[0])
            j_vec.append(second)
            p_vec.append(idx)

def padMatrix(x_batch):
    """
    """
    p_vec = []
    i_vec = []
    j_vec = []
    
    for idx, seq in enumerate(x_batch):
        get_permutations(idx, seq, i_vec, j_vec, p_vec)
    p_vec = tf.reshape(p_vec, [-1])

    return p_vec, i_vec, j_vec