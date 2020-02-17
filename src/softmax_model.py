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
        self.TargetNet = None
        self.ContextNet = None
        self.optimizer = tf.keras.optimizers.Adadelta() # set hparams
        self.epoch_loss_avg = []
        self.n2v_emb = load_emb_matrix(self.config.npydir) # set config.npydir
        self.glove_emb = load_emb_matrix(self.config.npydir) # set config.npydir
        self.concept2id = load_dictionary(config.dictdir) # set config.dir
        self.num_gpus = self.config.training_settings.num_gpus 
        self.max_len = config.max_len # set config.max_len
        self.build_TargetNet()
        self.build_ContextNet()

    def encode(self, ids):
        """make concatenated vector using n2v_emb and glove_emb"""
        n2v_embs = tf.nn.embedding_lookup(self.n2v_emb, ids)
        glove_embs = tf.nn.embedding_lookup(self.glove_emb, ids)
        return tf.concat([n2v_embs, glove_embs], axis=1)

    def build_TargetNet(self):
        """build target network"""
        inputs_target = tf.keras.layers.Input(shape=(256,))
        h1_target = tf.keras.layers.Dense(196, use_bias=False, name="mlp11")(inputs_target)
        h1_target = tf.keras.layers.BatchNormalization(name="batchnorm11")(h1_target)
        h1_target = tf.keras.layers.PReLU(name="prelu11")(h1_target)
        h2_target = tf.keras.layers.Dense(128, use_bias=False,name="mlp12")(h1_target)
        h2_target = tf.keras.layers.BatchNormalization(name="batchnorm12")(h2_target)
        outputs_target = tf.keras.layers.PReLU(name="prelu12")(h2_target)

        self.TargetNet = tf.keras.Model(inputs=inputs_target, outputs=outputs_target)

    def build_ContextNet(self):
        """build context network"""
        inputs_context = tf.keras.layers.Input(shape=(256,))
        h1_context = tf.keras.layers.Dense(196, use_bias=False, name="mlp21")(inputs_context)
        h1_context = tf.keras.layers.BatchNormalization(name="batchnorm21")(h1_context)
        h1_context = tf.keras.layers.PReLU(name="prelu21")(h1_context)
        h2_context = tf.keras.layers.Dense(128, name="mlp22")(h1_context)
        h2_context = tf.keras.layers.BatchNormalization(name="batchnorm22")(h2_context)
        outputs_context = tf.keras.layers.PReLU(name="prelu22")(h2_context)

        self.ContextNet = tf.keras.Model(inputs=inputs_context, outputs=outputs_context)
    
    def get_enhanced_rep(self):
        """Intended to use after loading trained weights"""
        self.enhanced_rep = self.TargetNet(self.encode(list(range(len(self.concept2id)))))

    def get_context_rep(self):
        """Intended to use after loading trained weights"""
        self.context_rep = self.ContextNet(self.encode(list(range(len(self.concept2id)))))

    def compute_wi(self, x_batch):
        flatten_batch = tf.reshape(x_batch, [-1])
        self.wi = tf.reshape(self.InputNet(self.encode(flatten_batch)), 
        [len(x_batch), self.max_len, 128]) # batch_size * max_len * emb_dim
        
    def compute_wj(self, x_batch):
        flatten_batch = tf.reshape(x_batch, [-1])
        self.wj = tf.reshape(self.ContextNet(self.encode(flatten_batch)), 
        [len(x_batch), self.max_len, 128]) # batch_size * max_len * emb_dim

    def compute_loss(self, x_batch):
        
        self.get_context_rep()
        self.compute_wi(x_batch)
        self.compute_wj(x_batch)
        
        wi_wj = tf.matmul( self.wi, tf.transpose(self.wj, perm=[0,2,1])) # dim : n * l * l
        wi_wj_rsum = tf.reduce_sum(wi_wj, axis=2) # dim : n * l
        boolean_mask = wi_wj_rsum != 0
        wi_wj_ndiag = tf.math.subtract(wi_wj_rsum, tf.linalg.diag_part(wi_wj)) # emb product sum w/o target concept itself
        
        wi_wk = tf.matmul(self.wi, tf.transpose(self.context_rep)) # dim : n * l * k
        wi_wk_max = tf.reduce_max(wi_wk, axis=2) # get the max value in each emb product for target concept
        wi_wk_rsum = tf.reduce_sum(wi_wk, axis=2) # dim : n * l 
        
        noms = tf.math.exp( tf.math.subtract( tf.boolean_mask( wi_wj_ndiag, boolean_mask),
                                             tf.boolean_mask( wi_wk_max, boolean_mask)))
        denoms = tf.math.exp( tf.math.subtract( tf.boolean_mask( wi_wk_rsum, boolean_mask),
                                               tf.boolean_mask( wi_wk_max, boolean_mask)))
        batch_loss = tf.reduce_sum(-tf.math.log(noms / denoms)) / len(x_batch) 

        return batch_loss

    def compute_gradients(self, x_batch):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x_batch)
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
                loss, gradients = self.compute_gradients(x_batch)
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