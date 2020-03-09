import tensorflow as tf

class EnhancingModel(tf.keras.Model):
    def __init__(self, config):
        super(EnhancingModel, self).__init__()
        self.config = config # apply json load function later
        self.optimizer = tf.keras.optimizers.Adadelta() # set hparams later
        self.concept2id = load_dictionary(self.config.concept2id)
        self.embedding_layer = None
        self.dense1 = None
        self.phe_classifier = tf.keras.layers.Dense(571, name="phe_classifier") 
        # output dim is the number of phecode classes

    def build_EnhancingNet(self):
        """build context network"""
        self.embedding_layer = tf.keras.layers.Embedding(len(self.concept2id), 1024)
        self.dense1 = tf.keras.layers.Dense(600, name="dense1")

    def forward_pass(self, x):
        h1 = self.dense1(self.embedding_layer(x)) # h1 represents enhanced rep
        return h1

    def compute_cost(self, x, label):
        visit_rep = self.forward_pass(x) / len(x) # avg of all the rep in the visit
        prediction = self.phe_classifier(visit_rep)
        cost = tf.math.reduce_sum(tf.multiply(label, tf.math.log(prediction)), 
        tf.multiply(1 - label, tf.math.log(1 - prediction)))