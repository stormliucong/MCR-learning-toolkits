# put model architecture here.
import keras
from keras.layers import Input, Flatten, dot, concatenate
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.layers.advanced_activations import PReLU
from src.data_loader import load_dictionary, build_weight_matrix

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.condition_model = None
        self.drug_model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save_weights(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model has been saved")

    def save_architecture(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")
        
        print("Saving model architecture")
        self.condition_model.to_json()

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError


class EnhancedModel(BaseModel):
    '''
    Define the EnhancedModel Class
    '''

    def __init__(self,config):
        super(EnhancedModel, self).__init__(config)
        self.weights_n2v = build_weight_matrix(config.weight_matrix.n2v_weights_dir)
        self.weights_glove = build_weight_matrix(config.weight_matrix.glove_weights_dir)
        self.concept2id = load_dictionary(config.dictionary.concept2id_dictionary)
        self.num_gpus = config.trainer.num_gpus
        self.build_model()

    def build_model(self):
        # first concept side
        n2v_input_1 = Input(shape=(1,))
        n2v_emb_layer_1 = Embedding(len(self.concept2id)+1, 128, input_length=1, name="n2vembedding_1", trainable=False) 
        n2v_emb_1 = n2v_emb_layer_1(n2v_input_1)
        n2v_emb_1 = Flatten()(n2v_emb_1)

        glove_input_1 = Input(shape=(1,))
        glove_emb_layer_1 = Embedding(len(self.concept2id)+1, 128, input_length=1, name="gloveembedding_1", trainable=False) 
        glove_emb_1 = glove_emb_layer_1(glove_input_1)
        glove_emb_1 = Flatten()(glove_emb_1)
        
        concat_emb_1 = concatenate([n2v_emb_1, glove_emb_1])
        MLP_layer_1_1 = Dense(196, use_bias=False, name="mlp11")(concat_emb_1)
        MLP_layer_1_1 = BatchNormalization(name="batchnorm11")(MLP_layer_1_1)
        MLP_layer_1_1 = PReLU(name="prelu11")(MLP_layer_1_1)
        MLP_layer_1_2 = Dense(128, use_bias=False,name="mlp12")(MLP_layer_1_1)
        MLP_layer_1_2 = BatchNormalization(name="batchnorm12")(MLP_layer_1_2)
        MLP_layer_1_2 = PReLU(name="mlp12")(MLP_layer_1_2)

        # second_concept_side
        n2v_input_2 = Input(shape=(1,))
        n2v_emb_layer_2 = Embedding(len(self.concept2id)+1, 128, input_length=1, name="n2vembedding_2", trainable=False) 
        n2v_emb_2 = n2v_emb_layer_2(n2v_input_2)
        n2v_emb_2 = Flatten()(n2v_emb_2)

        glove_input_2 = Input(shape=(1,))
        glove_emb_layer_2 = Embedding(len(self.concept2id)+1, 128, input_length=1, name="gloveembedding_2", trainable=False) 
        glove_emb_2 = glove_emb_layer_2(glove_input_2)
        glove_emb_2 = Flatten()(glove_emb_2)

        concat_emb_2 = concatenate([n2v_emb_2, glove_emb_2])
        MLP_layer_2_1 = Dense(196, use_bias=False, name="mlp21")(concat_emb_2)
        MLP_layer_2_1 = BatchNormalization(name="batchnorm21")(MLP_layer_2_1)
        MLP_layer_2_1 = PReLU(name="prelu21")(MLP_layer_2_1)
        MLP_layer_2_2 = Dense(128, name="mlp22")(MLP_layer_2_1)
        MLP_layer_2_2 = BatchNormalization(name="batchnorm22")(MLP_layer_2_2)
        MLP_layer_2_2 = PReLU(name="prelu22")(MLP_layer_2_2)

        # loss function to train
        dot_layer = dot([MLP_layer_1_2, MLP_layer_2_2], axes=1, normalize=True)
        output = Dense(1, kernel_initializer="random_uniform", activation="sigmoid")(dot_layer)

        self.model = Model(inputs=[n2v_input_1, glove_input_1, n2v_input_2, glove_input_2], outputs=output)
        n2v_emb_layer_1.set_weights(self.weights_n2v)
        n2v_emb_layer_2.set_weights(self.weights_n2v)
        glove_emb_layer_1.set_weights(self.weights_glove)
        glove_emb_layer_2.set_weights(self.weights_glove)

        # make it a multi-gpu model.
        self.model = multi_gpu_model(self.model, gpus=self.num_gpus)
        adam_my = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model.compile(loss="binary_crossentropy", optimizer=adam_my)


class EnhanceModelBeta(BaseModel):
    '''
    You can implement your own models here.
    '''
    pass