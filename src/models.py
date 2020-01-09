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
from data_loader import load_dictionary, load_weight_matrix

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
        self.condition_model.save_weights(checkpoint_path)
        self.drug_model.save_weights(checkpoint_path)
        print("Model saved")

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
        self.weights_n2v = load_weight_matrix(config.data.pretrain_n2v) 
        self.weights_patient = load_weight_matrix(config.data.pretrain_n2v) 
        self.concept2id = load_dictionary(config.dictionary.concept2id_dictionary)
        self.num_gpus = config.settings.num_gpus
        self.build_model()

    def build_model(self):
        # first concept side
        n2v_input_1 = Input(shape=(1,))
        n2v_emb_layer_1 = Embedding(len(self.concept2id), 128, input_length=1, name="n2vembedding_1", trainable=False) 
        n2v_emb_1 = n2v_emb_layer_1(n2v_input_1)
        n2v_emb_1 = Flatten()(n2v_emb_1)

        patient_input_1 = Input(shape=(1,))
        patient_emb_layer_1 = Embedding(len(self.concept2id), 128, input_length=1, name="patientembedding_1", trainable=False) 
        patient_emb_1 = patient_emb_layer_1(patient_input_1)
        patient_emb_1 = Flatten()(patient_emb_1)
        
        concat_emb_1 = concatenate([n2v_emb_1, patient_emb_1])
        MLP_layer_1_1 = Dense(196, use_bias=False, name="MLP11")(concat_emb_1)
        MLP_layer_1_1 = BatchNormalization()(MLP_layer_1_1)
        MLP_layer_1_1 = PReLU()(MLP_layer_1_1)
        MLP_layer_1_2 = Dense(128, use_bias=False,name="MLP12")(MLP_layer_1_1)
        MLP_layer_1_2 = BatchNormalization()(MLP_layer_1_2)
        MLP_layer_1_2 = PReLU()(MLP_layer_1_2)
        self.condition_model = Model(inputs=[n2v_input_1, patient_input_1], outputs=MLP_layer_1_2,
        name = "condition_model")

        # second_concept_side
        n2v_input_2 = Input(shape=(1,))
        n2v_emb_layer_2 = Embedding(len(self.concept2id), 128, input_length=1, name="n2vembedding_2", trainable=False) 
        n2v_emb_2 = n2v_emb_layer_2(n2v_input_2)
        n2v_emb_2 = Flatten()(n2v_emb_2)

        patient_input_2 = Input(shape=(1,))
        patient_emb_layer_2 = Embedding(len(self.concept2id), 128, input_length=1, name="patientembedding_2", trainable=False) 
        patient_emb_2 = patient_emb_layer_2(patient_input_2)
        patient_emb_2 = Flatten()(patient_emb_2)

        concat_emb_2 = concatenate([n2v_emb_2, patient_emb_2])
        MLP_layer_2_1 = Dense(196, use_bias=False, name="MLP21")(concat_emb_2)
        MLP_layer_2_1 = BatchNormalization()(MLP_layer_2_1)
        MLP_layer_2_1 = PReLU()(MLP_layer_2_1)
        MLP_layer_2_2 = Dense(128, name="MLP22")(MLP_layer_2_1)
        MLP_layer_2_2 = BatchNormalization()(MLP_layer_2_2)
        MLP_layer_2_2 = PReLU()(MLP_layer_2_2)
        self.drug_model = Model(inputs=[n2v_input_2, patient_input_2], outputs=MLP_layer_2_2,
        name = "drug_model")

        # loss function to train
        dot_layer = dot([self.condition_model([n2v_input_1, patient_input_1]), 
        self.drug_model([n2v_input_2, patient_input_2])], axes=1, normalize=True)
        output = Dense(1, kernel_initializer="random_uniform", activation="sigmoid")(dot_layer)

        self.model = Model(inputs=[n2v_input_1, patient_input_1, n2v_input_2, patient_input_2], outputs=output)
        n2v_emb_layer_1.set_weights(self.weights_n2v)
        n2v_emb_layer_2.set_weights(self.weights_n2v)
        patient_emb_layer_1.set_weights(self.weights_patient)
        patient_emb_layer_2.set_weights(self.weights_patient)

        # make it a multi-gpu model.
        self.model = multi_gpu_model(self.model, gpus=self.num_gpus)
        adam_my = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model.compile(loss="binary_crossentropy", optimizer=adam_my)


class EnhanceModelBeta(BaseModel):
    '''
    You can implement your own models here.
    '''
    pass