from keras.models import Model
import os

model = ...  # include here your original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)


class EnhancedModelGenerator(object):
    def __init__(self,config):
        self.model = os.path.join(self.config.callbacks.checkpoint_dir, 
        '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name)
        self.model = Model(inputs=[n2v_input_1, patient_input_1, n2v_input_2, patient_input_2], outputs=output)
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(data)


    
    

    