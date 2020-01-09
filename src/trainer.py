# put training here.
import os
from keras.callbacks import ModelCheckpoint
from data_loader import generate_pairs
from data_loader import load_dictionary

class BaseTrainer(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data # data is a dir to load batches.
        self.config = config
        self.concept2id = load_dictionary(config.dictionary.concept2id_dictionary)

    def train(self):
        raise NotImplementedError

class EnhancedModelTrainer(BaseTrainer):
    def __init__(self, model, data, config):
        super(EnhancedModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()
    
    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )
    
    def train(self):
        history = self.model.fit_generator(
            generate_pairs(self.data, batch_size=self.config.trainer.batch_size, 
            concept_dictionary=self.concept2id), 
            steps_per_epoch=self.config.trainer.steps_per_epoch, 
            epochs=self.config.trainer.epoch, 
            callbacks=self.callbacks)
            
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
