import os
from keras.callbacks import ModelCheckpoint
<<<<<<< HEAD
<<<<<<< HEAD
from src.data_loader import generate_pairs
from src.data_loader import load_dictionary
=======
from src.data_loader import generate_pairs,load_dictionary
>>>>>>> a3c980c6c002f1448ce43c218205cce1a600df02
=======
from src.data_loader import generate_pairs,load_dictionary
>>>>>>> a3c980c6c002f1448ce43c218205cce1a600df02

class BaseTrainer(object):
    def __init__(self, model, config):
        self.model = model
        self.data = config.data.training_batch
        self.config = config
        self.concept2id = load_dictionary(config.dictionary.concept2id_dir)

    def train(self):
        raise NotImplementedError

class EnhancedModelTrainer(BaseTrainer):
    def __init__(self, model, config):
        super(EnhancedModelTrainer, self).__init__(model, config)
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
            generate_pairs(self.data, batch_size=self.config.trainer.batch_size), 
            steps_per_epoch=self.config.trainer.steps_per_epoch, 
            epochs=self.config.trainer.epoch, 
            callbacks=self.callbacks)
            
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
