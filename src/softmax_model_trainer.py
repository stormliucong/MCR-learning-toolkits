import tensorflow as tf
import numpy as np
import os
from utils.config import get_config_from_json
from src.softmax_model import EnhancingNet

def load_train_data(data_dir):
    return np.load(data_dir)

def get_permutations(idx, seq, i_vec, j_vec, p_vec):
    for first in seq:
        for second in seq:
            if first == 0:
                continue
            if second == 0:
                continue
            if first == second: 
                continue
            i_vec.append(first)
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

def model_train(model, json_dir):
    
    config = get_config_from_json(json_dir)
    train_data = load_train_data(config.dir)
    # need data_load functions
    batch_size = config.batch_size
    for epoch in range(config.num_epochs):
        
        # random shuffle train data
        total_batch = int(np.ceil(len(train_data) / batch_size))
        loss_avg = tf.keras.metrics.Mean()

        for i in range(total_batch):
            x_batch = train_data[i * batch_size : (i+1) * batch_size]
            p_vec, i_vec, j_vec = padMatrix(x_batch)
            
            loss, gradients = compute_gradients(model, x_batch, p_vec, i_vec, j_vec)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            loss_avg(loss) 
            print("Step {}: Loss: {:.4f}".format(model.optimizer.iterations.numpy(), loss))
        
        if epoch % 1 == 0:
            avg_loss = loss_avg.result()
            model.epoch_loss_avg.append(avg_loss)
            model.save_weights(os.path.join(config.save_dir, "e{:03d}_loss{:.4f}.ckpt".format(epoch, avg_loss)))
            print("Epoch {}: Loss: {:.4f}".format(epoch, loss_avg.result()))

@tf.function
def compute_loss(model, x_batch, p_vec, i_vec, j_vec):
    """
    --model: Enhancing model
    --x_batch: designated size of x
    --k: total number of concepts
    """
    model.compute_X(len(x_batch))
    model.compute_v(x_batch)
    matmul_vX = tf.linalg.normalize(
        tf.matmul(model.v, tf.transpose(model.X, [0,2,1])), axis=-1, ord=1) # n * l * k matrix
    denom_mat = tf.reduce_sum(matmul_vX, axis=-1) # n * l matrix

    nom_ids = tf.transpose([p_vec, i_vec, j_vec]) # length = n * l(l-1)
    denom_ids = tf.transpose([p_vec, i_vec]) # length = n * l(l-1)

    noms = tf.exp(tf.gather_nd(matmul_vX, nom_ids))
    denoms = tf.exp(tf.gather_nd(denom_mat, denom_ids))

    batch_loss = tf.math.reduce_sum(-tf.math.log(noms / denoms), axis=0) / len(x_batch)
    # batch training : take average

    return batch_loss

@tf.function
def compute_gradients(model, x_batch, p_vec, i_vec, j_vec):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x_batch, p_vec, i_vec, j_vec)
        
    return loss, tape.gradient(loss, model.trainable_variables)

if __name__ == "__main__":
    pass