import tensorflow as tf
from src.softmax_model import EnhancingNet

def model_train(model, config):
    
    # train_data = load_data(config.dir)
    # need data_load functions
    for epoch in range(config.num_epochs):
        loss_avg = tf.keras.metrics.Mean()

        for x_train in train_dataset: 
            loss, gradients = compute_gradients(model, x_train)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            loss_avg(loss) 
            print("Step {}: Loss: {:.4f}".format(model.optimizer.iterations.numpy(), loss))
        
        if epoch % 1 == 0:
            avg_loss = loss_avg.result()
            model.epoch_loss_avg.append(avg_loss)
            model.save_weights(os.path.join(config.save_dir, "e{:03d}_loss{:.4f}.ckpt".format(epoch, avg_loss)))
            print("Epoch {}: Loss: {:.4f}".format(epoch, loss_avg.result()))

if __name__ == "__main__":
