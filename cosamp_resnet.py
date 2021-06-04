import FIHT
import ffht
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import datetime as dt
import numpy as np
import math
from mpi4py import MPI
import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import sys
from models import *
from sampler import *

# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
N = nproc - 1 # one node is the server
rank = comm.Get_rank()

x_train_local = np.zeros((1,))
y_train_local = np.zeros((1,))   
x_test_local = np.zeros((1,))   
y_test_local = np.zeros((1,))   

optimizer = tf.keras.optimizers.SGD()

## EXPERIMENT CONFIG ###########################
num_epoch = 70
alpha = 0.01 # learning rate

batch_size= 8

comp = 2
# model_size = 4912010
model_size = 668426
k = int(model_size/(12*comp))

base_model = 'FLNet'
fbk_status = 'fbk'
num_groups_per_node = 1
sample = 'iid'
################################################

# Loss metric
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
sparsity = tf.keras.metrics.Mean('sparsity', dtype=tf.float32)

# Accuracy metric
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

# Compressibility metric
sp_p = tf.keras.metrics.Mean('sp_p', dtype=tf.float32)

if rank == 0:
    
    if sample == "non_iid2":
        sampler = Sampler(False, N, "cifar10", var=2)
    elif sample == "non_iid0":
        sampler = Sampler(False, N, "cifar10", var=0)
        sampler.sample_noniid(0, comm)
    else:
        sampler = Sampler(True, N, "cifar10")
        sampler.sample_iid(comm)
    
    # Instantiate Model
    model = flnet_init()
    
    # Set compressibility writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    c_log_dir = 'logs/'+base_model+'/'+sample+str(num_groups_per_node)+'/'+fbk_status+'/fihtwht_k'+str(k)+'_'+str(rank)+'_alpha' + str(alpha) + '/' + current_time + '/sparse_pt'
    c_summary_writer = tf.summary.create_file_writer(c_log_dir)

    # Augment d
    d = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    print("d", d)
    d_aug = 2 ** np.ceil(np.log2(d)).astype('int32')

    # Generate row indices of \Phi
    Q = k * 12
    rng = np.random.default_rng(2)
    Phi_row_idx = rng.choice(d_aug, Q, replace=False)

    # Send row indices to clients
    for n in range(1, N+1):
        comm.send(Phi_row_idx, dest=n, tag=11)

    r = np.zeros((0,))
    e = np.zeros((0,))
    for epoch in range(num_epoch):
        if sample == "non_iid2":
            sampler.sample_noniid(2, comm)

        # dummy dataset to get number of steps
        dset = tf.data.Dataset.from_tensor_slices((sampler.x_train_local, sampler.y_train_local))
        dset = dset.shuffle(buffer_size=60000).batch(batch_size)
         
        for step, (dummy_x, dummy_y) in enumerate(dset):
            # Aggregation
            # obtain y from each node (assuming at least one other node)
            z = comm.recv(source=1, tag=11)
            grad = comm.recv(source=1, tag=11)
            for n in range(2,N+1):
                y = comm.recv(source=n, tag=11)
                grad_n = comm.recv(source=n, tag=11)
                z = np.array(z) + np.array(y)
                grad = np.array(grad) + np.array(grad_n)

            # Ramp up alpha
            if epoch <= 5:
              alpha_t = alpha + epoch*(0.3-alpha)/5
            elif epoch > 5 and epoch <= 10:
              alpha_t = 0.3 - (epoch-5)*(0.3-alpha)/5
            else:
              alpha_t = alpha

            z = alpha * (1.0/N) * z
            grad = alpha * (1.0/N) * grad

            if step == 0 and epoch == 0:
                # init r
                r = np.zeros(z.shape)
                e = np.zeros(grad.shape)
            
            if fbk_status == 'fbk':
                z += r
                grad += e
            
            # Compute sparsity of grad
            sparsity_grad = (np.linalg.norm(grad, ord=1)**2)/(np.linalg.norm(grad, ord=2)**2 * grad.shape[0])
            sp_p(sparsity_grad)

            # Recover K-sparse signal
            g_rec = FIHT.FastIHT_WHT(z, k, Q, d_aug, Phi_row_idx, top_k_func=1)[0:d]
            
            e = grad - g_rec

            g_rec_wht = np.concatenate((g_rec, np.zeros(d_aug-d))) / np.sqrt(Q)            
            ffht.fht(g_rec_wht)
            z_rec = g_rec_wht[Phi_row_idx]

            r = z - z_rec

            # Unflatten g_rec_wht
            shapes = [model.trainable_weights[i].shape for i in range(len(model.trainable_weights))]

            grad_tx = []
            n_prev = 0
            for i in range(len(shapes)):
                n = n_prev + tf.math.reduce_prod(shapes[i])
                grad_tx.append(tf.cast(tf.reshape(g_rec[n_prev:n], shapes[i]), tf.float32))
                n_prev = n
                
            # Send g_rec_wht back to clients
            for n in range(1, N+1):
                comm.send(grad_tx, dest=n, tag=11)

        with c_summary_writer.as_default():
          tf.summary.scalar("Sparsity(p(t))", sp_p.result(), step=epoch)
          sp_p.reset_states()
            
else:
    if sample == "iid" or sample == "non_iid0":
        # Receive partitioned data at node
        x_train_local, y_train_local, x_test_local, y_test_local = comm.recv(source=0, tag=11)
        x_test_local = tf.keras.applications.resnet.preprocess_input(x_test_local)    
        x_train_local = tf.keras.applications.resnet.preprocess_input(x_train_local)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_local, y_train_local))
        train_dataset = train_dataset.shuffle(buffer_size=60000).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test_local, y_test_local))
        val_dataset = val_dataset.shuffle(buffer_size=60000).batch(batch_size)
    
    # Instantiate Model
    model = flnet_init()

    # Unfreeze Batch Norm layers                                                                              
    for layer in model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True

    # Set up summary writers
    if rank == 2:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/'+base_model+'/'+sample+str(num_groups_per_node)+'/'+fbk_status+'/fihtwht_k'+str(k)+'_b'+str(batch_size)+"_"+str(rank)+'_alpha'+'/' + current_time + '/train'
        test_log_dir = 'logs/'+base_model+'/'+sample+str(num_groups_per_node)+'/'+fbk_status+'/fihtwht_k'+str(k)+"_"+str(rank)+'_alpha'+'/' + current_time + '/test'
        sparse_log_dir = 'logs/'+base_model+'/'+sample+str(num_groups_per_node)+'/'+fbk_status+'/fihtwht_k'+str(k)+"_"+str(rank)+'_alpha'+'/' + current_time + '/sparse'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        sparse_summary_writer = tf.summary.create_file_writer(sparse_log_dir)

    # Receive \Phi row indices from server
    Phi_row_idx = comm.recv(source=0, tag=11)
    Q = k * 12

    for epoch in range(num_epoch):
        print(f"\nStart of Training Epoch {epoch}")
        if sample == "non_iid2":
            # Receive dataset for this epoch
            x_train_local, y_train_local, x_test_local, y_test_local = comm.recv(source=0, tag=11)
            x_test_local = tf.keras.applications.resnet.preprocess_input(x_test_local)    
            x_train_local = tf.keras.applications.resnet.preprocess_input(x_train_local)

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train_local, y_train_local))
            train_dataset = train_dataset.batch(batch_size)

            val_dataset = tf.data.Dataset.from_tensor_slices((x_test_local, y_test_local))
            val_dataset = val_dataset.shuffle(buffer_size=60000).batch(batch_size)

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                loss = loss_fn(y_batch_train, y_pred)
                train_accuracy(y_batch_train, y_pred)
                train_loss(loss)
                #print("Step", step, loss)
                
                grad = tape.gradient(loss, model.trainable_weights)

                # Flatten gradients
                concat_grads = tf.zeros((0,), dtype=tf.dtypes.float32)

                for i in range(len(grad)):
                    flattened = tf.reshape(grad[i], -1) #flatten
                    concat_grads = tf.concat((concat_grads, flattened), 0)
                
                # Compute sparsity metric
                s = (tf.norm(concat_grads, ord=1)**2)/(tf.norm(concat_grads, ord=2)**2 * concat_grads.shape[0])
                sparsity(s)

                # Compute y (WHT)
                # Augment d
                d = concat_grads.shape[0]
                d_aug = 2 ** np.ceil(np.log2(d)).astype('int32')
                
                g_wht = np.concatenate((concat_grads, np.zeros(d_aug-d))) / np.sqrt(Q)
                ffht.fht(g_wht)
                y = g_wht[Phi_row_idx]
                
                # Send y to server
                comm.send(y, dest=0, tag=11)
                comm.send(concat_grads, dest=0, tag=11)

            ## NOT WORKING: Receive and set weights from server
            #weights = comm.recv(source=0, tag=11)
            #model.set_weights(weights)
            
            ## WORK AROUND: Receive and apply gradients
            grad_rx = comm.recv(source=0, tag=11)
            optimizer.learning_rate = 1 # alpha already applied server-side
            #optimizer.momentum = 0.9
            optimizer.apply_gradients(zip(grad_rx, model.trainable_weights))

        print(f"Accuracy over epoch {train_accuracy.result()}")
        print(f"Loss over epoch {train_loss.result()}")
        
        if rank == 2:
            # Log train metrics
            with train_summary_writer.as_default():
              tf.summary.scalar('loss', train_loss.result(), step=epoch)
              tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
        #train_acc = train_accuracy.result()
        #print(f"Accuracy over epoch {train_acc}")

            # Reset metrics every epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
                
            # Run validation
            for x_batch_val, y_batch_val in val_dataset:
                val_logits = model(x_batch_val, training = False)
                val_loss(loss_fn(y_batch_val, val_logits))
                val_accuracy(y_batch_val, val_logits)

            # Log Validation metrics
            with test_summary_writer.as_default():
                tf.summary.scalar('val_loss', val_loss.result(), step=epoch)
                tf.summary.scalar('val_accuracy', val_accuracy.result(), step=epoch)

            print("Validation acc: %.4f" % (float(val_accuracy.result()),))
            
            with sparse_summary_writer.as_default():
                tf.summary.scalar('sparsity', sparsity.result(), step=epoch)

        val_loss.reset_states()
        val_accuracy.reset_states()
        sparsity.reset_states()

