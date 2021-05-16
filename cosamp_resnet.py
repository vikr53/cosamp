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

def iid_sampler(dataset):
  # Load data
  if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  else:
    print("Invalid Dataset")
    sys.exit()
    
  # Normalize input images
  x_train, x_test = tf.cast(x_train, tf.float32),  tf.cast(x_test, tf.float32)

  train_size = x_train.shape[0] # number of training samples
  test_size = x_test.shape[0] # number of testing samples

  # Partition training data
  split_train_idx = np.random.choice(train_size, (N, math.floor(train_size/N)), replace=False)

  d_xtrain = np.array((len(split_train_idx[0]),))
  d_ytrain = np.array((len(split_train_idx[0]),))
    
  # Communicate data partition (point-to-point)
  for n in range(1,N+1):
    x_train_local = np.array([x_train[idx] for idx in split_train_idx[n-1]])
    y_train_local = np.array([y_train[idx] for idx in split_train_idx[n-1]])
        
    if n == 1:
      d_xtrain = x_train_local
      d_ytrain = y_train_local

    comm.send([x_train_local, y_train_local, x_test, y_test], dest=n, tag=11)

  return d_xtrain, d_ytrain

def noniid_sampler(dataset, option=0, num_groups_per_node=2):
  # Load data
  num_classes = 0
  if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    num_classes = 10
  else:
    print("Invalid Dataset")
    sys.exit()

  train_size = x_train.shape[0] # number of training samples
  test_size = x_test.shape[0] # number of testing samples

  # Each node gets 1 class
  group = {i: [] for i in range(num_classes)}
  for i in range(train_size):
    group[y_train[i][0]].append(x_train[i])

  d_xtrain = np.array((len(group[0]),))
  d_ytrain = np.array((len(group[0]),))
  if option == 0:
    # Send groups to each node
    for n in range(1, N+1):
      x_train_local = np.array(group[n-1])
      y_train_local = (n-1)*np.ones((x_train_local.shape[0],))
      
      if n==1:
        d_xtrain = x_train_local
        d_ytrain = y_train_local

      print("x_train", x_train_local.shape)
      print("y_train",y_train_local.shape)

      comm.send([x_train_local, y_train_local, x_test, y_test], dest=n, tag=11)

    return d_xtrain, d_ytrain
  elif option == 1:
    # Send num_groups_per_node randomly chosen groups to each node
    for n in range(1, N+1):
      groups_n = np.random.randint(0, len(group), num_groups_per_node)
      print("groups_n", groups_n)
      print("group", len(group))
      x_train_local, y_train_local = [], []
      for i in range(num_groups_per_node):
        x_train_local += group[groups_n[i]]
        y_train_local += [groups_n[i] for _ in range(len(group[groups_n[i]]))]
        
      x_train_local, y_train_local = np.array(x_train_local), np.array(y_train_local)
      print("x_train", x_train_local.shape)
      print("y_train",y_train_local.shape)
      if n==1:
        d_xtrain = x_train_local
        d_ytrain = y_train_local

      comm.send([x_train_local, y_train_local, x_test, y_test], dest=n, tag=11)

    return d_xtrain, d_ytrain
  
  else:
    print("Invalid non-iid sampler option")
    sys.exit()
  
    

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
num_epoch = 150
alpha = 0.1 # learning rate

batch_size=32
k = 6000

base_model = 'flnet'
fbk_status = 'fbk'
num_groups_per_node = 1
sample = 'iid'
################################################

# Loss metric
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

# Accuracy metric
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

# Compressibility metric
rel_err = tf.keras.metrics.Mean('rel_err', dtype=tf.float32)

if rank == 0:
    
    if sample == "non_iid":
        d_xtrain, d_ytrain = noniid_sampler("cifar10",1, num_groups_per_node)
    else:
        d_xtrain, d_ytrain = iid_sampler("cifar10")
    # Instantiate model
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 8
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    # load weights
    model.load_weights("./chkpts/init_resnet8.ckpt")

    # dummy dataset to get number of steps
    dset = tf.data.Dataset.from_tensor_slices((d_xtrain, d_ytrain))
    dset = dset.shuffle(buffer_size=60000).batch(batch_size)
    
    # Set compressibility writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    c_log_dir = 'logs/'+base_model+'/'+sample+str(num_groups_per_node)+'/'+fbk_status+'/cosamp_k'+str(k)+'_'+str(rank)+'_alpha' + str(alpha) + '/' + current_time + '/compress'
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
    for epoch in range(num_epoch):
        for step, (dummy_x, dummy_y) in enumerate(dset):
            # Aggregation
            # obtain y from each node (assuming at least one other node)
            z = comm.recv(source=1, tag=11)
            for n in range(2,N+1):
                y = comm.recv(source=n, tag=11)
                z = np.array(z) + np.array(y)

            z = alpha * (1.0/N) * z

            if step == 0 and epoch == 0:
                # init r
                r = np.zeros(z.shape)
            
            if fbk_status == 'fbk':
                z += r

            # Recover K-sparse signal
            g_rec = FIHT.FastIHT_WHT(z, k, Q, d_aug, Phi_row_idx, top_k_func=1)[0:d]

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
          tf.summary.scalar("rel err", rel_err.result(), step=epoch)
          rel_err.reset_states()
            
else:
    # Receive partitioned data at node
    x_train_local, y_train_local, x_test_local, y_test_local = comm.recv(source=0, tag=11)
    x_test_local = tf.keras.applications.resnet.preprocess_input(x_test_local)    
    x_train_local = tf.keras.applications.resnet.preprocess_input(x_train_local)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_local, y_train_local))
    train_dataset = train_dataset.shuffle(buffer_size=60000).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test_local, y_test_local))
    val_dataset = val_dataset.shuffle(buffer_size=20000).batch(batch_size)
    
    # Instantiate model
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 8
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    
    # model load weights
    model.load_weights("./chkpts/init_resnet8.ckpt")

    # Unfreeze Batch Norm layers                                                                              
    for layer in model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True

    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/'+base_model+'/'+sample+str(num_groups_per_node)+'/'+fbk_status+'/cosamp_k'+str(k)+"_"+str(rank)+'_alpha'+str(alpha)+'/' + current_time + '/train'
    test_log_dir = 'logs/'+base_model+'/'+sample+str(num_groups_per_node)+'/'+fbk_status+'/cosamp_k'+str(k)+"_"+str(rank)+'_alpha_'+str(alpha)+'/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Receive \Phi row indices from server
    Phi_row_idx = comm.recv(source=0, tag=11)
    Q = k * 12

    for epoch in range(num_epoch):
        print(f"\nStart of Training Epoch {epoch}")
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

                # Compute y (WHT)
                # Augment d
                d = concat_grads.shape[0]
                d_aug = 2 ** np.ceil(np.log2(d)).astype('int32')
                
                g_wht = np.concatenate((concat_grads, np.zeros(d_aug-d))) / np.sqrt(Q)
                ffht.fht(g_wht)
                y = g_wht[Phi_row_idx]
                
                # Send y to server
                comm.send(y, dest=0, tag=11)

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
    
        val_loss.reset_states()
        val_accuracy.reset_states()
