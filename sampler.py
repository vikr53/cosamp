import tensorflow as tf
import numpy as np
import math
import sys

class Sampler:
    def __init__(self, iid, num_nodes, dataset, comm, num_groups_per_node=1):
        if dataset == "cifar10":
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
            self.x_train, self.x_test = tf.cast(self.x_train, tf.float32),  tf.cast(self.x_test, tf.float32)
            self.train_size, self.test_size = self.x_train.shape[0], self.x_test.shape[0]
            self.num_classes = 10
        else:
            print("Invalid Dataset")
            sys.exit()

        self.num_groups_per_node = 1
        self.N = num_nodes
        self.comm = comm
        
        if not iid:
            self.num_clients_total = 10000

            # Split dataset into groups
            self.group = {i: [] for i in range(self.num_classes)}
            for i in range(self.train_size):
                self.group[self.y_train[i][0]].append(self.x_train[i])

            for i in range(len(self.group)):
                self.group[i] = np.array(self.group[i])

            n_w_group = self.num_clients_total // 10
            split_train_idx = np.random.choice(len(self.group[0]), (n_w_group, math.floor(len(self.group[0])/n_w_group)), replace=False)

            # Form large dataset with all groups [(10000, 5)]
            self.final_x_dataset = []
            self.final_y_dataset = []

            for g in self.group:
                for j in range(n_w_group):
                    self.final_x_dataset.append(self.group[g][split_train_idx[j]].tolist())
                    self.final_y_dataset.append([g for i in range(len(split_train_idx[j]))])
                    
            self.final_x_dataset = np.array(self.final_x_dataset)
            self.final_y_dataset = np.array(self.final_y_dataset)
            
            # print(self.final_x_dataset.shape)
            #self.final_x_dataset = np.reshape(np.fromfile("noniid_xclient_data.dat"), (10000, 5, 32, 32, 3))
            #self.final_y_dataset = np.reshape(np.fromfile("noniid_yclient_data.dat"), (10000, 5))
            
            #pd.DataFrame(self.final_x_dataset).to_csv("noniid_client_data.csv")
            #self.final_x_dataset.tofile("noniid_xclient_data.dat")
            #self.final_y_dataset.tofile("noniid_yclient_data.dat")

    def sample_iid(self):
        ##### IID #####
        # Partition training data
        split_train_idx = np.random.choice(self.train_size, (N, math.floor(self.train_size/N)), replace=False)

        # Communicate data partition (point-to-point)
        for n in range(1, self.N+1):
            self.x_train_local = np.array([self.x_train[idx] for idx in split_train_idx[n-1]])
            self.y_train_local = np.array([self.y_train[idx] for idx in split_train_idx[n-1]])

            self.comm.send([self.x_train_local, self.y_train_local, self.x_test, self.y_test], dest=n, tag=11)

    def sample_noniid(self):
        ###### NON-IID [100 clients sampled from 10,000 total (with 5 images of 1 class each)] #####
        for n in range(1, self.N+1):
            # Randomly choose 100 idx from (0,num_clients_total)
            idx = np.random.randint(0, self.num_clients_total, 100)
            self.x_train_local = np.reshape(self.final_x_dataset[idx], (500,32,32,3))
            self.y_train_local = np.reshape(self.final_y_dataset[idx], (500,1))

            print(self.x_train_local.shape)
            print(self.y_train_local.shape)
            
            self.comm.send([self.x_train_local, self.y_train_local, self.x_test, self.y_test], dest=n, tag=11)
