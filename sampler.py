class Sampler:
    def __init__(self, iid, num_nodes, dataset, num_groups_per_node=1):
        if dataset == "cifar10":
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
            self.x_train, self.x_test = tf.cast(x_train, tf.float32),  tf.cast(x_test, tf.float32)
            self.train_size, self.test_size = self.x_train.shape[0], self.x_test.shape[0]
            self.num_clasess = 10
        else:
            print("Invalid Dataset")
            sys.exit()

        self.num_groups_per_node = 1
        self.N = num_nodes
        
        if not iid:
            num_clients_total = 10000

            # Split dataset into groups
            group = {i: [] for i in range(self.num_classes)}
            for i in range(self.train_size):
                group[self.y_train[i][0]].append(self.x_train[i])

            n_w_group = num_clients_total // 10
            split_train_idx = np.random.choice(len(group[0]), (n_w_group, math.floor(len(group[0])/n_w_group)), replace=False)

            # Form large dataset with all groups [(10000, 5)]
            self.final_x_dataset = np.zeros((0,))
            self.final_y_dataset = np.zeros((0,))

    def sample_iid():
        ##### IID #####
        # Partition training data
        split_train_idx = np.random.choice(self.train_size, (N, math.floor(self.train_size/N)), replace=False)

        # Communicate data partition (point-to-point)
        for n in range(1, self.N+1):
            self.x_train_local = np.array([self.x_train[idx] for idx in split_train_idx[n-1]])
            self.y_train_local = np.array([self.y_train[idx] for idx in split_train_idx[n-1]])

            comm.send([self.x_train_local, self.y_train_local, self.x_test, self.y_test], dest=n, tag=11)

    def sample_noniid():
        ###### NON-IID [100 clients sampled from 10,000 total (with 5 images of 1 class each)] #####
        for g in groups:
            np.concatenate((self.final_x_dataset, np.array(g[split_train_idx])))
            np.concatenate((self.final_y_dataset, g*np.ones((len(split_train_idx,)))))

        for n in range(1, self.N+1):
            # Randomly choose 100 idx from (0,num_clients_total)
            idx = np.random.randint(0, num_clients_total, 100)
            self.x_train_local = self.final_x_dataset[idx]
            self.y_train_local = self.final_y_dataset[idx]
            
            comm.send([self.x_train_local, self.y_train_local, self.x_test, self.y_test], dest=n, tag=11)
