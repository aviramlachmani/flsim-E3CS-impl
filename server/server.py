import client
import load_data
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread
import torch
import utils.dists as dists  # pylint: disable=no-name-in-module
import math


def random_n(b1, b2, b3):
    rand_list = []
    out = [0, 0, 0, 0]
    for i in range(20):
        rand_list.append(random.randint(1, 100))
    for rand in rand_list:
        if rand <= b1:
            out[0] += 1
        elif b1 < rand <= b2:
            out[1] += 1
        elif b2 < rand <= b3:
            out[2] += 1
        else:
            out[3] += 1
    return out


def random_sim(sample_clients):
    ans = []
    out = random_n(25, 50, 75)
    pick = np.random.binomial(size=out[0], n=1, p=0.1)
    pick = np.append(pick, np.random.binomial(size=out[1], n=1, p=0.3))
    pick = np.append(pick, np.random.binomial(size=out[2], n=1, p=0.6))
    pick = np.append(pick, np.random.binomial(size=out[3], n=1, p=0.9))
    for i in range(len(pick)):
        if pick[i] == 1:
            ans.append(sample_clients[i])
    return ans


def fedcs_sim(sample_clients):
    ans = []
    pick = np.random.binomial(size=20, n=1, p=0.9)
    for i in range(len(pick)):
        if pick[i] == 1:
            ans.append(sample_clients[i])
    return ans


def pow_d_sim(clients_per_round, class_a, class_b, class_c, class_d):
    out = random_n(53, 76, 90)
    pick_a = np.random.binomial(size=out[0], n=1, p=0.1)
    pick_a = sum(pick_a)
    ans_a = [client for client in random.sample(class_a, pick_a)]
    pick_b = np.random.binomial(size=out[1], n=1, p=0.3)
    pick_b = sum(pick_b)
    ans_b = [client for client in random.sample(class_b, pick_b)]
    pick_c = np.random.binomial(size=out[2], n=1, p=0.6)
    pick_c = sum(pick_c)
    ans_c = [client for client in random.sample(class_c, pick_c)]
    pick_d = np.random.binomial(size=out[3], n=1, p=0.9)
    pick_d = sum(pick_d)
    ans_d = [client for client in random.sample(class_d, pick_d)]
    ans = ans_a + ans_b + ans_c + ans_d
    return ans


def _make_class(size, p):
    '''
    :param size: size og returned group
    :param p: probability of binomial distribution
    :return: array of size 'size', with 0/1 values according to p
    '''
    return np.random.binomial(size=size, n=1, p=p)


def _create_clients_group(K=100, groups=4):
    '''
    to simulate the 'heterogeneous volatility' the autors of the article divide the
    whole set of clients into 4 classes, with the success rate respectively
    set as 0.1, 0.3, 0.6, and 0.9.
    :param K: number of total active clients
    :param groups: number of different groups (with different binomial distribution of success rate
    :return: group of size K, with equally divided clients into 4 classes
    '''
    Xt = []
    group_size = int(K / groups)
    Xt = np.concatenate((_make_class(group_size, 0.1), _make_class(group_size, 0.3)))
    Xt = np.concatenate((Xt, _make_class(group_size, 0.6)))
    Xt = np.concatenate((Xt, _make_class(group_size, 0.9)))
    return Xt


def _num_sigma(T, s_type, num=1, k=20, K=100):
    def _sigma_t(t):
        return (num * k / K)

    def _inc_sigma_t(t):
        if t < (T / 4):
            return 0
        else:
            return k / K

    if s_type == "num":
        return _sigma_t
    else:
        return _inc_sigma_t


def ProbAlloc(k, sigma_t, W_t, K=100):
    '''
    :param k: the number of involved clients in each round
    :param sigma_t: fairness quota of round t
    :param W_t: exponential weights for round (vector of size K)
    :param K: total num of activate clients
    :return: - Pt: probability allocation vector for round t
             - St: overflowed set for round t
    '''
    St = []
    P_t = np.zeros(len(W_t))
    for i in range(0, len(W_t)):
        P_t[i] = sigma_t + (((k - (K * sigma_t)) * W_t[i]) / sum(W_t))
        if P_t[i] > 1:
            P_t[i] = 1
            St.append(i)

    return P_t, St


def E3CS_FL_algorithm(k, sigma_t, W_t, x_t, K=100, eta=0.5):
    '''
    :param k: the number of involved clients in each round
    :param K: accessible clients in the system
    :param eta: the learning rate of weights update
    :return: At: the selected group in round t
    '''
    At = np.zeros(k)
    Pt, St = ([] for i in range(2))

    # x_t = _create_clients_group(K)  # the success status of a client in each round
    Pt, St = ProbAlloc(k, sigma_t, W_t, K)
    Pt_tensor = torch.tensor(Pt)
    At = torch.multinomial(Pt_tensor, k, replacement=False)
    x_estimator_t = np.zeros(K)
    for i in range(0, K):
        x_estimator_t[i] = x_t[i] / Pt[i] if Pt[i] > 0.014 else x_t[i] / 0.014
        W_t[i] = W_t[i] if (i in St) else W_t[i] * math.exp((k - (K * sigma_t)) * eta * x_estimator_t[i] / K)
    return At, W_t


class Server(object):
    """Basic federated learning server."""

    def __init__(self, config):
        self.config = config

    # Set up server
    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total

        # Add fl_model to import path
        sys.path.append(model_path)

        # Set up simulated server
        self.load_data()
        self.load_model()
        self.make_clients(total_clients)

    def load_data(self):
        import fl_model  # pylint: disable=import-error

        # Extract config for loaders
        config = self.config

        # Set up data generator
        generator = fl_model.Generator()

        # Generate data
        data_path = self.config.paths.data
        data = generator.generate(data_path)
        labels = generator.labels

        logging.info('Dataset size: {}'.format(
            sum([len(x) for x in [data[label] for label in labels]])))
        logging.debug('Labels ({}): {}'.format(
            len(labels), labels))

        # Set up data loader
        self.loader = {
            'basic': load_data.Loader(config, generator),
            'bias': load_data.BiasLoader(config, generator),
            'shard': load_data.ShardLoader(config, generator)
        }[self.config.loader]

        logging.info('Loader: {}, IID: {}'.format(
            self.config.loader, self.config.data.IID))

    def load_model(self):
        import fl_model  # pylint: disable=import-error

        model_path = self.config.paths.model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.save_model(self.model, model_path)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    def make_clients(self, num_clients):
        IID = self.config.data.IID
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading

        if not IID:  # Create distribution for label preferences if non-IID
            dist = {
                "uniform": dists.uniform(num_clients, len(labels)),
                "normal": dists.normal(num_clients, len(labels))
            }[self.config.clients.label_distribution]
            random.shuffle(dist)  # Shuffle distribution

        # Make simulated clients
        clients = []
        for client_id in range(num_clients):

            # Create new client
            new_client = client.Client(client_id)

            if not IID:  # Configure clients for non-IID data
                if self.config.data.bias:
                    # Bias data partitions
                    bias = self.config.data.bias
                    # Choose weighted random preference
                    pref = random.choices(labels, dist)[0]

                    # Assign preference, bias config
                    new_client.set_bias(pref, bias)
                elif self.config.data.shard:
                    # Shard data partitions
                    shard = self.config.data.shard

                    # Assign shard config
                    new_client.set_shard(shard)

            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))

        if loader == 'bias':
            logging.info('Label distribution: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if loading == 'static':
            if loader == 'shard':  # Create data shards
                self.loader.create_shards()

            # Send data partition to all clients
            [self.set_client_data(client) for client in clients]

        self.clients = clients

    # Run federated learning
    def run(self):
        name_file = "output_emnist_E3CS_05_non_iid_p.txt"
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports
        f = open(name_file, 'w')
        f.write(self.config.model + "  " + self.config.method + "  \n")
        f.close()

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Perform rounds of federated learning
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Run the federated learning round
            accuracy = self.round(round)
            f = open(name_file, 'a')
            f.write("round: " + str(round) + "    accuracy: " + str(accuracy * 100) + " \n")
            f.close()

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                f.write("done this test\n")
                f.close()
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

    def round(self, round):
        import fl_model  # pylint: disable=import-error
        if self.config.method == "E3CS_0" or self.config.method == "E3CS_05" or self.config.method == "E3CS_08" or self.config.method == "E3CS_inc":
            if round == 1:
                W_t_old = np.ones(100)  # change according to weights of round t
            else:
                W_t_old = self.W_t
        # Select clients to participate in the round
        if self.config.method == "E3CS_0" or self.config.method == "E3CS_05" or self.config.method == "E3CS_08" or self.config.method == "E3CS_inc":
            sample_clients, W_t, A_t, sample_index = self.selection(round, W_t_old)
            self.W_t = W_t
        else:
            sample_clients = self.selection(round, w_t_up=0)

        # Configure sample clients
        self.configuration(sample_clients)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client updates
        reports = self.reporting(sample_clients)

        # Perform weight aggregation
        logging.info('Aggregating updates')
        if self.config.method == "E3CS_0" or self.config.method == "E3CS_05" or self.config.method == "E3CS_08" or self.config.method == "E3CS_inc":
            updated_weights = self.aggregation(reports, W_t_old, A_t, sample_index)
        else:
            updated_weights = self.aggregation(reports, 0, A_t=0, sample_index=0)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy

    # Federated learning phases

    def selection(self, t, w_t_up):
        clients_per_round = self.config.clients.per_round

        if self.config.method == "random":

            # Select devices to participate in round

            # Select clients randomly
            sample_clients = [client for client in random.sample(
                self.clients, clients_per_round)]

            return random_sim(sample_clients)

        elif self.config.method == "FedCS":
            class_d = []
            for i in range(25):
                class_d.append(self.clients[i])
            sample_clients = [client for client in random.sample(class_d, clients_per_round)]

            return fedcs_sim(sample_clients)

        elif self.config.method == "pow-d":
            class_a = []
            class_b = []
            class_c = []
            class_d = []
            for i in range(25):
                class_a.append(self.clients[i])
                class_b.append(self.clients[i + 25])
                class_c.append(self.clients[i + 50])
                class_d.append(self.clients[i + 75])
            return pow_d_sim(clients_per_round, class_a, class_b, class_c, class_d)

        elif self.config.method == "E3CS_0":
            k = 20  # num of selected clients in each round
            K = 100  # num of total activated clients
            T = 2500  # num of total rounds
            sample_client = []
            sample_index = []
            Xt = _create_clients_group(K)  # the success status of a client in each round
            sigma_t = (_num_sigma(T, s_type="num", num=0))(t)  # the minimum selection probability of each client
            At, Wt = E3CS_FL_algorithm(k=k, sigma_t=sigma_t, x_t=Xt, W_t=w_t_up, K=K, eta=0.5)
            At = At.detach().numpy()
            for i in range(k):
                if Xt[At[i]] == 1:
                    sample_client.append(self.clients[int(At[i])])
                    sample_index.append(int(At[i]))

            return sample_client, Wt, At, sample_index

        elif self.config.method == "E3CS_05":
            k = 20  # num of selected clients in each round
            K = 100  # num of total activated clients
            T = 2500  # num of total rounds
            sample_client = []
            sample_index = []
            Xt = _create_clients_group(K)  # the success status of a client in each round
            sigma_t = (_num_sigma(T, s_type="num", num=0.5))(t)  # the minimum selection probability of each client
            At, Wt = E3CS_FL_algorithm(k=k, sigma_t=sigma_t, x_t=Xt, W_t=w_t_up, K=K, eta=0.5)
            At = At.detach().numpy()
            for i in range(k):
                if Xt[At[i]] == 1:
                    sample_client.append(self.clients[int(At[i])])
                    sample_index.append(int(At[i]))

            return sample_client, Wt, At, sample_index


        elif self.config.method == "E3CS_08":
            k = 20  # num of selected clients in each round
            K = 100  # num of total activated clients
            T = 2500  # num of total rounds
            sample_client = []
            sample_index = []
            Xt = _create_clients_group(K)  # the success status of a client in each round
            sigma_t = (_num_sigma(T, s_type="num", num=0.8))(t)  # the minimum selection probability of each client
            At, Wt = E3CS_FL_algorithm(k=k, sigma_t=sigma_t, x_t=Xt, W_t=w_t_up, K=K, eta=0.5)
            At = At.detach().numpy()
            for i in range(k):
                if Xt[At[i]] == 1:
                    sample_client.append(self.clients[int(At[i])])
                    sample_index.append(int(At[i]))

            return sample_client, Wt, At, sample_index

        elif self.config.method == "E3CS_inc":
            k = 20  # num of selected clients in each round
            K = 100  # num of total activated clients
            T = 2500  # num of total rounds
            sample_client = []
            sample_index = []
            Xt = _create_clients_group(K)  # the success status of a client in each round
            sigma_t = (_num_sigma(T, s_type="inc", num=1))(t)  # the minimum selection probability of each client
            At, Wt = E3CS_FL_algorithm(k=k, sigma_t=sigma_t, x_t=Xt, W_t=w_t_up, K=K, eta=0.5)
            At = At.detach().numpy()
            for i in range(k):
                if Xt[At[i]] == 1:
                    sample_client.append(self.clients[int(At[i])])
                    sample_index.append(int(At[i]))

            return sample_client, Wt, At, sample_index

    def configuration(self, sample_clients):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuraion on client
            client.configure(config)

    def reporting(self, sample_clients):
        # Recieve reports from sample clients
        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports

    def aggregation(self, reports, updated_W, A_t, sample_index):
        return self.federated_averaging(reports, updated_W, A_t, sample_index)

    # Report aggregation
    def extract_client_updates(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                if self.config.method == "E3CS_0" or self.config.method == "E3CS_05" or self.config.method == "E3CS_08" or self.config.method == "E3CS_inc":
                    delta = weight
                else:
                    delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def federated_averaging(self, reports, W_t, A_t, sample_index):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        updates = self.extract_client_updates(reports)


        # Extract total number of samples
        if self.config.method == "E3CS_0" or self.config.method == "E3CS_05" or self.config.method == "E3CS_08" or self.config.method == "E3CS_inc":
            total_W = sum(W_t)
        else:
            total_samples = sum([report.num_samples for report in reports])

        if len(updates) != 0:
            # Perform weighted averaging
            avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
                          for _, x in updates[0]]

            if self.config.method == "E3CS_0" or self.config.method == "E3CS_05" or self.config.method == "E3CS_08" or self.config.method == "E3CS_inc":
                for i, update in enumerate(updates):
                    num_samples = reports[i].num_samples
                    for j, (_, delta) in enumerate(update):
                        # Use weighted average by number of samples
                        avg_update[j] += delta * (W_t[sample_index[i]] / total_W)

            else:
                for i, update in enumerate(updates):
                    num_samples = reports[i].num_samples
                    for j, (_, delta) in enumerate(update):
                        # Use weighted average by number of samples
                        avg_update[j] += delta * (num_samples / total_samples)

        # Extract baseline model weights
        else:
            baseline_weights = fl_model.extract_weights(self.model)
            avg_update = []
            for i, _ in enumerate(baseline_weights):
                avg_update.append(0)

        baseline_weights = fl_model.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        if self.config.method == "E3CS_0" or self.config.method == "E3CS_05" or self.config.method == "E3CS_08" or self.config.method == "E3CS_inc":
            for i, (name, weight) in enumerate(baseline_weights):
                ne_weight = 0
                for j in range(100):
                    if not (j in sample_index):
                        ne_weight += weight * (W_t[j] / total_W)
                updated_weights.append((name, ne_weight + avg_update[i]))
        else:
            for i, (name, weight) in enumerate(baseline_weights):
                updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def accuracy_averaging(self, reports):
        # Get total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        accuracy = 0
        for report in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def set_client_data(self, client):
        loader = self.config.loader

        # Get data partition size
        if loader != 'shard':
            if self.config.data.partition.get('size'):
                partition_size = self.config.data.partition.get('size')
            elif self.config.data.partition.get('range'):
                start, stop = self.config.data.partition.get('range')
                partition_size = random.randint(start, stop)

        # Extract data partition for client
        if loader == 'basic':
            data = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            data = self.loader.get_partition(partition_size, client.pref)
        elif loader == 'shard':
            data = self.loader.get_partition()
        else:
            logging.critical('Unknown data loader type')

        # Send data to client
        client.set_data(data, self.config)

    def save_model(self, model, path):
        path += '/global'
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def save_reports(self, round, reports):
        import fl_model  # pylint: disable=import-error

        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
                report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            fl_model.extract_weights(self.model))
