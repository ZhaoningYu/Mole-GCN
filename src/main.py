import argparse
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import FileLoader
from utils.ops import Generator
import numpy as np
from tqdm import tqdm
from utils.model import GCN, norm

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-data', default='PTC', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=400, help='epochs')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-edge_weight', type=bool, default=True, help='If data have edge labels')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-w_d', type=float, default=0.0005, help='weight decay')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0., help='drop net')
    parser.add_argument('-drop_c', type=float, default=0., help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=float, default=[0.9, 0.8, 0.7])
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args


def gen_feature_adj_labels(g_graph, labels, num_cliques):
    number_of_cliques = num_cliques
    number_of_moles = len(labels)
    feature_c = np.eye(number_of_cliques, dtype=np.float32)
    feature_m = np.zeros((number_of_moles, number_of_cliques))
    for i in tqdm(range(1, number_of_moles+1),desc="feature matrix", unit="graph"):
        for k, dict in g_graph.adj["M{}".format(i)].items():
            feature_m[i-1][int(k[1:])-1] = 1

    features = np.r_[feature_c, feature_m]

    # Generate node labels
    y = np.empty(g_graph.number_of_nodes(), int)
    # print(len(y))
    for i in range(len(feature_c), len(y)):
        y[i] = labels[i - len(feature_c)]

    adj = np.zeros((g_graph.number_of_nodes(), g_graph.number_of_nodes()))
    for k, v in g_graph.adj.items():

        for item in v.keys():
            if k[0] == "M":
                if item[0] == "M":
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) + number_of_cliques - 1] = g_graph.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) + number_of_cliques - 1][int(item[1:]) - 1] = g_graph.get_edge_data(k, item)['weight']
            else:
                if item[0] == "M":
                    adj[int(k[1:]) - 1][int(item[1:]) + number_of_cliques - 1] = g_graph.get_edge_data(k, item)['weight']
                else:
                    adj[int(k[1:]) - 1][int(item[1:]) - 1] = g_graph.get_edge_data(k, item)['weight']
    adj = norm(adj)
    return features, adj, y


def accuracy(outputs, y):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == y).item() / len(preds)


def Average(lst):
    return sum(lst) / len(lst)


def train(features, adj, node_labels, labels, DEVICE, args, num_cliques):
    # 10-folder cross validation
    num_of_test = int(len(labels) * 0.1)
    total_acc = []
    for j in range(10):
        all_accuracy = []
        for i in range(10):
            net = GCN(input_size=num_cliques).to(DEVICE)
            loss = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.w_d)
            test_indices = []
            for j in range(num_of_test):
                test_indices.append(i * num_of_test + j + num_cliques)
            mask_train = [False if x in test_indices or x < num_cliques else True for x in range(num_cliques + len(labels))]
            mask_test = [False if x not in test_indices or x < num_cliques else True for x in range(num_cliques + len(labels))]
            # print("labels:", node_labels[mask_test])
            if mask_train == mask_test:
                print("ERROR")
            test_accuracy = 0
            train_acc_list, test_acc_list = [], []
            for epoch in range(args.num_epochs):
                net.train()
                output = net(adj, features)
                l = loss(output[mask_train], node_labels[mask_train])
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                net.eval()
                with torch.no_grad():
                    test_output = net(adj, features)
                    test_acc = accuracy(test_output[mask_test], node_labels[mask_test])
                    test_acc_list.append(test_acc)
                    if test_acc > test_accuracy:
                        test_accuracy = test_acc
            all_accuracy.append(test_accuracy)
            print(f"Best test accuracu of round {i} is {test_accuracy}")
        print(f"Test accuracy of model is {Average(all_accuracy)}")
        total_acc.append(Average(all_accuracy))
    print("Total average test acccuracy:", Average(total_acc))


def main():
    args = get_args()
    DEVICE = "cpu"
    data = FileLoader(args).load_data()
    labels = data.graph_labels
    for y in range(len(labels)):
        # labels[y] = labels[y] - 1
        if labels[y] == -1:
            labels[y] = 0
    g_graph, vocab = Generator(data).gen_large_graph()
    num_cliques = len(vocab)
    features, adj, node_labels = gen_feature_adj_labels(g_graph, labels, num_cliques)
    features = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    node_labels = torch.tensor(node_labels, dtype=torch.long).to(DEVICE)
    adj = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
    train(features, adj, node_labels, labels, DEVICE, args, num_cliques)


if __name__ == '__main__':
    main()


