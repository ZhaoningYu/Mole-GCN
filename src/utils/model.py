import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(adj):
    adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    print("degree:", degree)
    return degree.dot(adj).dot(degree)


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, adj, features):
        output = torch.mm(adj, features)
        output = self.linear(output)
        return output


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, 512)
        self.gcn2 = GraphConvolution(512, 128)
        self.gcn3 = GraphConvolution(128, 48)
        self.gcn4 = GraphConvolution(48, 2)
        self.dropout = nn.Dropout(0.2)
        self.fcl1 = nn.Linear(48, 16)
        self.fcl2 = nn.Linear(16, 2)


    def forward(self, adj, features):
        output = F.relu(self.gcn1(adj, features))
        # output = self.dropout(output)
        output = F.relu(self.gcn2(adj, output))
        # output = self.dropout(output)
        output = F.relu(self.gcn3(adj, output))
        # output = self.gcn4(adj, output)
        output = F.relu(self.fcl1(output))
        output = self.fcl2(output)
        return output