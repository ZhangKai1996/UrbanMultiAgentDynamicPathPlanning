import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim,
                 emb_dim=64,
                 other_dim=0,
                 hidden_dim=64):
        super(DQNEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.fc1 = nn.Linear(emb_dim*2+other_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = self.embedding(x[:, 0].to(torch.int32))
        x2 = self.embedding(x[:, 1].to(torch.int32))
        x3 = x[:, 2:]
        x = torch.cat([x1, x2, x3], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RNN(nn.Module):
    def __init__(self, input_dim, output_dim,
                 num_layers=1,
                 emb_dim=64,
                 hidden_dim=64):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(input_size=emb_dim*3,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          bidirectional=True,  # 双向 RNN
                          batch_first=True)    # 输入格式为 (batch_size, seq_length, input_size)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 预测访问顺序

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(*x.shape[:2], -1)
        x, *_ = self.rnn(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim,
                 emb_dim=256,
                 hidden_dim=64):
        super(GCN, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, emb_dim)

        self.fc1 = nn.Linear(emb_dim*3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state, x, edge_index, edge_weight=None, batch=None):
        x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        x = self.fc(x)

        x1 = self.embedding(state[:, 0].to(torch.int32))
        x2 = self.embedding(state[:, 1].to(torch.int32))
        x = torch.cat([x1, x2, x], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        return self.fc(x)
