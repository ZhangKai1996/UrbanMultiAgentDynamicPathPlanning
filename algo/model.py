import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 emb_dim=64,
                 hidden_dim=128):
        super(GCN, self).__init__()
        self.embedding = nn.Embedding(state_dim, emb_dim)

        self.gcn = GCNConv(emb_dim, hidden_dim)
        self.fc1 = nn.Linear(emb_dim + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.value_fc = nn.Linear(hidden_dim, 1)
        self.advantage_fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, x, edge_index, edge_weight=None, batch=None):
        """
        state: [batch_size, 1]  # 当前状态索引
        x: [num_nodes, 2]       # 节点特征
        edge_index: [2, num_edges]  # PyG 格式的边索引
        edge_weight: [num_edges]    # 边权重，可选
        batch: [num_nodes]          # 批次节点索引，可选
        """
        x1 = self.embedding(state[:, 0].to(th.int64))  # [batch_size, emb_dim]
        x = self.embedding(x.to(th.int64))
        # 如果有 batch，PyG 也能处理分批图
        # print(type(state), type(x), type(edge_index), type(edge_weight))
        # print(state.shape, x.shape, edge_index.shape, edge_weight.shape)
        x2 = self.gcn(x, edge_index, edge_weight=edge_weight)
        if batch is not None:
            x2 = global_mean_pool(x2, batch)  # 对每个图节点求均值
        else:
            x2 = x2.mean(dim=0, keepdim=True).repeat(x1.size(0), 1)  # 没有 batch 时，直接全局平均

        # 拼接 embedding
        x_cat = th.cat([x1, x2], dim=-1)
        x = F.relu(self.fc1(x_cat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.value_fc(x)  # [batch_size, 1]
        advantage = self.advantage_fc(x)  # [batch_size, action_dim]
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class GraphConvolution(nn.Module):
    """支持加权邻接矩阵的GCN层"""
    def __init__(self, emb_dim, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(emb_dim, out_features)

    def forward(self, x, adj):
        """
        x: [batch_size, num_nodes, in_features]
        adj: [batch_size, num_nodes, num_nodes] 加权邻接矩阵（可带自环）
        """
        # 对邻接矩阵进行归一化：D^(-0.5) * A * D^(-0.5)
        # adj 权重可能不为 0/1
        deg = adj.sum(dim=-1, keepdim=True) + 1e-6  # [B, N, 1] 避免除零
        adj_norm = adj / th.sqrt(deg * deg.transpose(1, 2))  # [B, N, N]

        support = self.linear(x)                 # [B, N, out_features]
        out = th.bmm(adj_norm, support)       # [B, N, out_features]
        return out


class DuelingDQN(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 emb_dim=64,
                 other_dim=0,
                 hidden_dim=128):
        super(DuelingDQN, self).__init__()
        self.embedding = nn.Embedding(state_dim, emb_dim)

        self.fc1 = nn.Linear(emb_dim+other_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Value 分支
        self.value_fc = nn.Linear(hidden_dim, 1)
        self.advantage_fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, dyn_features=None):
        x = self.embedding(state[:, 0].to(th.int32))
        if dyn_features is not None:
            x = th.cat([x, dyn_features], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.value_fc(x)  # [batch_size, 1]
        advantage = self.advantage_fc(x)  # [batch_size, action_dim]
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class MLP(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 emb_dim=64,
                 hidden_dim=128):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(state_dim, emb_dim)

        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.embedding(state[:, 0].to(th.int32))
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


class Critic(nn.Module):
    def __init__(self, n_agent, dim_obs, dim_act):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.FC = nn.Linear(3136, 128)
        self.FC1 = nn.Linear((128 + dim_act) * n_agent, 256)
        self.FC2 = nn.Linear(256, 128)
        self.FC3 = nn.Linear(128, 1)

    def forward(self, obs_n, act_n):
        batch_size = obs_n.size(0)
        combined = []
        for i in range(self.n_agent):
            out = F.relu(self.mp(self.conv1(obs_n[:, i])))
            out = F.relu(self.mp(self.conv2(out)))
            out = F.relu(self.mp(self.conv3(out)))
            out = F.relu(self.conv4(out))
            out = out.view(batch_size, -1)
            out = F.relu(self.FC(out))
            combined.append(th.cat([out, act_n[:, i]], 1))
        combined = th.cat(combined, 1)
        out = F.relu(self.FC1(combined))
        out = F.relu(self.FC2(out))
        out = self.FC3(out)
        return out


class Actor(nn.Module):
    def __init__(self, dim_obs, dim_act):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_obs, 512)
        self.FC2 = nn.Linear(512, 128)
        self.FC3 = nn.Linear(128, dim_act)

    def forward(self, obs_n):
        in_size = obs_n.size(0)
        out = out.view(in_size, -1)
        out = F.relu(self.FC1(out))
        out = F.relu(self.FC2(out))
        out = th.tanh(self.FC3(out))
        return out