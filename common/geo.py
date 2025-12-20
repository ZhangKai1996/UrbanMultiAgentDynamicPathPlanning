import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

base_speed = 45000.0 / 3600.0  # m/s


def load_graph(place_name):
    graph = ox.graph_from_place(place_name)
    p, num_action = remove_isolated_nodes(graph)
    add_dynamic_weight(graph)
    return graph, p, num_action


def remove_isolated_nodes(graph):
    print(f"Initial number of nodes: {len(graph.nodes)}")
    while True:
        # 删除孤立节点：没有连接任何其他节点的节点
        isolated_nodes = []
        ret, max_number = {}, 0
        for node1 in graph.nodes:
            tmp = []
            for node2, data in graph[node1].items():
                if node1 == node2: continue
                if data.get('length', 1) > 0: tmp.append(node2)
            if len(tmp) <= 0:
                isolated_nodes.append(node1)
            else:
                ret[node1] = tmp
                max_number = max(max_number, len(tmp))

        if len(isolated_nodes) <= 0: break
        print(isolated_nodes)
        # 从图中移除孤立节点
        graph.remove_nodes_from(isolated_nodes)
    print(f"Number of nodes after removing isolated nodes: {len(graph.nodes)}")
    print(f"Number of edges after removing isolated nodes: {len(graph.edges)}")
    return ret, max_number


def add_dynamics(graph, dynamic=None, limitations=None):
    if dynamic is None: return None, None

    nodes = list(graph.nodes)
    edge_index, edge_attr = [], []
    for u, v, data in graph.edges(data=True):
        key = '{}-{}'.format(u, v)
        if limitations is not None:
            if key not in limitations: continue

        times = dynamic[key]
        length = data.get('length', 1)
        traffic_time = length / (base_speed * times)
        data['times'] = times
        data['dynamic_weight'] = traffic_time

        i1 = nodes.index(u)
        i2 = nodes.index(v)
        edge_index.append([i1, i2])
        edge_attr.append(times)
        # edge_attr.append(traffic_time)
    return edge_index, edge_attr


def add_dynamic_weight(graph, nodes=None, dynamic=False):
    if nodes is None:
        nodes = list(graph.nodes)

    edge_index, edge_attr = [], []
    for u, v, data in graph.edges(data=True):
        times = 1.0
        if dynamic:
            times = np.random.uniform(0.1, 1.5)
        length = data.get('length', 1)
        traffic_time = length / (base_speed * times)
        data['times'] = times
        data['dynamic_weight'] = traffic_time

        i1 = nodes.index(u)
        i2 = nodes.index(v)
        edge_index.append([i1, i2])
        edge_attr.append(times)
        # edge_attr.append(traffic_time)
    return edge_index, edge_attr


def plot_graph(graph, name='graph', subgraph=None, node_size=10, show=False):
    """
    可视化原图和子图（子图以红色高亮显示）
    """
    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
    plt.figure(figsize=(12, 12))
    # 绘制原图
    nx.draw(graph,
            pos=pos,
            node_size=node_size * 2,
            edge_color='lightgray',
            node_color='gray',
            alpha=0.3,
            linewidths=0.2)

    if subgraph is not None:
        # 绘制子图
        nx.draw(subgraph,
                pos=pos,
                node_size=node_size,
                edge_color='red',
                node_color='red',
                alpha=0.9,
                linewidths=1.0)
    plt.title(name)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(name+'.png', dpi=300)
    if show: plt.show()


if __name__ == '__main__':
    graph_, *_ = load_graph(place_name='南京航空航天大学(将军路校区)')
    plot_graph(graph_)
