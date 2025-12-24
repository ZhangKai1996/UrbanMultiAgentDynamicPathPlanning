import math

import numpy as np
import osmnx as ox
import networkx as nx

from common.utils import haversine
from env.rendering import StaticRender

base_speed = 45000.0 / 3600.0  # m/s


def load_graph(place_name='Nanjing, China',
               network_type='all',
               center_idx=0,
               radius=1e6,
               remove=True,
               render=False):
    print('-------Loading urban graph-----------')
    print('\t Place name:', place_name)
    print('\t Network type:', network_type)
    print('\t Radius:', radius)
    print('\t Center index:', center_idx)
    print('\t Remove isolated nodes:', remove)
    print('\t Render the graph:', render)
    print('\t Graph process:')

    graph_ = ox.graph_from_place(place_name, network_type=network_type)
    print(f"\t\t Initial {len(graph_.nodes)} nodes and {len(graph_.edges)} edges")
    print(f"\t\t Strong Connection: {nx.is_strongly_connected(graph_)}")

    graph, center_node = extract_subgraph(radius, graph_, center_idx)
    print(f"\t\t After clipped by distance:", end=' ')
    print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"\t\t Strong Connection: {nx.is_strongly_connected(graph)}")

    if remove:
        g_strong = max(nx.strongly_connected_components(graph), key=len)
        graph = graph.subgraph(g_strong).copy()
        print(f"\t\t After strongly connected:", end=' ')
        print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")

    p, num_action = remove_isolated_nodes(graph)
    print(f"\t\t After removing isolated nodes:", end=' ')
    print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"\t\t Number of action: {num_action}")

    if render:
        name = 'fig_ranges_{}'.format(radius)
        ranges = {'center': center_node, 'radius': radius}
        StaticRender(graph_).draw(vf=list(graph.nodes), ranges=ranges, name=name, show=True)
        StaticRender(graph, height=600).draw(name=name+'_sub', show=True)

    print(f"\t\t Center node (origin): ", center_node, center_node in list(graph.nodes))
    center_node, ranked_nodes = get_rank(graph, center_node, change_end_node=True)
    print(f"\t\t Center node (new): ", center_node, center_node in list(graph.nodes))

    add_dynamic_weight(graph)
    print(f"\t\t Strong Connection: {nx.is_strongly_connected(graph)}")
    return graph, p, num_action, center_node, ranked_nodes


def get_rank(graph, end_node, change_end_node=False):
    nodes = list(graph.nodes)
    end_node_idx = nodes.index(end_node)

    ret = {}
    for node_idx in range(len(nodes)):
        if node_idx == end_node_idx: continue

        u = graph.nodes[nodes[node_idx]]
        v = graph.nodes[end_node]
        ret[node_idx] = haversine(u, v)

    sorted_items = sorted(ret.items(), key=lambda x: x[1])
    dist_arr = np.array([item[1] for item in sorted_items])
    rank_key = [item[0] for item in sorted_items]

    if change_end_node:
        end_node = nodes[rank_key[-1]]
        return get_rank(graph, end_node, change_end_node=False)
    return end_node, rank_key


def extract_subgraph(radius, graph, center_idx):
    center_node = list(graph.nodes)[center_idx]
    nodes = graph.nodes

    sub_nodes = []
    for i, node in enumerate(list(nodes)):
        dist = haversine(nodes[center_node], nodes[node], km=True)
        if dist <= radius: sub_nodes.append(node)
    subgraph = graph.subgraph(sub_nodes).copy()
    return subgraph, center_node


def remove_isolated_nodes(graph):
    """从图中移除孤立节点：没有连接任何其他节点的节点"""
    while True:
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
        graph.remove_nodes_from(isolated_nodes)
    return ret, max_number


def add_dynamic_weight(graph, dynamics=False, alpha=0.5):
    nodes = list(graph.nodes)

    edge_index, edge_attr = [], []
    for u, v, data in graph.edges(data=True):
        times = 1.0
        if isinstance(dynamics, dict):
            times = dynamics['{}-{}'.format(u, v)]
        elif dynamics:
            times = np.random.uniform(0.1, 1.5)

        length = data.get('length', 1)
        traffic_time = length / (base_speed * times)
        static_time = length / base_speed

        data['times'] = times
        data['dynamic_weight'] = traffic_time
        data['static_weight'] = static_time
        data['mixed_weight'] = alpha * traffic_time + (1 - alpha) * static_time

        edge_index.append([nodes.index(u), nodes.index(v)])
        edge_attr.append(times)
    return edge_index, edge_attr


def _get_all_neighbors(graph, source, n):
    visited = {source}  # 已访问过的节点（包含自己）
    current_layer = {source}

    for depth in range(1, n + 1):
        next_layer = set()
        for node in current_layer:
            neighbors = set(graph.neighbors(node))
            new_nodes = neighbors - visited
            next_layer |= new_nodes
        visited |= next_layer
        current_layer = next_layer
    return list(visited)


def get_hop_neighborhoods(graph, target=None, n=1, k=1.0):
    print('------Get n-hop neighborhoods--------')
    print('\t n =', n)
    print('\t k =', k)

    if n < 0:  return {'subgraphs': None, 'max_len': len(graph.edges)}
    if n == 0: return {'subgraphs': {}, 'max_len': 1}
    nodes = graph.nodes

    ret, len_nodes = {}, []
    for u in list(nodes):
        node_list = []
        for x in  _get_all_neighbors(graph, u, n=n):
            if is_in_sector(u, target, x, nodes, k=k):
                node_list.append(x)

        subgraph = graph.subgraph(node_list).copy()
        len_nodes.append(len(subgraph.edges))
        ret[u] = subgraph

    print(f"\t Max Subgraph Length: {max(len_nodes)}")
    print(f"\t Min Subgraph Length: {min(len_nodes)}")
    return {'subgraphs': ret, 'max_len': max(len_nodes)}


def is_in_sector(u, v, x, nodes, k=1.0):
    """ 判断节点 x 是否位于 u 指向 v 的扇形（±angle_deg°）内 """
    assert 0.0 <= k <= 1.0
    if v is None: return True

    angle_uv = math.atan2(nodes[v]['y'] - nodes[u]['y'], nodes[v]['x'] - nodes[u]['x'])
    angle_ux = math.atan2(nodes[x]['y'] - nodes[u]['y'], nodes[x]['x'] - nodes[u]['x'])
    diff = abs((angle_ux - angle_uv + math.pi) % (2 * math.pi) - math.pi)
    return diff <= math.radians(k * 180)
