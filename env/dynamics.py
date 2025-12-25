import os
import yaml

from common.geo import add_dynamic_weight


class DynamicsIterator:
    def __init__(self, graph, radius, num_dyn, env_id):
        self.graph = graph
        self.radius = radius
        self.num_dyn = num_dyn

        self.load_path = f'data/dynamics_{radius}_{num_dyn}_{env_id}.yaml'
        self.dynamics = self.__parse_env_dynamics()
        self.idx = 0

    def now(self): return self.dynamics[self.idx]

    def reset(self): self.idx = -1

    def next(self):
        self.idx += 1
        self.idx %= len(self.dynamics)
        return self.dynamics[self.idx]

    def __generate_env_dynamics(self):
        nodes = list(self.graph.nodes)

        data = {}
        for i in range(self.num_dyn):
            edge_index, edge_attr, *_ = add_dynamic_weight(self.graph, dynamics=True)
            ret = {}
            for j, (i1, i2) in enumerate(edge_index):
                key = '{}-{}'.format(nodes[i1], nodes[i2])
                ret[key] = edge_attr[j]
            data[i] = ret

        with open(self.load_path, 'w', encoding='utf-8') as f:
            yaml.dump(data=data, stream=f)
            print('\t Generate a new dynamics.')

    def __parse_env_dynamics(self):
        print('--------Load env dynamics------------')
        print('\t Load path:', self.load_path)
        if not os.path.exists(self.load_path):
            print('\t No env dynamics !')
            print('\t Generating env dynamics ...')
            self.__generate_env_dynamics()

        with open(self.load_path, 'r') as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)

        dynamics = list(data.values())
        print('\t Dynamics instances: ', len(dynamics))
        return dynamics

