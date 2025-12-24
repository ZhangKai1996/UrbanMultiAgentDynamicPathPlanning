import cv2
import numpy as np


text_kwargs = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.2,
                   color=(255, 255, 255),
                   thickness=4)
colors = [
    (255, 0, 0),    # 蓝色
    (0, 165, 255),  # 橙色
    (255, 255, 0),  # 青色/黄色偏青
    (255, 0, 255),  # 紫色
    (0, 255, 255),  # 黄色/青色偏黄
]


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


class StaticRender:
    def __init__(self, graph, height=2160, padding=0.075):
        self.graph = graph

        lats = [graph.nodes[node]['y'] for node in graph.nodes]
        lons = [graph.nodes[node]['x'] for node in graph.nodes]
        self.bounder = (min(lats), max(lats), min(lons), max(lons))

        delta_lat = (self.bounder[3] - self.bounder[2])
        delta_lon = (self.bounder[1] - self.bounder[0])
        width = int(height * delta_lat / delta_lon)
        self.padding = (int(width * padding), int(height * padding))
        self.size = (width, height)

        self.img = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        self.__draw_edges()
        self.__draw_nodes()

    def latlon_to_pixel(self, lat, lon):
        min_lat, max_lat, min_lon, max_lon = self.bounder
        pad_w, pad_h = self.padding
        w_ = self.size[0] - pad_w * 2
        h_ = self.size[1] - pad_h * 2

        x = int((lon - min_lon) / (max_lon - min_lon) * w_)
        y = h_ - int((lat - min_lat) / (max_lat - min_lat) * h_)
        return x+pad_w, y+pad_h

    def __draw_nodes(self, nodes=None, color=(0, 255, 0)):
        if nodes is None: nodes = self.graph.nodes
        for i, u in enumerate(nodes):
            pt = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
            cv2.circle(self.img, pt, 2, color, -1)

    def __draw_edges(self):
        nodes = self.graph.nodes
        for u, v, data in self.graph.edges(data=True):
            pt1 = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
            pt2 = self.latlon_to_pixel(nodes[v]['y'], nodes[v]['x'])
            color, thickness = (0, 255, 0), 1

            # if 'times' in data.keys():
            #     if data['times'] <= 0.5: color = (0, 0 ,255)
            cv2.line(self.img, pt1, pt2, color, thickness)

    def __draw_markers(self, node, text, radius=20, color=(128, 0, 128)):
        nodes = self.graph.nodes
        pt = self.latlon_to_pixel(nodes[node]['y'], nodes[node]['x'])
        cv2.circle(self.img, pt, radius, color, -1)
        cv2.putText(self.img, text, org=(pt[0], pt[1] + 60), **text_kwargs)

    def __draw_paths(self, paths):
        if paths is None: raise NotImplementedError

        count = 0
        nodes = self.graph.nodes
        for key, [seq, cost, path] in paths.items():
            color = colors[count]
            for i, u in enumerate(path[:-1]):
                v = path[i + 1]
                pt1 = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
                pt2 = self.latlon_to_pixel(nodes[v]['y'], nodes[v]['x'])
                cv2.line(self.img, pt1, pt2, color, 8)

            text = f'Agent {key}: ({round(cost, 3)},{len(path)}), '
            text += '-'.join([str(x) for x in seq])
            new_kwargs = text_kwargs.copy()
            new_kwargs['color'] = color
            cv2.putText(self.img, text, (120, 60 * count + 160), **new_kwargs)
            count += 1

    def draw(self, paths=None, init_nodes=None, tasks=None, name=None, show=True, **kwargs):
        if init_nodes is not None:
            self.__draw_markers(init_nodes[0], text='Start', color=(0, 0, 255))
            self.__draw_markers(init_nodes[1], text='End', color=(0, 0, 255))

        if tasks is not None:
            for task in tasks:
                self.__draw_markers(task['node'], text='Task{}'.format(task['id']))
        self.__draw_paths(paths)

        if name is not None: cv2.imwrite(name + '.png', self.img)
        if show:
            cv2.imshow("City Instant Delivery Simulation", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def close(self): pass
