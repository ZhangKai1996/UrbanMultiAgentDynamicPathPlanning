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
    def __init__(self, env, height=2160, padding=0.075, filename='video'):
        self.env = env
        self.graph = env.graph

        lats = [self.graph.nodes[node]['y'] for node in env.nodes]
        lons = [self.graph.nodes[node]['x'] for node in env.nodes]
        self.bounder = (min(lats), max(lats), min(lons), max(lons))

        delta_lat = (self.bounder[3] - self.bounder[2])
        delta_lon = (self.bounder[1] - self.bounder[0])
        width = int(height * delta_lat / delta_lon)
        self.padding = (int(width * padding), int(height * padding))
        self.size = (width, height)
        self.video_writer = cv2.VideoWriter(filename=filename+'.avi',
                                            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                                            fps=10,
                                            frameSize=(width, height))
        self.base_img = None
        self.build_base_image()

    def build_base_image(self):
        self.base_img = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8) * 255
        self.__draw_nodes(self.base_img)
        self.__draw_edges(self.base_img)
        for task in self.env.all_tasks:
            self.__draw_markers(self.base_img, task.node, text=str(task.name))
        for agent in self.env.agents:
            self.__draw_markers(self.base_img, agent.start_node, text='Start', color=(0, 0, 255))
            self.__draw_markers(self.base_img, agent.end_node, text='End', color=(0, 0, 255))

    def latlon_to_pixel(self, lat, lon):
        min_lat, max_lat, min_lon, max_lon = self.bounder
        pad_w, pad_h = self.padding
        w_ = self.size[0] - pad_w * 2
        h_ = self.size[1] - pad_h * 2

        x = int((lon - min_lon) / (max_lon - min_lon) * w_)
        y = h_ - int((lat - min_lat) / (max_lat - min_lat) * h_)
        return x+pad_w, y+pad_h

    def __draw_nodes(self, img, nodes=None, color=(0, 255, 0)):
        if nodes is None: nodes = self.graph.nodes
        for i, u in enumerate(nodes):
            pt = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
            cv2.circle(img, pt, 2, color, -1)

    def __draw_edges(self, img):
        nodes = self.graph.nodes
        for u, v in self.graph.edges(data=False):
            pt1 = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
            pt2 = self.latlon_to_pixel(nodes[v]['y'], nodes[v]['x'])
            color, thickness = (0, 255, 0), 1
            data = self.env.dynamics.now()
            if data[f'{u}-{v}'] <= 0.5: color = (0, 0, 255)
            cv2.line(img, pt1, pt2, color, thickness)

    def __draw_markers(self, img, node, text, radius=20, color=(128, 0, 128)):
        nodes = self.graph.nodes
        pt = self.latlon_to_pixel(nodes[node]['y'], nodes[node]['x'])
        cv2.circle(img, pt, radius, color, -1)
        cv2.putText(img, text, org=(pt[0], pt[1] + 60), **text_kwargs)

    def __draw_paths(self, img, paths):
        if paths is None: raise NotImplementedError

        count = 0
        nodes = self.graph.nodes
        for key, [seq, cost, path] in paths.items():
            color = colors[count]
            for i, u in enumerate(path[:-1]):
                v = path[i + 1]
                pt1 = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
                pt2 = self.latlon_to_pixel(nodes[v]['y'], nodes[v]['x'])
                cv2.line(img, pt1, pt2, color, 8)

            text = f'Agent {key}: ({round(cost, 3)},{len(path)}), '
            text += '-'.join([str(x) for x in seq])
            new_kwargs = text_kwargs.copy()
            new_kwargs['color'] = color
            cv2.putText(img, text, (120, 60 * count + 160), **new_kwargs)
            count += 1

    def draw(self, show=True, has_dynamics=False, **kwargs):
        for task in self.env.all_tasks:
            if task.is_completed():
                text = str(task.name) + ' (Done)'
                self.__draw_markers(self.base_img, task.node, text=text, color=(0, 255, 0))
            else:
                self.__draw_markers(self.base_img, task.node, text=str(task.name))

        base_img = self.base_img.copy()
        cv2.putText(base_img, f'Clock: {self.env.clock}', (120, 100), **text_kwargs)

        nodes = self.graph.nodes
        for i, agent in enumerate(self.env.agents):
            color = colors[i]
            completed_tasks = agent.completed_tasks
            assigned_tasks = agent.assigned_tasks

            if not agent.is_over():
                self.__draw_markers(base_img, agent.current_node,
                                    text=f'Here {agent.id}',
                                    color=(0, 255, 255))
                u = agent.current_node
                v = agent.last_node
                if v is not None:
                    pt1 = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
                    pt2 = self.latlon_to_pixel(nodes[v]['y'], nodes[v]['x'])
                    cv2.line(self.base_img, pt1, pt2, color, 8)
                    cv2.line(base_img, pt1, pt2, color, 8)

                # u_ = u
                # for t in assigned_tasks:
                #     v_ = t.node
                #     pt1_ = self.latlon_to_pixel(nodes[u_]['y'], nodes[u_]['x'])
                #     pt2_ = self.latlon_to_pixel(nodes[v_]['y'], nodes[v_]['x'])
                #     cv2.line(base_img, pt1_, pt2_, color, 8)
                #     u_ = v_
                # v_ = agent.end_node
                # pt1_ = self.latlon_to_pixel(nodes[u_]['y'], nodes[u_]['x'])
                # pt2_ = self.latlon_to_pixel(nodes[v_]['y'], nodes[v_]['x'])
                # cv2.line(base_img, pt1_, pt2_, color, 8)

                path = agent.planned_path
                for j, u in enumerate(path[:-1]):
                    v = path[j + 1]
                    pt1 = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
                    pt2 = self.latlon_to_pixel(nodes[v]['y'], nodes[v]['x'])
                    cv2.line(base_img, pt1, pt2, color, 8)

            text = f'Agent {agent.id}: ({round(agent.cost, 3)},'
            text += f'{len(completed_tasks)},{len(assigned_tasks)}), '
            text += 'S'
            if len(completed_tasks) > 0:
                text += '+'
                text += '+'.join([str(task.name) for task in completed_tasks])
            if len(assigned_tasks) > 0:
                text += '-'
                text += '-'.join([str(task.name) for task in assigned_tasks])
            text += '+E' if agent.is_over() else '-E'
            new_kwargs = text_kwargs.copy()
            new_kwargs['color'] = color
            cv2.putText(base_img, text, (120, 60 * i + 160), **new_kwargs)

        self.video_writer.write(base_img)  # Write frame to video file
        if has_dynamics: self.__draw_edges(self.base_img)
        if show:
            cv2.imshow("City Instant Delivery Simulation", base_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def close(self): pass
