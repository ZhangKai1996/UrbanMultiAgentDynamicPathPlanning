import cv2
import numpy as np


def latlon_to_pixel(lat, lon, bounder, size):
    min_lat, max_lat, min_lon, max_lon = bounder
    x = int((lon - min_lon) / (max_lon - min_lon) * size[0])
    y = int((lat - min_lat) / (max_lat - min_lat) * size[1])
    return x, size[1] - y


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


class StaticRender:
    def __init__(self, graph, height=1080):
        self.graph = graph

        lats = [graph.nodes[node]['y'] for node in graph.nodes]
        lons = [graph.nodes[node]['x'] for node in graph.nodes]
        # min_lat, max_lat, min_lon, max_lon
        self.bounder = (min(lats), max(lats), min(lons), max(lons))
        # 计算地图长宽比
        delta_lat = (self.bounder[3] - self.bounder[2])
        delta_lon = (self.bounder[1] - self.bounder[0])
        width = int(height * delta_lat / delta_lon)
        self.size = (width, height)

    def draw(self, path, vo, **kwargs):
        img = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8) * 255

        nodes = self.graph.nodes
        # 绘制边
        for edge in self.graph.edges:
            node1 = nodes[edge[0]]
            node2 = nodes[edge[1]]
            pt1 = latlon_to_pixel(node1['y'], node1['x'], self.bounder, self.size)
            pt2 = latlon_to_pixel(node2['y'], node2['x'], self.bounder, self.size)
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)

        # 绘制实际轨迹 (金色)
        for i, u in enumerate(path[:-1]):
            v = path[i+1]
            pt1 = latlon_to_pixel(nodes[u]['y'], nodes[u]['x'], self.bounder, self.size)
            pt2 = latlon_to_pixel(nodes[v]['y'], nodes[v]['x'], self.bounder, self.size)
            cv2.line(img, pt1, pt2, (255, 215, 0), 4)  # 金色

        # 绘制任务点
        for i, u in enumerate(vo):
            pt = latlon_to_pixel(nodes[u]['y'], nodes[u]['x'], self.bounder, self.size)

            color, text = (128, 0, 128), str(i)
            if i == 0:
                color, text = (255, 0, 0), 'Start'
            if i == len(vo) - 1:
                color, text = (0, 255, 0), 'End'

            cv2.circle(img, pt, 20, color, -1)
            cv2.putText(img, text,
                        org=(pt[0] + 15, pt[1] - 15),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=(255, 255, 255),
                        thickness=4)

        cv2.imshow("City Instant Delivery Simulation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class EnvRender:
    def __init__(self, env):
        self.env = env
        city_map = env.city_map
        self.lats = [city_map.nodes[node]['y'] for node in city_map.nodes]
        self.lons = [city_map.nodes[node]['x'] for node in city_map.nodes]

        min_lat, max_lat = min(self.lats), max(self.lats)
        min_lon, max_lon = min(self.lons), max(self.lons)

        # 计算地图长宽比
        map_width = max_lon - min_lon
        map_height = max_lat - min_lat
        aspect_ratio = map_width / map_height

        screen_height = 2560
        screen_width = int(screen_height * aspect_ratio)
        self.screen_size = (screen_width, screen_height)

        self.video_writer = cv2.VideoWriter(
            'trained/vedio.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            10, (self.screen_size[0], self.screen_size[1])
        )

    def draw(self, mode='human', threshold=0.6):
        img = np.ones((self.screen_size[1], self.screen_size[0], 3), dtype=np.uint8) * 255

        min_lat, max_lat = min(self.lats), max(self.lats)
        min_lon, max_lon = min(self.lons), max(self.lons)

        city_map = self.env.city_map
        vehicle = self.env.vehicle
        # 绘制边
        for edge in city_map.edges:
            node1 = city_map.nodes[edge[0]]
            node2 = city_map.nodes[edge[1]]
            pt1 = latlon_to_pixel(node1['y'], node1['x'], min_lat, max_lat, min_lon, max_lon, self.screen_size)
            pt2 = latlon_to_pixel(node2['y'], node2['x'], min_lat, max_lat, min_lon, max_lon, self.screen_size)

            weight = city_map.edges[edge].get('times', 1.0)
            if weight >= threshold:
                color = (169, 169, 169)  # 灰色
            else:
                color = (169, 169, int(155 * (1.0 - weight/threshold)) + 100)
            cv2.line(img, pt1, pt2, color, 2)

        # 绘制实际轨迹 (金色)
        for i in range(1, len(vehicle.trajectory)):
            node1 = city_map.nodes[vehicle.trail[i - 1]]
            node2 = city_map.nodes[vehicle.trail[i]]
            pt1 = latlon_to_pixel(node1['y'], node1['x'], min_lat, max_lat, min_lon, max_lon, self.screen_size)
            pt2 = latlon_to_pixel(node2['y'], node2['x'], min_lat, max_lat, min_lon, max_lon, self.screen_size)
            cv2.line(img, pt1, pt2, (255, 215, 0), 4)  # 金色

        # 绘制任务点 (取货点和送货点)
        for task in self.env.tasks:
            thickness = 4 if not task.is_picked() else -1
            pickup_pt = latlon_to_pixel(city_map.nodes[task.pickup_point]['y'], city_map.nodes[task.pickup_point]['x'],
                                        min_lat, max_lat, min_lon,
                                        max_lon, self.screen_size)
            # 取货点（蓝色）
            cv2.circle(img, pickup_pt, 20, (0, 0, 0), thickness)  # 蓝色
            text = f'P{task.task_id}'
            text += '({})'.format(round(task.pickup_time/60.0, 1)) if task.is_picked() else ''
            cv2.putText(img, text, (pickup_pt[0] + 15, pickup_pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 0), 4)

            thickness = 4 if not task.is_delivered() else -1
            delivery_pt = latlon_to_pixel(city_map.nodes[task.delivery_point]['y'],
                                          city_map.nodes[task.delivery_point]['x'], min_lat, max_lat, min_lon,
                                          max_lon, self.screen_size)
            # 送货点（绿色）
            cv2.circle(img, delivery_pt, 20, (128, 0, 128), thickness)  # 绿色
            text = f'D{task.task_id}'
            text += '({})'.format(round(task.delivery_time/60.0, 1)) if task.is_delivered() else ''
            cv2.putText(img, text, (delivery_pt[0] + 15, delivery_pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (128, 0, 128), 4)

        # 绘制最终目标点 (绿色)
        end_node = city_map.nodes[self.env.end_node]
        pt = latlon_to_pixel(end_node['y'], end_node['x'], min_lat,
                             max_lat, min_lon,
                             max_lon, self.screen_size)
        cv2.circle(img, pt, 20, (34, 34, 178), -1)  # 绿色实心圆
        cv2.putText(img, 'Final Target', (pt[0] + 15, pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (34, 34, 178), 4)

        # 绘制车辆当前位置 (紫色)
        vehicle_node = city_map.nodes[vehicle.position]
        pt = latlon_to_pixel(vehicle_node['y'], vehicle_node['x'], min_lat, max_lat, min_lon, max_lon, self.screen_size)
        cv2.circle(img, pt, 16, (255, 215, 0), -1)  # 紫色
        cv2.putText(img, 'Vehicle', (pt[0] + 15, pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 215, 0), 4)

        # 绘制初始点 (橙色)
        initial_point = city_map.nodes[vehicle.initial_position]
        pt = latlon_to_pixel(initial_point['y'], initial_point['x'], min_lat, max_lat, min_lon, max_lon,
                             self.screen_size)
        cv2.circle(img, pt, 20, (79, 79, 47), -1)  # 橙色
        cv2.putText(img, 'Initial Point', (pt[0] + 15, pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (79, 79, 47), 4)

        # 图例
        legend_start_x, legend_start_y = int(self.screen_size[0] * 0.6), 100
        cv2.putText(img, 'Clock: {}'.format(self.env.clock), (legend_start_x, legend_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        cv2.putText(img, 'Time Cost: {:>6.1f} s'.format(vehicle.time_cost), (legend_start_x, legend_start_y+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        cv2.putText(img, 'Distance Cost: {:>6.1f} km'.format(vehicle.distance_cost), (legend_start_x, legend_start_y+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        cv2.putText(img, 'Number of Task: {}'.format(len(self.env.tasks)), (legend_start_x, legend_start_y+150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        cv2.putText(img, 'Algo: {}'.format(mode), (legend_start_x, legend_start_y+200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

        self.video_writer.write(img)  # Write frame to video file
        cv2.imshow("City Instant Delivery Simulation", img)
        cv2.waitKey(1)

    def close(self):
        self.video_writer.release()
        cv2.destroyAllWindows()
