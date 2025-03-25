import os
import random
from typing import Tuple

import numpy as np
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic as gd  # type: ignore
import json
import geopandas as gpd

from haversine import haversine, Unit
from scipy.interpolate import interp1d


class RouteGenerator:
    def __init__(self, config_path: str = "config.json", **kwargs):
        """Конструктор класса, которому в именованных аргументах передаётся
        либо название местности, либо точные координаты местности.
        Если было передано название, то происходит обращение к базе данных OSM,
        где потом извлекаются точные координаты. На основе координат строится граф дорог.


        Args:
            config_path (str, optional): Путь к файлу конфигурации. По умолчанию стоит "config.json".

        Raises:
            Exception: Не были переданы ни название местности, ни его координаты.
        """

        self.__load_config(config_path)
        self.data = {'X': [], 'y': []}

        if "place_name" in kwargs.keys():
            self.__place_bbox = list(
                ox.geocode_to_gdf(kwargs["place_name"]).geometry.total_bounds
            )

        elif "place_bbox" in kwargs.keys():
            self.__place_bbox = kwargs["place_bbox"]
        else:
            raise Exception(
                "Укажите название места согласно базе данных OSM либо координаты местности."
            )

        self.graph = ox.graph_from_bbox(self.__place_bbox, network_type="drive", simplify=False)  # Граф дорог местности

    def __load_config(self, file_path: str) -> None:
        """Загрузка данных о константах через файл конфигурации.

        Args:
            file_path (str): Путь к файлу конфигурации.
        """

        with open(file_path, "r") as file:
            config = json.load(file)
            self.__data_amount = config["data_amount"]  # Размер генерируемой выборки
            self.__min_segment = config[
                "min_segment"
            ]  # Минимальное значение отрезка для создания отклонения
            self.__max_segment = config[
                "max_segment"
            ]  # Максимальное значение отрезка для создания отклонения
            self.__min_offset = config["min_offset"]  # Минимальное отклонение
            self.__max_offset = config["max_offset"]  # Максимальное отклонение
            self.__max_route_len = config["max_route_len"]
            self.__min_route_len = config["min_route_len"]

    def save_false_route(self, main_route: list) -> Tuple["nx.MultiDiGraph", list, list]:
        """Генерация одного искажённого маршрута на основе исходного.

        Args:
            main_route: (list): Исходный маршрут.

        Returns:
            Tuple[nx.Graph, list]: Кортеж, внутри которого помещён изменённый граф и полученный маршрут.
        """

        path = main_route
        G = self.graph.copy()
        new_nodes = [path[0]]

        for i in range(len(path) - 1):
            # Начальная и конечная точки отрезка
            u, v = path[i], path[i + 1]
            point1 = (G.nodes[u]["y"], G.nodes[u]["x"])
            point2 = (G.nodes[v]["y"], G.nodes[v]["x"])

            # Расстояние между узлами
            edge_length = gd(point1, point2).meters
            direction_bearing = ox.bearing.calculate_bearing(
                point1[0], point1[1], point2[0], point2[1]
            )

            # Добавление точек через случайное расстояние между 20 и 60 метров
            current_dist = 0
            previous_node = u
            while current_dist < edge_length:
                # Случайное расстояние до следующей точки
                random_dist = random.uniform(self.__min_segment, self.__max_segment)
                current_dist += random_dist

                if current_dist >= edge_length:
                    break

                # Вычисление промежуточной точки
                new_point = gd(meters=current_dist).destination(
                    point1, direction_bearing
                )
                new_lat, new_lon = new_point.latitude, new_point.longitude

                # Случайное отклонение влево или вправо
                offset_direction = direction_bearing + (
                    90 if random.choice([True, False]) else -90
                )
                offset_dist = random.uniform(self.__min_offset, self.__max_offset)
                offset_point = gd(meters=offset_dist).destination(
                    (new_lat, new_lon), offset_direction
                )
                offset_lat, offset_lon = (offset_point.latitude, offset_point.longitude)

                # Добавление новой вершины и её координат
                new_node = max(G.nodes) + 1
                G.add_node(new_node, y=offset_lat, x=offset_lon)
                new_nodes.append(new_node)

                # Добавление ребра между новой точкой и предыдущей точкой
                G.add_edge(previous_node, new_node, length=random_dist)
                G.add_edge(
                    new_node, v, length=edge_length - current_dist
                )  # Связь с основным маршрутом

                previous_node = new_node  # Сместить начальную точку для следующего шага
            new_nodes.append(path[i + 1])

        false_route = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in new_nodes]
        return G, new_nodes, false_route

    def save_main_route(self) -> Tuple[list, list]:
        """Генерация и сохранение исходного маршрута

        Args: _
        """
        keys = list(self.graph.nodes.keys()).copy()
        node_ids = []

        while len(node_ids) < self.__min_route_len or len(node_ids) > self.__max_route_len:
            try:
                start = random.choice(keys)
                keys.remove(start)
                end = random.choice(keys)
                # Поиск кратчайшего пути
                node_ids = nx.astar_path(self.graph, start, end, weight="length")
            except nx.NetworkXNoPath:
                pass
        main_route = [(self.graph.nodes[n]["x"], self.graph.nodes[n]["y"])
                      for n in node_ids]
        return node_ids, main_route

    @staticmethod
    def calculate_cumulative_distances(route: "np.ndarray"):
        distances = [0]  # Начинаем с 0 элемента
        for i in range(1, len(route)):
            lon1, lat1 = route[i - 1]
            lon2, lat2 = route[i]
            distance = haversine((lon1, lat1), (lon2, lat2), unit=Unit.METERS)
            distances.append(distances[-1] + distance)
        return np.array(distances)

    # Функция для интерполяции маршрута

    def make_equal(self, route: list, num_points: int) -> list:
        route = np.array(route)
        # Вычисляем кумулятивное расстояние
        distances = self.calculate_cumulative_distances(route)

        # Создаем интерполяционные функции для широты и долготы
        interpolation_func_lon = interp1d(distances, route[:, 0], kind='linear')
        interpolation_func_lat = interp1d(distances, route[:, 1], kind='linear')

        new_distances = np.linspace(0, distances[-1], num_points)

        new_lon = interpolation_func_lon(new_distances)
        new_lat = interpolation_func_lat(new_distances)
        new_route = list(np.column_stack((new_lon, new_lat)))
        new_route = [tuple(point) for point in new_route]
        return new_route

    def save_data(self) -> None:
        for i in range(self.__data_amount):
            route_ids, main_route = self.save_main_route()
            _, _, false_route = self.save_false_route(route_ids)

            main_route = self.make_equal(main_route, len(false_route))
            self.data['y'].append(main_route)
            self.data['X'].append(false_route)
            if (i + 1) % 100 == 0:
                print(f"Сделано {i + 1}/{self.__data_amount} маршрутов")


if __name__ == "__main__":
    # generator = RouteGenerator(
    #     place_bbox=[39.0296, 51.7806, 39.3414, 51.5301]
    # )  # использованы примерные границы Воронежа
    # generator.save_data()
    #
    # with (open('test.json', 'w')) as f:
    #     json.dump(generator.data, f, indent=2)

    dirpath = os.path.dirname(__file__)
    config = os.path.join(dirpath, "..\\configs\\config.json")
    save = os.path.join(dirpath, "..\\training data")
    generator = RouteGenerator(config_path=config, place_bbox=[39.0296, 51.7806, 39.3414,
                                                               51.5301])  # 39.10146, 51.66079, 39.13532, 51.65488
    G = generator.graph
    # target, _ = generator.save_main_route()
    # _, _, input = generator.save_false_route(target)
    nodes, edges = ox.graph_to_gdfs(G)
    nodes.to_csv(os.path.join(save, 'nodes.csv'))
    edges.to_csv(os.path.join(save, 'edges.csv'))
