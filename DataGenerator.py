import random

import osmnx as ox
import networkx as nx
import geopandas as gpd
import json

# TODO
"""
1. Подумать, как можно построить путь на графе из списка координат.
2. Подумать, как можно вычислять расстояние между координатами,
чтобы в будущем через каждые 20-60 м ставить помехи в маршруте. 
3. Подумать, как добавлять новые вершины в уже существующий граф.
"""


class DataGenerator:
    def __init__(self, config_path: str = "config.json", **kwargs):
        self.__load_config(config_path)
        self.main_route = []  # основной маршрут, содержит координаты вершин
        self.false_routes = []  # список сгенерированных "ложных" маршрутов
        self.__place_bbox = []  # border box, задаёт границы для исследуемой местности
        # через координаты противоположных углов

        if "place_name" in kwargs.keys():
            self.__place_bbox = (ox.geocode_to_gdf(kwargs["place_name"])
                                 .geometry
                                 .total_bounds)
        elif "place_bbox" in kwargs.keys():
            self.__place_bbox = kwargs["place_bbox"]
        else:
            pass
        self.__graph = ox.graph_from_bbox(*self.__place_bbox, network_type='drive')

    def __load_config(self, file_path: str) -> None:
        with open(file_path, 'r') as file:
            config = json.load(file)
            self.__data_amount = config["data_amount"]
            self.__node_interval_left = config["node_interval_left"]
            self.__node_interval_right = config["node_interval_right"]

    def __get_main_route(self) -> None:
        # Выбор случайных точек в графе
        keys = list(self.__data_amount.keys())
        start_node = random.choice(keys)
        end_node = random.choice(keys)
        while start_node == end_node:
            end_node = random.choice(keys)

        # Поиск кратчайшего пути с помощью алгоритма А*
        self.main_route = nx.astar_path(self.__graph, start_node,
                                        end_node, weight='length')
        self.__main_route_coordinates = self.__get_route_coordinates(self.main_route)

    def __get_route_coordinates(self, route: list) -> list:
        coordinates = []
        for node_id in route:
            lat, lon = self.__graph.nodes[node_id]['y'], self.__graph.nodes[node_id]['x']
            coordinates.append((lat, lon))
        return coordinates

    def __get_one_false_route(self) -> list:
        false_route = []
        return false_route

    # Метод, генерирующий ложные маршруты
    def get_false_routes(self) -> None:
        self.__get_main_route()
        for _ in range(self.__data_amount):
            self.false_routes.append(self.__get_one_false_route())

    def show_main_route(self) -> None:
        if not self.main_route:
            self.__get_main_route()
        ox.plot_graph_route(self.__graph, self.main_route)

    def show_route(self, route: list) -> None:
        ox.plot_graph_route(self.__graph, route)


if __name__ == '__main__':
    data_generator = DataGenerator(place_name="Тепличный, Voronezh, Russia")
    data_generator.show_main_route()
