import os
import random
from typing import Tuple
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic as gd  # type: ignore
import json


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
        self.__data = {'place_bbox': '', 'samples': []}

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

        self.__graph = ox.graph_from_bbox(self.__place_bbox)  # Граф дорог местности
        self.__data['place_bbox'] = str(self.__place_bbox)
        self.__data['samples'] = {'inputs': [], 'targets': []}

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

    def __save_false_route(self, main_route: list) -> list:
        """Генерация одного искажённого маршрута на основе исходного.

        Args:
            list: Исходный маршрут.

        Returns:
            Tuple[nx.Graph, list]: Кортеж, внутри которого помещён изменённый граф и полученный маршрут.
        """

        path = main_route
        G = self.__graph.copy()
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

        false_route = [{'x': G.nodes[n]["x"], 'y': G.nodes[n]["y"]} for n in new_nodes]
        self.__data['samples']['inputs'].append(false_route)

    def __save_main_route(self) -> list:
        """Генерация и сохранение исходного маршрута в указанную папку в виде единого изображения,
        а также в виде отдельных частей размером 256х256.
        Получение изображения происходит с помощью запроса по протоколу TMS.

        Args:
            save_folder (str): Путь к папке для сохранения.
        """
        keys = list(self.__graph.nodes.keys()).copy()
        node_ids = []
        start = random.choice(keys)
        keys.remove(start)
        while len(node_ids) == 0:
            try:
                end = random.choice(keys)
                # Поиск кратчайшего пути
                node_ids = nx.astar_path(self.__graph, start, end, weight="length")
            except nx.NetworkXNoPath:
                pass
        main_route = [{'id': n, 'x': self.__graph.nodes[n]["x"], 'y': self.__graph.nodes[n]["y"]}
                      for n in node_ids]
        self.__data['samples']['targets'].append(main_route)
        return node_ids

    def save_data(self, save_file: str) -> None:
        for i in range(self.__data_amount):
            main_route_ids = self.__save_main_route()
            self.__save_false_route(main_route_ids)
        with open(save_file, "w") as file:
            json.dump(self.__data, file, indent=2)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    data_generator = RouteGenerator(
        place_bbox=[39.121447, 51.646002, 39.135578, 51.653782]
    )
    data_generator.save_data("test.json")
