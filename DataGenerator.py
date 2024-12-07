import os
import random
from typing import Tuple
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic as gd  # type: ignore
import json
from TMSRequest import TMSRequest


class DataGenerator:
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
        self.main_routes = []  # Исходные маршруты
        self.false_routes = []  # Искажённые маршруты

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

    def __generate_main_route(self) -> list:
        """Генерация одного исходного маршрута на основе алгоритма А*.

        Returns:
            list: Список идентификаторов узлов графа.
            Сами идентификаторы заданы заранее на основе базы OSM.
        """

        # Выбор случайных точек в графе
        keys = list(self.__graph.nodes.keys()).copy()
        start = random.choice(keys)
        keys.remove(start)
        end = random.choice(keys)

        # Поиск кратчайшего пути
        main_route = nx.astar_path(self.__graph, start, end, weight="length")
        return main_route

    def __get_one_false_route(self, main_route: list) -> Tuple[nx.Graph, list]:
        """Генерация одного искажённого маршрута на основе исходного.

        Args:
            list: Исходный маршрут.

        Returns:
            Tuple[nx.Graph, list]: Кортеж, внутри которого помещён изменённый граф и полученный маршрут.
        """

        path = main_route
        G = self.__graph
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
        return G, new_nodes

    def __transform_bbox(self, bbox: list) -> dict:
        """
        Преобразование списка координат в словарь.
        Args:
            bbox (list): Исходный список.

        Returns:
            dict: Преобразованный словарь.
        """
        copy = bbox.copy()
        copy.sort()

        return {
            "lat1": min(copy[2], copy[3]),
            "lon1": min(copy[0], copy[1]),
            "lat2": max(copy[2], copy[3]),
            "lon2": max(copy[0], copy[1]),
        }

    def __save_main_route(self, save_folder: str) -> None:
        """Генерация и сохранение исходного маршрута в указанную папку в виде единого изображения,
        а также в виде отдельных частей размером 256х256.
        Получение изображения происходит с помощью запроса по протоколу TMS.

        Args:
            save_folder (str): Путь к папке для сохранения.
        """
        request = TMSRequest()
        main_route = self.__generate_main_route()
        self.main_routes.append(main_route)
        index = len(self.main_routes)
        bbox = self.__transform_bbox(self.__place_bbox)

        request.get(
            bbox,
            self.__graph,
            main_route,
            save_folder,
            f"main{index}",
            "png",
        )

    def __save_false_route(self, save_folder: str) -> None:
        """Генерация и сохранение искажённого маршрута в указанную папку в виде единого изображения,
        а также в виде отдельных частей размером 256х256.
        Получение изображения происходит с помощью запроса по протоколу TMS.

        Args:
            save_folder (str): Путь к папке для сохранения.
        """

        request = TMSRequest()
        index = len(self.main_routes)
        false_graph, false_route = self.__get_one_false_route(
            self.main_routes[index - 1]
        )
        self.false_routes.append((false_graph, false_route))
        bbox = self.__transform_bbox(self.__place_bbox)

        request.get(
            bbox,
            false_graph,
            false_route,
            save_folder,
            f"false{index}",
            "png",
        )

    def save_data(self, save_folder: str) -> None:
        """Генерация выборки с заранее указанным размером
        и сохранение изображений маршрутов в указанную папку.

        Args:
            save_folder (str): Путь к папке для сохранения.
        """

        for i in range(1, self.__data_amount + 1):
            route_folder = os.path.join(save_folder, f"route{i}")
            os.makedirs(route_folder, exist_ok=True)
            self.__save_main_route(route_folder)
            self.__save_false_route(route_folder)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    data_generator = DataGenerator(
        place_bbox=[39.121447, 51.646002, 39.135578, 51.653782]
    )
    data_generator.save_data("images")
