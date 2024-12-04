import random
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic as gd
import json
from TMS_request import TMSRequest


class DataGenerator:
    def __init__(self, config_path: str = "config.json", **kwargs):
        self.__load_config(config_path)
        self.main_route = []  # основной маршрут, содержит id вершин
        self.false_routes = []  # список сгенерированных "ложных" маршрутов

        # border box, задаёт границы для исследуемой местности
        # через координаты противоположных углов

        if "place_name" in kwargs.keys():
            self.__place_bbox = sorted(list(ox.geocode_to_gdf(
                kwargs["place_name"]
            ).geometry.total_bounds), reverse=True)

        elif "place_bbox" in kwargs.keys():
            self.__place_bbox = kwargs["place_bbox"]
        else:
            raise Exception(
                "Укажите название места согласно базе данных OSM либо координаты местности."
            )

        self.__graph = ox.graph_from_bbox(*self.__place_bbox, network_type="drive")  # граф местности

    def __load_config(self, file_path: str) -> None:
        with open(file_path, "r") as file:
            config = json.load(file)
            self.__data_amount = config["data_amount"]
            self.__min_segment = config["min_segment"]
            self.__max_segment = config["max_segment"]
            self.__min_offset = config["min_offset"]
            self.__max_offset = config["max_offset"]

    def __generate_main_route(self) -> None:
        # Выбор случайных точек в графе
        keys = list(self.__graph.nodes.keys())
        start = random.choice(keys)
        keys.remove(start)
        end = random.choice(keys)

        # Поиск кратчайшего пути с помощью алгоритма А*
        self.main_route = nx.astar_path(self.__graph, start, end, weight="length")

    def __get_one_false_route(self) -> ("nx.Graph", list):
        path = self.main_route
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

    # Метод, генерирующий ложные маршруты
    def generate_false_routes(self) -> None:
        for _ in range(self.__data_amount):
            self.false_routes.append(self.__get_one_false_route())

    def save_main_route(self, save_folder: str) -> None:
        if len(self.main_route) == 0:
            self.__generate_main_route()
        request = TMSRequest()
        request.get(self.__place_bbox, self.__graph, self.main_route,
                    save_folder, "main", "png")

    def save_false_route(self, index: int, save_folder: str) -> None:
        if len(self.false_routes) == 0:
            raise Exception(
                "Для получения определённого ложного маршрута требуется генерация."
            )
        if index < 1 or index > len(self.false_routes):
            raise Exception("Нет маршрута с заданным номером.")
        request = TMSRequest()
        false_graph, false_route = self.false_routes[index - 1]
        request.get(self.__place_bbox, false_graph, false_route,
                    save_folder, f"false_route{index}", "png")


if __name__ == "__main__":
    data_generator = DataGenerator(place_name="Тепличный, Voronezh, Russia")
    data_generator.save_main_route("images")
    data_generator.generate_false_routes()
    data_generator.save_false_route(2, "images")
