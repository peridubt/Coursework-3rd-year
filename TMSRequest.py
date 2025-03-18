import math
from time import sleep
from typing import Tuple
import requests
from PIL import Image, ImageDraw  # type: ignore
from io import BytesIO
import networkx as nx
import os


class TMSRequest:
    def __init__(self):
        self.__tile_size = 256  # Размер одной "плитки" - 256 пикселей
        self.__zoom_size = 15  # Параметр масштабирования карты - 15

        # Координаты верхней правой плитки
        self.__x1_tile = 0
        self.__y1_tile = 0

        # Координаты нижней левой плитки
        self.__x2_tile = 0
        self.__y2_tile = 0

        self.__map_img = None  # Полное изображение карты

    @staticmethod
    def __lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Конвертация из географических координат в координаты согласно протоколу TMS.

        Args:
            lat (float): Ширина.
            lon (float): Долгота.
            zoom (int): Параметр масштабирования.

        Returns:
            Tuple[int, int]: Координаты согласно TMS.
        """

        n = 2 ** zoom
        x_tile = int((lon + 180) / 360 * n)
        y_tile = int(
            (
                    1
                    - (
                            math.log(
                                math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))
                            )
                            / math.pi
                    )
            )
            / 2
            * n
        )
        return x_tile, y_tile

    @staticmethod
    def __download_tile(x: int, y: int, zoom: int) -> Image.Image:
        """Загрузка одной плитки из сервера OSM.

        Args:
            x (int): Координата плитки по Х.
            y (int): Координата плитки по У.
            zoom (int): Параметр масштабирования.

        Raises:
            Exception: Ответ от сервера не был получен либо был получен некорректно.

        Returns:
            Image.Image: Изображение в виде объекта библиотеки Pillow.
        """

        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        headers = {"User-Agent": "Chrome/58.0.3029.110"}
        sleep(1)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(
                f"Error downloading tile {x}/{y} at zoom {zoom}: {response.status_code}"
            )

    @staticmethod
    def __geo_to_pixel(
            lat: float, lon: float, x_tile: int, y_tile: int, zoom: int
    ) -> Tuple[int, int]:
        """Преобразование географических координат в пиксели на экране согласно размерам изображения.

        Args:
            lat (float): Ширина точки.
            lon (float): Долгота точки.
            x_tile (int): Координата плитки Х.
            y_tile (int): Координата плитки У.
            zoom (int): Параметр машстабирования.

        Returns:
            Tuple[int, int]: Координаты точки на экране.
        """

        n = 2.0 ** zoom
        x = (lon + 180.0) / 360.0 * n
        y = (
                (
                        1.0
                        - math.log(
                    math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))
                )
                        / math.pi
                )
                / 2.0
                * n
        )
        pixel_x = (x - x_tile) * 256
        pixel_y = (y - y_tile) * 256
        return int(pixel_x), int(pixel_y)

    def set_tiles_coords(
            self, lat1: float, lon1: float, lat2: float, lon2: float, zoom: int
    ):
        """
        Получение координат для двух плиток, находящихся в противоположных углах карты,
        на основе координат противоположных точек, составляющих местность.

        Args:
            lat1 (float): Ширина 1-й точки.
            lon1 (float): Долгота 1-й точки.
            lat2 (float): Ширина 2-й точки.
            lon2 (float): Долгота 2-й точки.
            zoom (int): Параметр масштабирования.
        """

        self.__x1_tile, self.__y1_tile = self.__lat_lon_to_tile(lat1, lon1, zoom)
        self.__x2_tile, self.__y2_tile = self.__lat_lon_to_tile(lat2, lon2, zoom)

        # Создание пустого изображения для вставки более мелких изображений

        width = int(math.fabs(self.__x2_tile - self.__x1_tile) + 1) * self.__tile_size
        height = int(math.fabs(self.__y2_tile - self.__y1_tile) + 1) * self.__tile_size
        self.__map_img = Image.new("RGB", (width, height))

        self.__x1_tile, self.__x2_tile = min(self.__x1_tile, self.__x2_tile), max(
            self.__x1_tile, self.__x2_tile
        )
        self.__y1_tile, self.__y2_tile = min(self.__y1_tile, self.__y2_tile), max(
            self.__y1_tile, self.__y2_tile
        )

    def fill_img_with_tiles(self, zoom: int):
        """Заполнение полного изображения плитками, полученными с помощью запроса TMS.

        Args:
            zoom (int): Параметр масштабирования.
        """

        for x in range(self.__x1_tile, self.__x2_tile + 1):
            for y in range(self.__y1_tile, self.__y2_tile + 1):
                tile_image = self.__download_tile(x, y, zoom)
                if tile_image:
                    self.__map_img.paste(
                        tile_image,
                        (
                            (x - self.__x1_tile) * self.__tile_size,
                            (y - self.__y1_tile) * self.__tile_size,
                        ),
                    )

    def draw_route_on_map(
            self, route_coords: list, zoom: int, save_folder: str, name: str, extension: str
    ):
        """Отрисовка указанного маршрута на изображении карты
        и сохранение полученного изображения.

        Args:
            route_coords (list): Маршрут, представленный списком координат.
            zoom (int): Параметр масштабирования.
            save_folder (str): Путь к папке для сохранения.
            name (str): Название файла для сохранения.
            extension (str): Расширение файла.
        """

        draw = ImageDraw.Draw(self.__map_img)
        for i in range(len(route_coords) - 1):
            x1, y1 = tuple(route_coords[i])
            x2, y2 = tuple(route_coords[i + 1])

            x1_px, y1_px = self.__geo_to_pixel(
                x1, y1, self.__x1_tile, self.__y1_tile, zoom
            )
            x2_px, y2_px = self.__geo_to_pixel(
                x2, y2, self.__x1_tile, self.__y1_tile, zoom
            )
            draw.line([(x1_px, y1_px), (x2_px, y2_px)], fill="blue", width=3)
        self.__map_img.save(f"{save_folder}/{name}.{extension}")

    def __split_img_into_tiles(
            self, save_folder: str, name: str, extension: str
    ) -> None:
        """Разделение изображения карты на плитки и сохранение в указанную папку.

        Args:
            save_folder (str): Путь к папке для сохранения.
            name (str): Название файла для сохранения.
            extension (str): Расширение файла.
        """

        os.makedirs(save_folder, exist_ok=True)
        img = Image.open(os.path.join(save_folder, name, f"{name}.{extension}"))
        img_width, img_height = img.size

        for i in range(0, img_width, self.__tile_size):
            for j in range(0, img_height, self.__tile_size):
                box = (i, j, i + self.__tile_size, j + self.__tile_size)
                cropped = img.crop(box)
                cropped.save(
                    os.path.join(save_folder, name, f"{name}_{i}_{j}.{extension}")
                )

    def get(
            self,
            geo_coords: dict,
            route_coords: list,
            save_folder: str,
            route_name: str,
            extension: str,
    ):
        """Объединение предыдущих методов в единый запрос.

        Args:
            geo_coords (dict): Координаты местности.
            route_coords (list): Маршрут для отрисовки в виде списка идентификаторов вершин.
            save_folder (str): Путь к папке для сохранения.
            route_name (str): Название файла для сохранения.
            extension (str): Расширение файла.
        """

        lat1, lon1, lat2, lon2 = (
            geo_coords["lat1"],
            geo_coords["lon1"],
            geo_coords["lat2"],
            geo_coords["lon2"],
        )

        zoom = self.__zoom_size
        self.set_tiles_coords(lat1, lon1, lat2, lon2, zoom)
        self.fill_img_with_tiles(zoom)
        self.draw_route_on_map(route_coords, zoom, save_folder, route_name, extension)
        # self.__split_img_into_tiles(save_folder, route_name, extension)


if __name__ == "__main__":
    pass
