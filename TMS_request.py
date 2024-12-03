import math
from time import sleep
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import osmnx as ox
import networkx as nx
import os


class TSMRequest:
    def __init__(self):
        self.__tile_size = 256
        self.__zoom_size = 16
        self.__x1_tile = 0
        self.__y1_tile = 0
        self.__x2_tile = 0
        self.__y2_tile = 0
        self.__map_img = None

    @staticmethod
    def lat_lon_to_tile(lat, lon, zoom) -> (int, int):
        """Convert latitude and longitude to tile numbers."""
        n = 2**zoom
        x_tile = int((lon + 180) / 360 * n)
        y_tile = int(
            (1 - (math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))/ math.pi)) / 2* n
        )
        return x_tile, y_tile

    @staticmethod
    def tile_to_lat_lon(x, y, zoom) -> (float, float):
        """Convert tile numbers to latitude and longitude."""
        n = 2**zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return lat, lon

    @staticmethod
    def download_tile(x, y, zoom) -> Image.Image:
        """Download a tile image from the TMS server."""
        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        headers = {"User-Agent": "Chrome/58.0.3029.110"}
        sleep(1)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(
                f"Error downloading tile {x}/{y} at zoom {zoom}: {response.status_code}"
            )
            return None

    @staticmethod
    def geo_to_pixel(lat, lon, x_tile, y_tile, zoom) -> (int, int):
        n = 2.0**zoom
        x = (lon + 180.0) / 360.0 * n
        y = (
        (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)/ 2.0* n
        )
        pixel_x = (x - x_tile) * 256
        pixel_y = (y - y_tile) * 256
        return pixel_x, pixel_y

    def set_tiles_coords(self, lat1, lon1, lat2, lon2, zoom):
        self.__x1_tile, self.__y1_tile = self.lat_lon_to_tile(lat1, lon1, zoom)
        self.__x2_tile, self.__y2_tile = self.lat_lon_to_tile(lat2, lon2, zoom)

        # Create a blank image to paste tiles into

        width = int(math.fabs(self.__x2_tile - self.__x1_tile) + 1) * self.__tile_size
        height = int(math.fabs(self.__y2_tile - self.__y1_tile) + 1) * self.__tile_size
        self.__map_img = Image.new("RGB", (width, height))

        if self.__x1_tile > self.__x2_tile:
            self.__x1_tile, self.__x2_tile = self.__x2_tile, self.__x1_tile
        if self.__y1_tile > self.__y2_tile:
            self.__y1_tile, self.__y2_tile = self.__y2_tile, self.__y1_tile

    def fill_img_with_tiles(self, zoom):
        for x in range(self.__x1_tile, self.__x2_tile + 1):
            for y in range(self.__y1_tile, self.__y2_tile + 1):
                tile_image = self.download_tile(x, y, zoom)
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
        draw = ImageDraw.Draw(self.__map_img)
        for i in range(len(route_coords) - 1):
            x1, y1 = route_coords[i]
            x2, y2 = route_coords[i + 1]

            # Преобразование географических координат в пиксели
            x1_px, y1_px = self.geo_to_pixel(
                x1, y1, self.__x1_tile, self.__y1_tile, zoom
            )
            x2_px, y2_px = self.geo_to_pixel(
                x2, y2, self.__x1_tile, self.__y1_tile, zoom
            )
            draw.line([(x1_px, y1_px), (x2_px, y2_px)], fill="blue", width=3)
        os.makedirs(os.path.join(save_folder, name), exist_ok=True)
        self.__map_img.save(f"{save_folder}/{name}/{name}.{extension}")

    def split_img_into_tiles(self, save_folder: str, name: str, extension: str) -> None:
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

    def main(self, place_bbox: tuple, G: nx.Graph, route: list):
        lat1, lon1, lat2, lon2, zoom = (
            place_bbox[1],
            place_bbox[0],
            place_bbox[3],
            place_bbox[2],
            self.__zoom_size,
        )
        self.set_tiles_coords(lat1, lon1, lat2, lon2, zoom)
        self.fill_img_with_tiles(zoom)

        route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
        self.draw_route_on_map(route_coords, zoom, "images", "main", "png")
        self.split_img_into_tiles("images", "main", "png")


if __name__ == "__main__":
    G = ox.graph_from_place("Тепличный, Voronezh, Russia")
    place_bbox = ox.geocode_to_gdf("Тепличный, Voronezh, Russia").geometry.total_bounds
    keys = list(G.nodes.keys())
    orig = keys[0]  # Начальная точка (широта, долгота)
    dest = keys[-1]  # Конечная точка (широта, долгота)
    route = nx.astar_path(G, orig, dest, weight="length")
    tsm_request = TSMRequest()
    tsm_request.main(place_bbox, G, route)
