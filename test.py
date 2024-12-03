from io import BytesIO

import networkx as nx
from PIL import Image, ImageDraw
import requests
import osmnx as ox
import math
import time


def geo_to_pixel(lat, lon, x_tile, y_tile, zoom):
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    pixel_x = (x - x_tile) * 256
    pixel_y = (y - y_tile) * 256
    return pixel_x, pixel_y


def lat_lon_to_tile(lat, lon, zoom) -> (int, int):
    """Convert latitude and longitude to tile numbers."""
    n = 2 ** zoom
    x_tile = int((lon + 180) / 360 * n)
    y_tile = int((1 - (math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)) / 2 * n)
    return x_tile, y_tile


def download_tile(x = 9971, y = 5437, zoom = 15) -> Image.Image:
    """Download a tile image from the TMS server."""
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    headers = {
        'User-Agent': 'Chrome/58.0.3029.110'
    }
    time.sleep(1)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Error downloading tile {x}/{y} at zoom {zoom}: {response.status_code}")
        return None


# Шаг 1: Получение маршрута
# G = ox.graph_from_place("Manhattan, New York, USA", network_type="walk")
# keys = list(G.nodes.keys())
# orig = keys[0]  # Начальная точка (широта, долгота)
# dest = keys[-1] # Конечная точка (широта, долгота)
# # route = nx.astar_path(G, orig, dest, weight="length")
# # route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
#
# # Шаг 2: Вычисление координат плитки
# zoom = 6
# x_tile, y_tile = lat_lon_to_tile(G.nodes[orig]["x"], G.nodes[orig]["y"], zoom)

# Шаг 3: Загрузка карты
tile = download_tile()
tile.show()
# # Шаг 4: Отображение маршрута на карте
# draw = ImageDraw.Draw(tile)
# for i in range(len(route_coords) - 1):
#     x1, y1 = route_coords[i]
#     x2, y2 = route_coords[i + 1]
#
#     # Преобразование географических координат в пиксели
#     x1_px, y1_px = geo_to_pixel(x1, y1, x_tile, y_tile, zoom)
#     x2_px, y2_px = geo_to_pixel(x2, y2, x_tile, y_tile, zoom)
#     draw.line([(x1_px, y1_px), (x2_px, y2_px)], fill="blue", width=3)
#
# tile.show()
