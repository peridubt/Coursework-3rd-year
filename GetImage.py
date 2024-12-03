from io import BytesIO

from PIL import Image
import requests
import time

def download_tile(x, y, zoom) -> Image.Image:
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


zoom = 6
x_tile, y_tile = 28, 5

# Шаг 3: Загрузка карты
tile = download_tile(x_tile, y_tile, zoom)
tile.save("map.png")
