import sys
import math
import requests
from PIL import Image
from io import BytesIO


def lat_lon_to_tile(lat, lon, zoom) -> (int, int):
    """Convert latitude and longitude to tile numbers."""
    n = 2 ** zoom
    x_tile = int((lon + 180) / 360 * n)
    y_tile = int((1 - (math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)) / 2 * n)
    return x_tile, y_tile


def tile_to_lat_lon(x, y, zoom) -> (float, float):
    """Convert tile numbers to latitude and longitude."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def download_tile(x, y, zoom):
    """Download a tile image from the TMS server."""
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Error downloading tile {x}/{y} at zoom {zoom}: {response.status_code}")
        return None


def main():
    # if len(sys.argv) != 6:
    #     print("Usage: python script.py <lat1> <lon1> <lat2> <lon2> <zoom>")
    #     sys.exit(1)

    # lat1, lon1, lat2, lon2, zoom = map(float, sys.argv[1:5]) + [int(sys.argv[5])]
    lat1, lon1, lat2, lon2, zoom = 51.63, 39.08, 51.64, 39.11, 7
    # Calculate tile coordinates for corners
    x1, y1 = lat_lon_to_tile(lat1, lon1, zoom)
    x2, y2 = lat_lon_to_tile(lat2, lon2, zoom)

    # Create a blank image to paste tiles into
    tile_size = 256
    width = (x2 - x1 + 1) * tile_size
    height = (y2 - y1 + 1) * tile_size
    map_image = Image.new('RGB', (width, height))

    # Download and place tiles in the image
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            tile_image = download_tile(x, y, zoom)
            if tile_image:
                map_image.paste(tile_image, ((x - x1) * tile_size, (y - y1) * tile_size))

    # Save the final image
    map_image.save("map.png")
    print("Map image saved as 'map.png'")


if __name__ == "__main__":
    main()
