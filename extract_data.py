import geopandas as gpd
import osmnx as ox

# Определение координаты интересующей области
north, south, east, west = 51.64, 51.63, 39.11, 39.08

bbox = [north, south, east, west]
# Загрузка карты в виде графа
graph = ox.graph_from_bbox(*bbox, network_type='drive')

ox.plot_graph(graph)
nodes, edges = ox.graph_to_gdfs(graph)

