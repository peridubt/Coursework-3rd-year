import geopandas as gpd
import osmnx as ox

# Разные примеры местностей

# 1. Дачный посёлок Анжелики:
# north, south, east, west = 51.64, 51.63, 39.11, 39.08
# bbox = [north, south, east, west]
# graph = ox.graph_from_bbox(*bbox, network_type='drive')

# 2. Тепличный, Советский район:
graph = ox.graph_from_place("Тепличный, Voronezh, Russia")

# ox.plot_graph(graph)
nodes, edges = ox.graph_to_gdfs(graph)

extracted = nodes.loc[[1029821343]]
extracted["y"] = 60
nodes.loc[[1029821343]] = extracted
graph = ox.graph_from_gdfs(nodes, edges)
ox.plot_graph(graph)
