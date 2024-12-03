import osmnx as ox, networkx as nx
from IPython.display import IFrame

ox.config(log_console=True, use_cache=True)
G = ox.graph_from_place('Piedmont, California, USA', network_type='drive')

# use networkx to calculate the shortest path between two nodes
origin_node = list(G.nodes())[0]
destination_node = list(G.nodes())[-1]
route = nx.shortest_path(G, origin_node, destination_node)

# plot the route with folium
route_map = ox.plot_route_folium(G, route)

# save as html file then display map as an iframe
filepath = 'piedmont.html'
route_map.save(filepath)
IFrame(filepath, width=600, height=500)

