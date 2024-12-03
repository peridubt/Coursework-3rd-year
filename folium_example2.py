import osmnx as ox
import networkx as nx

ox.config(use_cache=True, log_console=True)

G = ox.graph_from_point((-33.889606, 151.283306), dist=3000, network_type='drive')

G = ox.speed.add_edge_speeds(G)
G = ox.speed.add_edge_travel_times(G)

orig = ox.nearest_nodes(G, -33.889606, 151.283306)
dest = ox.nearest_nodes(G, -33.889927, 151.280497)
route = nx.shortest_path(G, orig, dest, 'travel_time')
route_map = ox.graph_to_gdfs(G, nodes=False).explore()
route_map.save('test.html')
