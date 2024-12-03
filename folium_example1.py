import osmnx as ox

ox.config(use_cache=True, log_console=True)

G = ox.graph_from_place('Munich, Germany', network_type='drive')
route1 = ox.shortest_path(G, 20461931, 75901933, weight=None)

orig = ox.nearest_nodes(G, X=11.58172095, Y=48.1336081)
dest = ox.nearest_nodes(G, X=11.53754219, Y=48.17822992)
route2 = ox.shortest_path(G, orig, dest, weight='travel_time')

route_map = ox.plot_route_folium(G, route1, route_color='#ff0000', opacity=0.5)
route_map = ox.plot_route_folium(G, route2, route_map=route_map, route_color='#0000ff', opacity=0.5)
route_map.save('route.html')
