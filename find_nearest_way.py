import osmnx as ox
import networkx as nx
import overpy

# Задаем место, для которого хотим получить данные
place_name = "Тепличный, Voronezh, Russia"

# Загружаем граф улиц
G = ox.graph_from_place(place_name)

# Выбираем начальную и конечную точки (например, по координатам)
start_point = (51.626421, 39.068563)  # Координаты начала
end_point = (51.623256, 39.070580)  # Координаты конца

# Находим ближайшие узлы в графе к заданным координатам
start_node = ox.nearest_nodes(G, start_point[1], start_point[0])
end_node = ox.nearest_nodes(G, end_point[1], end_point[0])

# Находим кратчайший путь
shortest_path = nx.astar_path(G, start_node, end_node, weight='length')
print(shortest_path)

coordinates = {}

# Ищем координаты по ID
for node_id in shortest_path:
    if node_id in G.nodes:
        lat, lon = G.nodes[node_id]['y'], G.nodes[node_id]['x']
        coordinates[node_id] = (lat, lon)

coordinates = [x for x in coordinates.values()]
print(coordinates)

print(G.nodes)

# Визуализируем путь
ox.plot_graph_route(G, shortest_path)

