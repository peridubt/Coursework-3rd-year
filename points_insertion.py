import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt


# 1. Получаем граф и находим кратчайший путь
G = ox.graph_from_place("Тепличный, Voronezh, Russia")
start_node = list(G.nodes)[0]
end_node = list(G.nodes)[-1]
shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight="length")

# 2. Добавление новых вершин и рёбер вдоль пути
# Пример добавления одной новой вершины между двумя узлами пути
new_node_id = max(G.nodes) + 1  # уникальный идентификатор для новой вершины
midpoint = ((G.nodes[shortest_path[3]]['y'] + G.nodes[shortest_path[4]]['y']) / 2 + 0.001,
            (G.nodes[shortest_path[3]]['x'] + G.nodes[shortest_path[4]]['x']) / 2)
newG = G.copy()
newG.add_node(new_node_id, y=midpoint[0], x=midpoint[1])  # добавляем новый узел на полпути между двумя узлами

# Добавляем рёбра для соединения нового узла с узлами пути
newG.add_edge(shortest_path[3], new_node_id, length=100, travel_time=5)  # ребро от одного узла к новому
newG.add_edge(new_node_id, shortest_path[4], length=100, travel_time=5)  # ребро от нового узла к следующему узлу

# 3. Модификация пути для включения новых узлов
modified_path = []
for i in range(len(shortest_path) - 1):
    modified_path.append(shortest_path[i])

    # Добавляем новый узел в путь между текущими узлами
    if i == 3:  # вставляем новый узел между узлами с индексами 3 и 4
        modified_path.append(new_node_id)
modified_path.append(shortest_path[-1])

# 4. Визуализация модифицированного пути на графе
fig, ax = ox.plot_graph_route(newG, modified_path)
plt.show()
