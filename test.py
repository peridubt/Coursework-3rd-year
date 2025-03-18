import os
import torch
import folium
from ImageGenerator import ImageGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from LSTMModel import LSTMModel


def make_equal(main_route: list, false_route: list) -> list:
    added_points = []
    main_copy = main_route.copy()
    pack = []

    for point in false_route:
        if point not in main_route:
            pack.append(point)
        else:
            added_points.append(pack)
            pack = []
    for j in range(len(main_route) - 1):
        if (size := len(added_points[j])) != 0:
            t = 1 / (size + 1)
            add = t
            start, end = main_route[j], main_route[j + 1]
            for _ in range(size):
                new_point = (start[0] + t * (end[0] - start[0]),
                             start[1] + t * (end[1] - start[1]))
                insert_idx = main_copy.index(start)
                main_copy.insert(insert_idx, new_point)
                t += add

    if len(main_copy) != len(false_route):
        main_copy.append(main_route[-1])
    return main_copy


def save_route(points: list, save_folder: str, name: str) -> None:
    # Создаем карту, центрированную на первой точке
    points = [(point[1], point[0]) for point in points]
    plot = folium.Map(location=points[0], zoom_start=15)

    # Соединяем точки линией (маршрут)
    folium.PolyLine(points, color="red", weight=2, opacity=1).add_to(plot)

    # Сохраняем карту в HTML-файл и открываем его
    plot.save(f"{save_folder}/{name}.html")


def lstm_test(save_folder: str) -> None:
    os.makedirs(save_folder, exist_ok=True)

    test_model = torch.load('lstm_model.pth', weights_only=False)
    test_model.eval()

    sc = MinMaxScaler(feature_range=(-1, 1))

    place_bbox = [39.16064, 51.72495, 39.18008, 51.71307]
    generator = ImageGenerator(place_bbox=place_bbox)
    G, main_route = generator.graph, generator.generate_main_route()
    G_false, false_route = generator.get_one_false_route(main_route)

    main_coords = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in main_route]
    false_coords = [[G_false.nodes[n]["x"], G_false.nodes[n]["y"]] for n in false_route]
    main_coords = make_equal(main_coords, false_coords)

    save_route(main_coords, save_folder, "target")
    save_route(false_coords, save_folder, "input")

    false_coords = torch.tensor(sc.fit_transform(false_coords), dtype=torch.float32)
    with torch.no_grad():
        predict = test_model(false_coords)
    predict = sc.inverse_transform(predict.detach().numpy())
    print(predict)
    save_route(predict, save_folder, "predict")
    print(f"MSE: {mean_squared_error(predict, main_coords)} \t MAE: {mean_absolute_error(predict, main_coords)}")


if __name__ == "__main__":
    lstm_test("images")
