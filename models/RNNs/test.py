import os
import torch
import folium
from generators.RouteGenerator import RouteGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

    test_model = torch.load('lstm_model(1-100).pth', weights_only=False)
    test_model.eval()

    sc = MinMaxScaler(feature_range=(-1, 1))

    place_bbox = [39.0296, 51.7806, 39.3414, 51.5301]
    generator = RouteGenerator(place_bbox=place_bbox)
    G, result = generator.graph, generator.save_main_route()
    main_ids, main_coords = result
    G_false, false_coords = generator.save_false_route(main_ids)

    main_coords = generator.make_equal(main_coords, len(false_coords))

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
