import psycopg2
import geopandas as gpd
import json
import os

db_properties = "..\\configs\\db_properties.json"  # адрес файла конфигурации БД
data_path = "..\\data\\training_data.csv"  # адрес сохранения файла с выборкой

dirname = os.path.dirname(__file__)
db_properties = os.path.join(dirname, db_properties)
data_path = os.path.join(dirname, data_path)

with open(db_properties) as f:
    db = json.load(f)

conn = psycopg2.connect(
    dbname=db["dbname"],
    user=db["user"],
    password=db["password"],
    host=db["host"],
    port=db["port"],
)

# Загрузка узлов
nodes_query = """
    SELECT
        osm_id,
        way
    FROM planet_osm_point
    WHERE highway IS NOT NULL;
"""
nodes_gdf = gpd.read_postgis(nodes_query, conn, geom_col="way")

# Загрузка ребер
edges_query = """
    SELECT
        l.osm_id AS line_id,
        p1.osm_id AS source,
        p2.osm_id AS target,
    ST_Length(l.way::geography) AS weight,
    l.way AS geometry  -- Добавляем геометрию
    FROM planet_osm_line l
    JOIN planet_osm_point p1 ON ST_Equals(ST_StartPoint(l.way), p1.way)
    JOIN planet_osm_point p2 ON ST_Equals(ST_EndPoint(l.way), p2.way);
"""
edges_gdf = gpd.read_postgis(edges_query, conn, geom_col="geometry")

conn.close()

nodes_gdf.to_file("nodes.geojson", driver="GeoJSON")
edges_gdf.to_file("edges.geojson", driver="GeoJSON")
