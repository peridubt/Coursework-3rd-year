import psycopg2
import geopandas as gpd
import json
import os

db_properties = "..\\configs\\db_properties.json"  # адрес файла конфигурации БД
data_path = "..\\training data\\training_data.csv"  # адрес сохранения файла с выборкой

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

conn.close()

nodes_gdf.to_csv(data_path, index=False)
