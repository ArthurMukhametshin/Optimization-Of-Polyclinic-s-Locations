# ----- Импорт библиотек -----

import os
import time
import warnings
import pandas as pd
import geopandas as gpd

# ----- Параметры -----

parking_polygons_shp = r'Data\parkings_pol.shp' # Путь к полигонам парковок
graph_nodes_shp = r'Data\modelData\roads_nodes_filtered.shp' # Путь к узлам графа
output_parking_points_shp = r'Data\modelData\parkings_nodes.shp' # Путь сохранения точек парковок

# Наименования полей
parking_id_col = 'fid'
node_id_col = 'NODE_ID'

# Используемая проекция
target_crs = "EPSG:32637"

# ----- Загрузка данных -----
start_time = time.time()

print("Загрузка данных")

gdf_parking_pol = gpd.read_file(parking_polygons_shp)
gdf_nodes = gpd.read_file(graph_nodes_shp)
print(f"  Загружено {len(gdf_parking_pol)} полигонов парковок")
print(f"  Загружено {len(gdf_nodes)} узлов графа")

# Проверка/создание ID парковок
if not gdf_parking_pol[parking_id_col].is_unique:
    print(f"  Исходный ID парковки '{parking_id_col}' не уникален")
    gdf_parking_pol = gdf_parking_pol.reset_index()
    parking_id_col_temp_unique = '_temp_unique_idx'
    gdf_parking_pol.rename(columns={'index': parking_id_col_temp_unique},
                           inplace=True)
else:
    gdf_parking_pol.set_index(parking_id_col, inplace=True)
    parking_id_col_temp_unique = None


# ----- Обработка системы координат -----

print("Обработка системы координат")

if str(gdf_parking_pol.crs).lower() != target_crs.lower():
    print(f"  Перепроецирование парковок в {target_crs}")
    gdf_parking_pol = gdf_parking_pol.to_crs(target_crs)

if str(gdf_nodes.crs).lower() != target_crs.lower():
    print(f"  Перепроецирование узлов графа в {target_crs}")
    gdf_nodes = gdf_nodes.to_crs(target_crs)

print(f"  CRS синхронизированы ({target_crs})")

# ----- Подготовка к поиску ближайших -----

print("Подготовка к поиску ближайших узлов")

# Удаление невалидных/пустых полигонов парковок и расчет центроидов
initial_parking_count_geom = len(gdf_parking_pol)
gdf_parking_pol = gdf_parking_pol[gdf_parking_pol.geometry.is_valid & ~gdf_parking_pol.geometry.is_empty].copy()
print(f"  Удалено {initial_parking_count_geom - len(gdf_parking_pol)} невалидных/пустых полигонов парковок")

print("  Расчет центроидов парковок")
gdf_parking_pol['centroid'] = gdf_parking_pol.geometry.centroid

# Удаление узлов графа без геометрии (на всякий случай)
gdf_nodes = gdf_nodes[~gdf_nodes.geometry.is_empty & gdf_nodes.geometry.notna()].copy()

print("  Создание пространственного индекса узлов графа")
nodes_sindex = gdf_nodes.sindex

# ----- Поиск ближайшего узла для каждой парковки -----
print("Поиск ближайшего узла графа для каждой парковки")

# Наиболее эффективный путь - использование sjoin_nearest
parking_centroids_gdf = gpd.GeoDataFrame(
    gdf_parking_pol[[parking_id_col_temp_unique if parking_id_col_temp_unique else parking_id_col]],
    geometry=gdf_parking_pol['centroid'],
    crs=target_crs
)
parking_centroids_gdf.index.name = parking_id_col_temp_unique if parking_id_col_temp_unique else parking_id_col

# Выполнение sjoin_nearest
gdf_joined = gpd.sjoin_nearest(parking_centroids_gdf, gdf_nodes[[node_id_col, 'geometry']],
                               how='left',
                               max_distance=None)

# Результат содержит дубликаты парковок, если несколько узлов одинаково близки.
# Оставляем только одну (первую найденную) связь для каждой парковки.
gdf_nearest = gdf_joined.loc[gdf_joined.groupby(gdf_joined.index).idxmin()['index_right']]

# ----- Создание точечного слоя точек доступа к парковкам -----

print("Создание точечного слоя точек доступа")

# Создание GeoDataFrame с использованием ID парковки и данных ближайшего узла
gdf_parking_access_points = gpd.GeoDataFrame(data={
    parking_id_col: gdf_nearest[parking_id_col].values if parking_id_col in gdf_nearest.columns else gdf_nearest.index, # Исходный ID парковки
    node_id_col: gdf_nearest[node_id_col].values                                                                        # ID связанного узла графа
    },
    geometry=gdf_nearest['geometry'].values # Геометрия узла графа
)

print(f"  Создано {len(gdf_parking_access_points)} точек доступа к парковкам")

# ----- Сохранение результата -----
print(f"Сохранение результата в {output_parking_points_shp}")

output_dir = os.path.dirname(output_parking_points_shp)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Конвертация типа данных перед сохранением
gdf_parking_access_points[node_id_col] = gdf_parking_access_points[node_id_col].astype(int)

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore",
                            category=UserWarning,
                            message="Column names longer than 10 characters will be truncated")
    gdf_parking_access_points.to_file(output_parking_points_shp,
                                      encoding='utf-8',
                                      driver='ESRI Shapefile')
print("  Результат успешно сохранен")

end_time = time.time()
print(f"Обработка парковок завершена за {end_time - start_time:.2f} секунд")