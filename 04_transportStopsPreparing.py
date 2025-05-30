# ----- Импорт библиотек -----

import os
import time
import warnings
import pandas as pd
import geopandas as gpd

# ----- Параметры -----

gtfs_stops_file = r'Data\GTFS\stops.txt' # Остановки ОТ
graph_nodes_shp = r'Data\modelData\roads_nodes_filtered.shp' # Узлы графа
graph_edges_shp = r'Data\modelData\roads_edges_filtered.shp' # Ребра графа
output_stop_nodes_shp = r'Data\modelData\transportStops_nodes.shp' # Путь сохранения узлов остановок

# Имена полей в GTFS stops.txt
gtfs_stop_id_col = 'stop_id'
gtfs_stop_name_col = 'stop_name'
gtfs_stop_lat_col = 'stop_lat'
gtfs_stop_lon_col = 'stop_lon'
gtfs_location_type_col = 'location_type'
gtfs_parent_station_col = 'parent_station'

# Имена полей в узлах/ребрах графа
node_id_col = 'NODE_ID'
edge_start_node_col = 'START_N'
edge_end_node_col = 'END_N'
edge_allow_car_col = 'ALLOW_CAR'

# Система координат
target_crs = "EPSG:32637"
gtfs_crs = "EPSG:4326"

# Параметры
include_stations = True
allowed_location_types = [0, 1] if include_stations else [0]

# ----- Загрузка данных -----

start_time = time.time()
print("Фаза 1: подготовка узлов остановок ОТ")

print("Загрузка данных")

df_stops_gtfs = pd.read_csv(gtfs_stops_file,
                            low_memory=False,
                            dtype={gtfs_stop_id_col: str})
gdf_nodes = gpd.read_file(graph_nodes_shp)
gdf_edges = gpd.read_file(graph_edges_shp)

print(f"  Загружено {len(df_stops_gtfs)} записей из stops.txt")
print(f"  Загружено {len(gdf_nodes)} узлов графа")
print(f"  Загружено {len(gdf_edges)} ребер графа")

# ----- Создание GeoDataFrame из GTFS остановок и их фильтрация -----

print("Создание GeoDataFrame остановок и их фильтрация")

gdf_stops = gpd.GeoDataFrame(df_stops_gtfs,
                             geometry=gpd.points_from_xy(df_stops_gtfs[gtfs_stop_lon_col], df_stops_gtfs[gtfs_stop_lat_col]),
                             crs=gtfs_crs)
gdf_stops.dropna(subset=[gtfs_stop_lat_col, gtfs_stop_lon_col, 'geometry'],
                 inplace=True)
print(f"  Создан GeoDataFrame с {len(gdf_stops)} остановками")

# Фильтрация остановок по location_type
initial_count = len(gdf_stops)
gdf_stops[gtfs_location_type_col] = gdf_stops[gtfs_location_type_col].fillna(0)
gdf_stops = gdf_stops[gdf_stops[gtfs_location_type_col].isin(allowed_location_types)].copy()
print(f"  Отфильтровано по location_type: осталось {len(gdf_stops)} остановок")

# Удаление дубликатов по stop_id
if gdf_stops[gtfs_stop_id_col].duplicated().any():
    print(f"  Найдены дубликаты в '{gtfs_stop_id_col}'! Удаляем, оставляя первое вхождение")
    gdf_stops.drop_duplicates(subset=[gtfs_stop_id_col],
                              keep='first',
                              inplace=True)

# Установка stop_id как индекса
gdf_stops.set_index(gtfs_stop_id_col, inplace=True)
print(f"  Итого {len(gdf_stops)} уникальных остановок для обработки")

# ----- Обработка CRS -----

print("Обработка системы координат")

if str(gdf_stops.crs).lower() != target_crs.lower():
    print("  Перепроецирование остановок")
    gdf_stops = gdf_stops.to_crs(target_crs)

if str(gdf_nodes.crs).lower() != target_crs.lower():
    print("  Перепроецирование узлов")
    gdf_nodes = gdf_nodes.to_crs(target_crs)

if str(gdf_edges.crs).lower() != target_crs.lower():
    print("  Перепроецирование ребер")
    gdf_edges = gdf_edges.to_crs(target_crs)

print(f"  CRS синхронизированы ({target_crs})")

# ----- Привязка остановок к узлам дорожного графа -----

print("Привязка остановок к узлам дорожного графа")

gdf_nodes = gdf_nodes[~gdf_nodes.geometry.is_empty & gdf_nodes.geometry.notna()].copy()
nodes_sindex = gdf_nodes.sindex

# Привязка к основному графу
print("Привязка к ближайшему узлу основной сети")
gdf_stops_for_sjoin = gdf_stops[['geometry', gtfs_stop_name_col]]
gdf_nodes_for_sjoin = gdf_nodes[[node_id_col, 'geometry']]
gdf_stop_nodes_linked = gpd.sjoin_nearest(gdf_stops_for_sjoin,
                                          gdf_nodes_for_sjoin,
                                          how='left',
                                          max_distance=None,
                                          distance_col='dist_walk_node')

# Обработка дубликатов sjoin_nearest
gdf_stop_nodes_linked = gdf_stop_nodes_linked.loc[gdf_stop_nodes_linked.groupby(level=0)['dist_walk_node'].idxmin()]
gdf_stop_nodes_linked.rename(columns={node_id_col: 'NODE_ID_WK'},
                             inplace=True)
gdf_stop_nodes_linked.drop(columns=['index_right'],
                           inplace=True,
                           errors='ignore')

missing_walk_links = gdf_stop_nodes_linked['NODE_ID_WK'].isnull().sum()
if missing_walk_links > 0:
    print(f"  Внимание: {missing_walk_links} остановок не привязано к основному графу! Удаляем их")
    gdf_stop_nodes_linked.dropna(subset=['NODE_ID_WK'],
                                 inplace=True)
    gdf_stop_nodes_linked['NODE_ID_WK'] = gdf_stop_nodes_linked['NODE_ID_WK'].astype(int)
else:
    gdf_stop_nodes_linked['NODE_ID_WK'] = gdf_stop_nodes_linked['NODE_ID_WK'].astype(int)

print(f"  Привязано {len(gdf_stop_nodes_linked)} остановок к основному графу")

# Привязка к авто-сети
print(f"Привязка к ближайшему узлу авто-сети")
car_nodes_ids = set(gdf_edges.loc[gdf_edges[edge_allow_car_col] == 1, edge_start_node_col]) \
              | set(gdf_edges.loc[gdf_edges[edge_allow_car_col] == 1, edge_end_node_col])
gdf_nodes_car_only = gdf_nodes[gdf_nodes[node_id_col].isin(car_nodes_ids)].copy()

nodes_car_sindex = gdf_nodes_car_only.sindex
gdf_stops_linked_car = gpd.sjoin_nearest(
    gdf_stop_nodes_linked[['geometry']],
    gdf_nodes_car_only[[node_id_col, 'geometry']],
    how='left',
    max_distance=None,
    distance_col='dist_to_car_node'
)
gdf_stops_linked_car = gdf_stops_linked_car.loc[gdf_stops_linked_car.groupby(level=0)['dist_to_car_node'].idxmin()]

# Добавление результата как новой колонки, используя индекс stop_id
gdf_stop_nodes_linked['NODE_ID_CR'] = gdf_stops_linked_car[node_id_col]
gdf_stop_nodes_linked['dist_car_node'] = gdf_stops_linked_car['dist_to_car_node']
print(f"  Найдена привязка к авто-сети для {gdf_stop_nodes_linked['NODE_ID_CR'].notna().sum()} остановок")

gdf_stop_nodes_linked['NODE_ID_CR'] = gdf_stop_nodes_linked['NODE_ID_CR'].astype('Int64')

# ----- Подготовка итогового слоя -----

print("Подготовка итогового слоя")

# Добавление недостающих колонок из исходного gdf_stops
cols_to_add_from_gtfs = [gtfs_location_type_col]

if gtfs_parent_station_col in gdf_stops.columns:
    cols_to_add_from_gtfs.append(gtfs_parent_station_col)

cols_to_add_existing = [col for col in cols_to_add_from_gtfs if col in gdf_stops.columns]

if cols_to_add_existing:
     gdf_final_stops = gdf_stop_nodes_linked.join(gdf_stops[cols_to_add_existing], how='left')
else:
     gdf_final_stops = gdf_stop_nodes_linked.copy()

# Сброс индекса, чтобы stop_id стал колонкой
gdf_final_stops.reset_index(inplace=True)

# Определение словаря переименования: {Старое имя: Новое имя}
rename_dict = {gtfs_stop_id_col: 'stop_id',
               gtfs_stop_name_col: 'NAME',
               gtfs_location_type_col: 'TYPE',
               gtfs_parent_station_col: 'PARENT_STA',
               'NODE_ID_WALK': 'NODE_ID_WK',
               'dist_walk_node': 'DIST_WK_ND',
               'NODE_ID_CAR': 'NODE_ID_CR',
               'dist_car_node': 'DIST_CR_ND'}

# Отбор только тех колонок, которые есть сейчас и нужны для переименования
cols_to_rename_present = {k: v for k, v in rename_dict.items() if k in gdf_final_stops.columns}
gdf_final_stops.rename(columns=cols_to_rename_present,
                       inplace=True)

# Определение финального списка колонок (по новым именам)
final_columns_order = ['stop_id', 'NAME', 'TYPE', 'PARENT_STA','NODE_ID_WK', 'DIST_WK_ND', 'NODE_ID_CR', 'DIST_CR_ND','geometry']

# Оставление только тех колонок из списка, которые реально существуют в датафрейме
final_columns_existing = [col for col in final_columns_order if col in gdf_final_stops.columns]

# Создание итогового GeoDataFrame с нужными колонками в правильном порядке
gdf_to_save = gdf_final_stops[final_columns_existing].copy()


# Конвертация типов (с использованием новых имен)
print("  Конвертация типов данных")

# Проверка наличия перед конвертацией
if 'stop_id' in gdf_to_save.columns:
    gdf_to_save['stop_id'] = gdf_to_save['stop_id'].astype(str)

if 'NODE_ID_WK' in gdf_to_save.columns:
    gdf_to_save['NODE_ID_WK'] = gdf_to_save['NODE_ID_WK'].astype(int)

if 'NODE_ID_CR' in gdf_to_save.columns:
    gdf_to_save['NODE_ID_CR'] = gdf_to_save['NODE_ID_CR'].astype('Int64')

if 'DIST_WK_ND' in gdf_to_save.columns:
    gdf_to_save['DIST_WK_ND'] = gdf_to_save['DIST_WK_ND'].round(3)

if 'DIST_CR_ND' in gdf_to_save.columns:
    mask_notna = gdf_to_save['DIST_CR_ND'].notna()
    gdf_to_save.loc[mask_notna, 'DIST_CR_ND'] = gdf_to_save.loc[mask_notna, 'DIST_CR_ND'].round(3)

if 'TYPE' in gdf_to_save.columns:
    gdf_to_save['TYPE'] = gdf_to_save['TYPE'].fillna(0).astype(int)

for col in ['NAME', 'PARENT_STA']:
    if col in gdf_to_save.columns:
        gdf_to_save[col] = gdf_to_save[col].fillna('').astype(str)

# ----- Сохранение результата -----

print(f"Сохранение результата в {output_stop_nodes_shp}")

output_dir = os.path.dirname(output_stop_nodes_shp)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"  Финальные колонки для сохранения: {gdf_to_save.columns.tolist()}")

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore",
                            category=UserWarning,
                            message="Column names longer than 10 characters will be truncated")
    gdf_to_save.to_file(output_stop_nodes_shp,
                        encoding='utf-8',
                        driver='ESRI Shapefile')

print(f"  Результат успешно сохранен")

end_time_phase1 = time.time()
print(f"Фаза 1 завершена за {end_time_phase1 - start_time:.2f} секунд")