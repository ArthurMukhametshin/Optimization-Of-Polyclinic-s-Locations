# ----- Импорт библиотек -----

import os
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from collections import Counter
from shapely.geometry import Point, LineString

# ----- Параметры -----

input_shapefile = 'Data/roads_planar_lin.shp' # Планарная дорожная сеть, очищенная от микроразрывов и висячих ребер
output_edges_shp = 'Data/roads_edges.shp' # Путь сохранения ребер графа
output_nodes_shp = 'Data/roads_nodes.shp' # Путь сохранения узлов графа
target_crs = 'EPSG:32637'  # Целевая проекция для расчета длин ребер - UTM zone 37N

# Наименования полей
ref_col = 'ref'
code_col = 'code'
name_col = 'name'
layer_col = 'layer'
bridge_col = 'bridge'
oneway_col = 'oneway'
tunnel_col = 'tunnel'
fclass_col = 'fclass'
maxspeed_col = 'maxspeed'

# Параметры обработки
create_nodes = True
clean_geometry = True
check_connectivity = True
default_speed_kmh = 30
pedestrian_speed_kmh = 5

# Словарь скоростей {fclass: maxspeed}
highway_default_speeds = {'motorway': 110, 'motorway_link': 110, 'trunk': 90, 'trunk_link': 90, 'primary': 70, 'primary_link': 70,
                          'secondary': 60, 'secondary_link': 60,'tertiary': 50, 'tertiary_link': 50, 'unclassified': 40,
                          'residential': 40, 'road': 40, 'service': 30, 'living_street': 20,'default': default_speed_kmh}

# Словарь строк в maxspeed, которые нужно преобразовать
maxspeed_string_map = {'ru:living_street': 20, 'ru:urban': 60, 'ru:rural': 90, 'ru:motorway': 110, 'walk': 5, 'none': 5}

# Типы дорог для исключения (значения из fclass_col)
exclude_highways = ['construction', 'proposed', 'raceway', 'abandoned', 'planned']

# Типы дорог, разрешенные для авто (значения из fclass_col)
car_allowed_highways = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary',
                        'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street',
                        'service', 'road']

# Типы дорог, разрешенные для пешеходов (значения из fclass_col)
ped_allowed_highways = ['primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link',
                        'unclassified', 'residential', 'living_street', 'service', 'road', 'pedestrian', 'footway',
                        'path', 'steps', 'track', 'bridleway', 'cycleway']

# Типы дорог, считающиеся строго пешеходными
strictly_ped_highways = ['pedestrian', 'footway', 'path', 'steps', 'track', 'bridleway', 'cycleway']

# Варианты написания NULL/None для обработки
null_variants = ['nan', 'none', '<na>', 'null', '', None]

# ----- Функции -----

def clean_maxspeed(speed_val, highway_type):
    """Очищение и преобразование значения MAXSPEED, используя дефолты. 0 считается как None"""

    if pd.isna(speed_val) or str(speed_val).lower() in null_variants or speed_val == 0 or str(speed_val) == '0':
        return highway_default_speeds.get(highway_type, highway_default_speeds['default'])

    s = str(speed_val).lower().strip()

    if s in maxspeed_string_map:
        return maxspeed_string_map[s]

    if ';' in s:
        speeds = [int(float(x.strip())) for x in s.split(';')]
        valid_speeds = [sp for sp in speeds if sp > 0]
        return min(valid_speeds) if valid_speeds else 1

    return highway_default_speeds.get(highway_type, highway_default_speeds['default'])

def reverse_geom(geom):
    """Разворот геометрии LineString"""

    if isinstance(geom, LineString) and geom.coords:
        return LineString(list(geom.coords)[::-1])

    return geom

# ----- Загрузка данных -----

start_time = time.time()

print(f"Загрузка данных из {input_shapefile}")

gdf_edges = gpd.read_file(input_shapefile)
print(f"  Загружено {len(gdf_edges)} ребер")
print(f"  Исходные колонки: {gdf_edges.columns.tolist()}")

required_input_cols = [fclass_col, oneway_col, maxspeed_col, bridge_col, tunnel_col]
missing_input_cols = [col for col in required_input_cols if col not in gdf_edges.columns]

# ----- Обработка системы координат -----

print("Обработка системы координат")

initial_crs = gdf_edges.crs
print(f"  Исходная СК: {initial_crs}")

if not initial_crs:
    print(f"  Исходная СК не определена. Установка целевой {target_crs}")
    gdf_edges.crs = target_crs
elif str(initial_crs).lower() != target_crs.lower():
    gdf_edges = gdf_edges.to_crs(target_crs)
else:
    print("  Данные уже в целевой СК")

# ----- Базовая очистка геометрии и топологии -----

if clean_geometry:
    print("Базовая очистка геометрии и топологии")

    initial_count = len(gdf_edges)
    gdf_edges = gdf_edges[~gdf_edges.geometry.is_empty & gdf_edges.geometry.notna() & (gdf_edges.geometry.geom_type == 'LineString')].copy()
    print(f"  Удалено {initial_count - len(gdf_edges)} пустых/нелинейных геометрий")

    initial_count = len(gdf_edges)
    duplicates = gdf_edges.geometry.duplicated(keep='first')
    gdf_edges = gdf_edges[~duplicates].copy()
    print(f"  Удалено {initial_count - len(gdf_edges)} дубликатов геометрии")

    initial_count = len(gdf_edges)
    self_loops = gdf_edges.geometry.apply(lambda g: g.coords[0] == g.coords[-1] if g.coords and len(g.coords)>1 else False)
    gdf_edges = gdf_edges[~self_loops].copy()
    print(f"  Удалено {initial_count - len(gdf_edges)} самопересекающихся линий")

    non_valid = gdf_edges[~gdf_edges.geometry.is_valid]
    if not non_valid.empty:
        print(f"  Найдено {len(non_valid)} невалидных геометрий после очистки!")
    else:
        print("  Проверка валидности геометрии пройдена")

    print(f"  Осталось {len(gdf_edges)} объектов после базовой очистки")

# ----- Фильтрация дорог по типу fclass -----

print(f"Фильтрация дорог по типу {fclass_col}...")

initial_count = len(gdf_edges)
gdf_edges[fclass_col] = gdf_edges[fclass_col].astype(str).str.lower()
gdf_edges = gdf_edges[~gdf_edges[fclass_col].isin(exclude_highways)].copy()
print(f"  Удалено {initial_count - len(gdf_edges)} объектов типов: {', '.join(exclude_highways)}. Осталось {len(gdf_edges)}")

# ----- Обработка атрибутов -----

print("Обработка атрибутов")

# Обработка ONEWAY
print(f"  Обработка поля {oneway_col}")
gdf_edges['oneway_orig'] = gdf_edges[oneway_col].astype(str).str.upper()
gdf_edges['ONEWAY'] = 'no'
gdf_edges.loc[gdf_edges['oneway_orig'].isin(['F', 'B']), 'ONEWAY'] = 'yes'
mask_motorway_trunk = gdf_edges[fclass_col].isin(['motorway', 'trunk', 'motorway_link', 'trunk_link'])
gdf_edges.loc[mask_motorway_trunk, 'ONEWAY'] = 'yes'

print("  Разворот геометрии для ONEWAY = 'B'")
reverse_mask = gdf_edges['oneway_orig'] == 'B'
num_reversed = reverse_mask.sum()
if num_reversed > 0:
    gdf_edges.loc[reverse_mask, 'geometry'] = gdf_edges.loc[reverse_mask, 'geometry'].apply(reverse_geom)
print(f"  Развернуто {num_reversed} геометрий")

# Установка ONEWAY='no' для пешеходных путей
print("  Принудительная установка ONEWAY='no' для пешеходных путей")
ped_mask = gdf_edges[fclass_col].isin(strictly_ped_highways)
num_ped_oneway = gdf_edges.loc[ped_mask & (gdf_edges['ONEWAY'] == 'yes')].shape[0]
gdf_edges.loc[ped_mask, 'ONEWAY'] = 'no'
print(f"  Установлено ONEWAY='no' для {ped_mask.sum()} пешеходных сегментов (из них {num_ped_oneway} были 'yes')")
print(f"  Итоговые значения ONEWAY: {dict(gdf_edges['ONEWAY'].value_counts())}")

# Обработка MAXSPEED
print(f"  Обработка поля {maxspeed_col} (0 считается как Null)")
gdf_edges['maxspeed_kph'] = gdf_edges.apply(lambda row: clean_maxspeed(row[maxspeed_col], row[fclass_col]),axis=1)
print(f"  Скорости обработаны. Средняя: {gdf_edges['maxspeed_kph'].mean():.1f} км/ч")

# Обработка BRIDGE/TUNNEL
print("  Создание флагов is_bridge, is_tunnel")

def check_flag(val):
    s = str(val).lower().strip()
    return s in ['1', 't', 'true', 'y', 'yes']

gdf_edges['is_bridge'] = gdf_edges[bridge_col].apply(check_flag)
gdf_edges['is_tunnel'] = gdf_edges[tunnel_col].apply(check_flag)

# ----- Расчет длины ребер -----

print(f"Расчет длины ребер (length_m)")

gdf_edges['length_m'] = gdf_edges.geometry.length
print(f"  Длина рассчитана. Общая длина сети: {gdf_edges['length_m'].sum() / 1000:.1f} км")

# ----- Добавление атрибутов для мультимодальности -----

print(f"Добавление атрибутов для мультимодальности")

print("  Флаги доступности (allow_car, allow_ped)")
gdf_edges['allow_car'] = gdf_edges[fclass_col].isin(car_allowed_highways)
gdf_edges['allow_ped'] = gdf_edges[fclass_col].isin(ped_allowed_highways)

print("  Расчет времени в пути (time_car_s, time_ped_s)")
speed_mps_car = gdf_edges['maxspeed_kph'] * 1000 / 3600
gdf_edges['time_car_s'] = np.where(gdf_edges['allow_car'] & (speed_mps_car > 0.1), gdf_edges['length_m'] / speed_mps_car, np.inf)

ped_speed_mps = pedestrian_speed_kmh * 1000 / 3600
gdf_edges['time_ped_s'] = np.where(gdf_edges['allow_ped'] & (gdf_edges['length_m'] > 0), gdf_edges['length_m'] / ped_speed_mps, np.inf)

print(f"  Среднее время авто (где доступно): {gdf_edges.loc[gdf_edges['time_car_s'] != np.inf, 'time_car_s'].mean():.1f} сек")
print(f"  Среднее время пешехода (где доступно): {gdf_edges.loc[gdf_edges['time_ped_s'] != np.inf, 'time_ped_s'].mean():.1f} сек")

# ----- Создание узлов и связывание ребер -----

print("Создание узлов и связывание ребер (v2 - через словарь координат)")

print("  Извлечение начальных и конечных точек ребер")
coord_precision = 6

def get_rounded_coords(geom):
    start = tuple(np.round(geom.coords[0], coord_precision))
    end = tuple(np.round(geom.coords[-1], coord_precision))
    return start, end

coords_list = gdf_edges.geometry.apply(get_rounded_coords)
start_coords_series = coords_list.apply(lambda x: x[0] if x else None)
end_coords_series = coords_list.apply(lambda x: x[1] if x else None)
initial_count_coords = len(gdf_edges)
valid_coords_mask = start_coords_series.notna() & end_coords_series.notna()
gdf_edges = gdf_edges[valid_coords_mask].copy()
start_coords_series = start_coords_series[valid_coords_mask]
end_coords_series = end_coords_series[valid_coords_mask]
print(f"  Удалено {initial_count_coords - len(gdf_edges)} ребер с некорректными координатами")

print(f"  Поиск уникальных координат узлов")
all_coords_series = pd.concat(objs=[start_coords_series, end_coords_series],
                              ignore_index=True)
unique_coords = all_coords_series.drop_duplicates().tolist()
print(f"  Найдено {len(unique_coords)} уникальных узлов")

print(f"  Создание GeoDataFrame узлов")
gdf_nodes = gpd.GeoDataFrame(data={'node_id': range(len(unique_coords))},
                             geometry=[Point(xy) for xy in unique_coords],
                             crs=target_crs)

print(f"  Создание словаря координат для ID узлов")
coord_to_nodeid_map = {coord: i for i, coord in enumerate(unique_coords)}

def get_node_id(coord_tuple, coord_map):
    return coord_map.get(coord_tuple, None)

print("  Привязка ребер к ID узлов через словарь (может занять время)")
gdf_edges['start_node'] = start_coords_series.apply(get_node_id, args=(coord_to_nodeid_map,))
gdf_edges['end_node'] = end_coords_series.apply(get_node_id, args=(coord_to_nodeid_map,))
missing_start = gdf_edges['start_node'].isnull().sum()
missing_end = gdf_edges['end_node'].isnull().sum()

if missing_start > 0 or missing_end > 0:
    print(f"  Обнаружены пропуски в ID узлов (start: {missing_start}, end: {missing_end})")
    initial_count = len(gdf_edges)
    gdf_edges.dropna(subset=['start_node', 'end_node'],
                     inplace=True)
    gdf_edges['start_node'] = gdf_edges['start_node'].astype(int)
    gdf_edges['end_node'] = gdf_edges['end_node'].astype(int)
    print(f"  Удалено {initial_count - len(gdf_edges)} ребер с отсутствующими узлами")
else:
    gdf_edges['start_node'] = gdf_edges['start_node'].astype(int)
    gdf_edges['end_node'] = gdf_edges['end_node'].astype(int)
    print("  ID узлов успешно добавлены к ребрам")

# ----- Анализ связности и присвоение Component ID -----
gdf_edges['component_id'] = -1
gdf_nodes['component_id'] = -1

if check_connectivity:

    print("Анализ связности графа")

    print("  Построение графа NetworkX")
    G = nx.Graph()
    edges_for_nx = gdf_edges[['start_node', 'end_node']].dropna().astype(int).values.tolist()
    G.add_edges_from(edges_for_nx)
    all_node_ids = gdf_nodes['node_id'].unique()
    G.add_nodes_from(all_node_ids)
    print(f"  Граф построен: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер")

    print("  Поиск компонент связности")
    components = list(nx.connected_components(G))
    num_components = len(components)
    print(f"  Найдено {num_components} компонент связности")

    # Анализ размеров компонент
    component_sizes = [len(c) for c in components]
    component_size_counts = Counter(component_sizes)

    print("  Распределение размеров компонент (число узлов: количество компонент):")
    top_counts = component_size_counts.most_common(5)
    large_components = {size: count for size, count in component_size_counts.items() if size > 10}
    print(f"    Топ-5 частых размеров: {top_counts}")
    print(f"    Крупные компоненты (>10 узлов): {large_components}")

    # Создание словаря {node_id: component_id}
    print("  Создание словаря node_id -> component_id")
    node_to_component = {}
    comp_id_counter = 0
    nodes_in_components = set()

    for comp_nodes in components:
        if comp_nodes:
            for node_id in comp_nodes:
                node_to_component[node_id] = comp_id_counter
                nodes_in_components.add(node_id)
            comp_id_counter += 1
    print(f"  Словарь создан для {len(node_to_component)} узлов")

    # Проверка, все ли узлы из gdf_nodes попали в карту
    nodes_not_in_map = set(all_node_ids) - nodes_in_components
    if nodes_not_in_map:
         print(f"  Внимание: {len(nodes_not_in_map)} узлов из gdf_nodes не найдены в компонентах NetworkX!")

    # Добавление component_id к ребрам (по start_node)
    print("  Присвоение component_id ребрам")
    gdf_edges['component_id'] = gdf_edges['start_node'].map(node_to_component).fillna(-1).astype(int)

    # Присвоение component_id узлам
    print("  Присвоение component_id узлам")
    gdf_nodes['component_id'] = gdf_nodes['node_id'].map(node_to_component).fillna(-1).astype(int)

    # Проверка результата
    assigned_node_count = (gdf_nodes['component_id'] != -1).sum()
    print(f"  ID компоненты присвоен {assigned_node_count} / {len(gdf_nodes)} узлам")

    print("  ID компонент присвоены ребрам и узлам")

# ----- Финальная подготовка и сохранение -----

print(f"Финальная подготовка и сохранение")

# Подготовка ребер
print("  Подготовка ребер для сохранения")

edge_cols_to_keep = ['length_m', 'maxspeed_kph', 'time_car_s', 'time_ped_s', 'component_id', 'ONEWAY', fclass_col,
                     'allow_car', 'allow_ped', 'is_bridge', 'is_tunnel', 'start_node', 'end_node']

for col in [name_col, ref_col, layer_col, code_col]:
     if col in gdf_edges.columns:
         edge_cols_to_keep.append(col)

gdf_edges_to_save = gdf_edges[[col for col in edge_cols_to_keep if col in gdf_edges.columns] + ['geometry']].copy()
edge_rename_dict = {'length_m': 'LEN_M', 'maxspeed_kph': 'SPEED_KPH', 'time_car_s': 'TIME_CAR_S', 'time_ped_s': 'TIME_PED_S',
                    'allow_car': 'ALLOW_CAR', 'allow_ped': 'ALLOW_PED', 'is_bridge': 'IS_BRIDGE', 'is_tunnel': 'IS_TUNNEL',
                    'start_node': 'START_N', 'end_node': 'END_N', 'component_id': 'COMP_ID', fclass_col: 'FCLASS',
                    name_col: 'NAME', ref_col: 'REF', layer_col: 'LAYER', code_col: 'CODE'}

gdf_edges_to_save.rename(columns={k: v for k, v in edge_rename_dict.items() if k in gdf_edges_to_save.columns},
                         inplace=True)
cols_to_drop_final = [oneway_col, maxspeed_col, bridge_col, tunnel_col, 'oneway_orig']
gdf_edges_to_save.drop(columns=[col for col in cols_to_drop_final if col in gdf_edges_to_save.columns],
                       errors='ignore',
                       inplace=True)

print("  Конвертация типов данных ребер")

for col in ['ALLOW_CAR', 'ALLOW_PED', 'IS_BRIDGE', 'IS_TUNNEL', 'COMP_ID']:
    if col in gdf_edges_to_save.columns:
        gdf_edges_to_save[col] = gdf_edges_to_save[col].astype(int)

for col in ['TIME_CAR_S', 'TIME_PED_S']:
    if col in gdf_edges_to_save.columns:
        gdf_edges_to_save[col] = gdf_edges_to_save[col].replace([np.inf, -np.inf], 99999999).astype(float)

if 'LEN_M' in gdf_edges_to_save.columns:
    gdf_edges_to_save['LEN_M'] = gdf_edges_to_save['LEN_M'].astype(float)

if 'SPEED_KPH' in gdf_edges_to_save.columns:
    gdf_edges_to_save['SPEED_KPH'] = gdf_edges_to_save['SPEED_KPH'].astype(int)

if 'START_N' in gdf_edges_to_save.columns:
    gdf_edges_to_save['START_N'] = gdf_edges_to_save['START_N'].astype(int)

if 'END_N' in gdf_edges_to_save.columns:
    gdf_edges_to_save['END_N'] = gdf_edges_to_save['END_N'].astype(int)

for col in ['FCLASS', 'ONEWAY', 'NAME', 'REF', 'CODE']:
     if col in gdf_edges_to_save.columns:
         gdf_edges_to_save[col] = gdf_edges_to_save[col].astype(str)

if 'LAYER' in gdf_edges_to_save.columns:
    gdf_edges_to_save['LAYER'] = pd.to_numeric(gdf_edges_to_save['LAYER'], errors='coerce').fillna(0).astype(int)

output_dir = os.path.dirname(output_edges_shp)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"  Сохранение ребер в {output_edges_shp}")
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore",
                            category=UserWarning,
                            message="Column names longer than 10 characters will be truncated")
    gdf_edges_to_save.to_file(output_edges_shp,
                              encoding='utf-8',
                              driver='ESRI Shapefile')
print("  Сохранение ребер завершено")

# --- Подготовка и сохранение узлов ---
if create_nodes and gdf_nodes is not None:

    print("  Подготовка узлов для сохранения")
    gdf_nodes_to_save = gdf_nodes.copy()
    node_rename_dict = {'node_id': 'NODE_ID', 'component_id': 'COMP_ID'}
    gdf_nodes_to_save.rename(columns={k:v for k,v in node_rename_dict.items() if k in gdf_nodes_to_save.columns},
                             inplace=True)

    print("  Конвертация типов данных узлов")
    if 'NODE_ID' in gdf_nodes_to_save.columns:
        gdf_nodes_to_save['NODE_ID'] = gdf_nodes_to_save['NODE_ID'].astype(int)
    if 'COMP_ID' in gdf_nodes_to_save.columns:
        gdf_nodes_to_save['COMP_ID'] = gdf_nodes_to_save['COMP_ID'].astype(int)

    # Сохранение узлов

    output_dir = os.path.dirname(output_nodes_shp)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"  Сохранение узлов в {output_nodes_shp}")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore",
                                category=UserWarning,
                                message="Column names longer than 10 characters will be truncated")
        gdf_nodes_to_save.to_file(output_nodes_shp, encoding='utf-8', driver='ESRI Shapefile')
    print("  Сохранение узлов завершено")

end_time = time.time()
print(f"Подготовка дорожного графа завершена за {end_time - start_time:.2f} секунд")