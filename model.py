# ----- Импорт библиотек -----

import os
import gc
import time
import pulp
import warnings
import traceback
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import rtree
rtree_available = True
from tqdm import tqdm
tqdm_available = True
from multiprocessing import Pool, cpu_count
from pysal.explore import inequality
inequality_available = True

# ----- Параметры -----

print("----- Загрузка настроек и параметров -----")
start_global_time = time.time()

# Пути к файлам
points_shp_path = r'Data\modelData\popHouses.shp' # Путь к файлу спроса - точкам жилых зданий
graph_edges_shp = r'Data\modelData\roads_edges_filtered.shp' # Путь к файлу ребер графа
graph_nodes_all_shp = r'Data\modelData\roads_nodes_filtered.shp' # Путь к файлу узлов графа
candidate_nodes_thinned_path = r'Data\modelData\roads_nodes_filtered_candidates_byDegreeCentrality.shp' # Путь к файлу кандидатов
output_dir = r'Data\modelData\results' # Директория для сохранения результатов

# Выходные файлы
optimal_locations_gpkg = os.path.join(output_dir, 'optimal_polyclinics.gpkg') # Путь для сохранения локаций поликлиник
metrics_csv = os.path.join(output_dir, 'optimization_metrics.csv') # Путь для сохранения метрик точности
demand_allocation_gpkg = os.path.join(output_dir, 'demand_allocation.gpkg') # Путь для сохранения приписанных жилых зданий

# Параметры модели
n_polyclinics = 263 # Количество поликлиник
target_crs = "EPSG:32637" # Целевая СК

# Параметры доступности
walk_speed_mps = 5 * 1000 / 3600 # Скорость пешехода - 5 км/ч
coverage_time_threshold_sec = 20 * 60
max_euclidean_distance_meters = 10000 # Максимальное евклидово расстояние для фильтрации задач

# Параметры пост-анализа
work_days_per_year = 252 # Количество рабочих дней в году
shifts_per_day = 2 # Количество рабочих смен в день
area_per_visit_sqm = 10 # Площадь участка поликлиники (кв.м) на 1 посещение в смену
min_area_sqm = 2000 # Минимальная площадь участка
number_of_visits_per_capita_per_year = 10.6214 # Количество посещений поликлиники на 1 человека в год

pop_to_visits_per_shift_coeff = number_of_visits_per_capita_per_year / work_days_per_year / shifts_per_day # Количество посещений поликлиники на 1 человека в смену

# Параметры производительности
max_demand_points = None # None для обработки всех точек спроса
num_processes = 19 # Количество параллельных процессов

# Имена колонок
points_pop_col = 'population'
points_id_col = 'id' # Или agg_id для агрегированных данных о населении
node_id_col = 'NODE_ID'
edge_start_n_col = 'START_N'
edge_end_n_col = 'END_N'
edge_time_ped_col = 'TIME_PED_S'
edge_time_car_col = 'TIME_CAR_S'
edge_allow_car_col = 'ALLOW_CAR'
edge_oneway_col = 'ONEWAY'

# ----- Функции -----

print("----- Инициализация Функций -----")

def calculate_walk_time(geom1, geom2, walk_speed):
    """Расчет времени пешком по прямому расстоянию"""

    if geom1 is None or geom2 is None or walk_speed <= 0:
        return np.inf

    try:
        return geom1.distance(geom2) / walk_speed
    except Exception:
        return np.inf

def calculate_path_time(G, source_node, target_node, weight='weight'):
    """Расчет времени пути в графе NetworkX"""

    try:
        source_node, target_node = int(source_node), int(target_node)
    except (ValueError, TypeError):
        return np.inf

    if source_node == target_node:
        return 0

    if not G.has_node(source_node) or not G.has_node(target_node):
        return np.inf

    try:
        return nx.dijkstra_path_length(G,
                                       source=source_node,
                                       target=target_node,
                                       weight=weight)

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return np.inf
    except Exception:
        return np.inf

def find_nearest_node(point_geom, nodes_gdf, nodes_sindex):
    """Находит ID ближайшего узла графа к точке. Возвращает скалярный ID или None"""

    if nodes_gdf is None or nodes_sindex is None or point_geom is None or nodes_gdf.empty:
        return None

    try:
        nearest_idx_or_array = nodes_sindex.nearest(point_geom)
        if isinstance(nearest_idx_or_array, (np.ndarray, list, tuple)) and len(nearest_idx_or_array) > 0:
            idx_label = nearest_idx_or_array[0]
        elif isinstance(nearest_idx_or_array, (int, np.integer, np.int32, np.int64)):
            idx_label = nearest_idx_or_array
        else:
            return None

        if isinstance(idx_label, (int, np.integer, np.int32, np.int64)):
            return idx_label
        else:
            try:
                scalar_id = idx_label[0]
                if isinstance(scalar_id, (int, np.integer, np.int32, np.int64)):
                    return scalar_id
                else:
                    return None
            except:
                return None

    except AttributeError:
        print("  [find_nearest_node] Ошибка: sindex не имеет метода 'nearest'. Установите/обновите rtree")
        return None
    except Exception:
        return None

def calculate_access_time_single(task_args):
    """Воркер-функция для расчета времени доступа (пешком/авто)"""

    i, j_iloc = task_args
    data = worker_data
    if not data or data.get('error', False): return i, j_iloc, np.inf, None
    if 'gdf_demand' not in data:
        return i, j_iloc, np.inf, None

    origin_row = data['gdf_demand'].iloc[i]
    origin_geom = origin_row.geometry
    origin_walk_node_id = origin_row['nearest_node_id']

    dest_candidate_row = data['gdf_candidates'].iloc[j_iloc]
    dest_node_id = dest_candidate_row.name
    dest_geom = dest_candidate_row.geometry

    origin_car_node_id = None
    if data.get('gdf_nodes_car_only') is not None and data.get('nodes_car_sindex') is not None:
         origin_car_node_id = find_nearest_node(origin_geom, data['gdf_nodes_car_only'], data['nodes_car_sindex'])

    if pd.isna(origin_walk_node_id):
        return i, j_iloc, np.inf, None

    results = {}
    try:
        dest_node_id_int = int(dest_node_id)
    except (ValueError, TypeError):
        dest_node_id_int = None

    # 1. Пешком
    time_path_walk = np.inf
    if dest_node_id_int is not None:
        time_path_walk = calculate_path_time(data['G_walk'], origin_walk_node_id, dest_node_id_int, weight=data['edge_time_ped_col'])
    results['walk'] = time_path_walk

    # 2. Авто
    t_car = np.inf
    if origin_car_node_id is not None and data['G_car'] is not None and dest_node_id_int is not None:
        try:
            origin_car_node_id_int = int(origin_car_node_id)
        except (ValueError, TypeError):
            origin_car_node_id_int = None

        if origin_car_node_id_int is not None:
            if data['G_car'].has_node(origin_car_node_id_int) and data['G_car'].has_node(dest_node_id_int):
                time_car_path = calculate_path_time(data['G_car'], origin_car_node_id_int, dest_node_id_int, weight=data['edge_time_car_col'])
                t_car = time_car_path if pd.notna(time_car_path) else np.inf
    results['car'] = t_car

    min_time = np.inf
    best_mode = None
    for mode, time_val in results.items():
         if pd.notna(time_val) and time_val < min_time:
             min_time = time_val
             best_mode = mode
    if min_time == np.inf:
        best_mode = None

    return i, j_iloc, min_time, best_mode

def calculate_metrics(gdf_demand_assigned, gdf_facilities, pop_to_visits_coeff, points_pop_col):
    """Рассчитывает метрики качества и добавляет атрибуты к поликлиникам."""

    print("Расчет метрик качества")

    metrics = {}
    valid_demand = gdf_demand_assigned[(gdf_demand_assigned['access_time_sec'] != np.inf) & pd.notna(gdf_demand_assigned['access_time_sec']) & (gdf_demand_assigned['facility_node_id'] != -1) & pd.notna(gdf_demand_assigned['facility_node_id'])].copy()
    if not valid_demand.empty:
        assigned_times_finite = valid_demand['access_time_sec'].values
        if points_pop_col not in valid_demand.columns:
            population_weights_finite = np.ones(len(valid_demand))
        else:
            valid_demand[points_pop_col] = pd.to_numeric(valid_demand[points_pop_col], errors='coerce').fillna(0)
            population_weights_finite = valid_demand[points_pop_col].values
        total_pop_assigned = np.sum(population_weights_finite)
        print(f"  Метрики считаются по {len(valid_demand)} домам, население: {total_pop_assigned:,.0f}")
        if total_pop_assigned > 0 and len(assigned_times_finite) > 0:
            metrics['avg_access_time_sec'] = np.average(assigned_times_finite, weights=population_weights_finite)
            metrics['max_access_time_sec'] = np.max(assigned_times_finite); metrics['pop_total_assigned'] = total_pop_assigned
            metrics['pop_covered_within_threshold'] = np.sum(population_weights_finite[assigned_times_finite <= coverage_time_threshold_sec])
            metrics['perc_pop_covered'] = (metrics['pop_covered_within_threshold'] / total_pop_assigned) * 100 if total_pop_assigned > 0 else 0
            if len(assigned_times_finite) > 1: metrics['std_dev_access_time_sec'] = np.sqrt(np.average((assigned_times_finite - metrics['avg_access_time_sec'])**2,weights=population_weights_finite))
            else:
                metrics['std_dev_access_time_sec'] = 0
            try:
                if inequality_available and len(np.unique(assigned_times_finite)) > 1:
                    metrics['gini_access_time'] = inequality.gini.Gini(assigned_times_finite, w=population_weights_finite).g
                else:
                    metrics['gini_access_time'] = np.nan
            except Exception:
                metrics['gini_access_time'] = np.nan
        else:
            metrics.update({k: 0 for k in ['avg_access_time_sec',
                                           'max_access_time_sec',
                                           'pop_total_assigned',
                                           'pop_covered_within_threshold',
                                           'perc_pop_covered',
                                           'std_dev_access_time_sec']})
            metrics['gini_access_time'] = np.nan
        mode_counts = valid_demand['access_mode'].value_counts()
        mode_pop_sum = valid_demand.groupby('access_mode')[points_pop_col].sum()
        total_pop_for_mode_perc = mode_pop_sum.sum()
        mode_stats = pd.DataFrame({'  Доля домов (%)': mode_counts / len(valid_demand) * 100 if len(valid_demand) > 0 else 0, 'Доля населения (%)': mode_pop_sum / total_pop_for_mode_perc * 100 if total_pop_for_mode_perc > 0 else 0}).fillna(0)
        print("  Статистика по моде доступа:")
        print(mode_stats.round(2).to_string())
        metrics['mode_stats_house_perc'] = mode_stats['Доля домов (%)'].to_dict()
        metrics['mode_stats_pop_perc'] = mode_stats['Доля населения (%)'].to_dict()
    else:
        print("  Нет доступных домов для метрик!")
        metrics.update({k: 0 for k in ['avg_access_time_sec',
                                       'max_access_time_sec',
                                       'pop_total_assigned',
                                       'pop_covered_within_threshold',
                                       'perc_pop_covered',
                                       'std_dev_access_time_sec']})
        metrics['gini_access_time'] = np.nan
        metrics['mode_stats_house_perc'] = {}
        metrics['mode_stats_pop_perc'] = {}

    if 'load_persons' in gdf_facilities.columns and not gdf_facilities.empty:
        facility_load = gdf_facilities['load_persons'].astype(float).fillna(0).values
        if len(facility_load) > 0 and facility_load.sum() > 0:
            metrics['avg_load_persons'] = np.mean(facility_load)
            metrics['std_dev_load_persons'] = np.std(facility_load)
            metrics['cv_load_persons'] = (metrics['std_dev_load_persons'] / metrics['avg_load_persons']) if metrics['avg_load_persons'] > 0 else np.nan
            try:
                 if inequality_available and len(np.unique(facility_load)) > 1:
                     metrics['gini_load'] = inequality.gini.Gini(facility_load).g
                 else:
                     metrics['gini_load'] = np.nan
            except Exception:
                metrics['gini_load'] = np.nan
        else:
            metrics.update({k: 0 for k in ['avg_load_persons', 'std_dev_load_persons', 'cv_load_persons', 'gini_load']})
            metrics['gini_load'] = np.nan
    else:
        metrics.update({k: 0 for k in ['avg_load_persons', 'std_dev_load_persons', 'cv_load_persons', 'gini_load']})
        metrics['gini_load'] = np.nan
    if 'load_persons' in gdf_facilities.columns:
        print("  Расчет мощности и площади")
        gdf_facilities['visits_per_shift'] = (gdf_facilities['load_persons'] * pop_to_visits_coeff).round().astype(int)
        gdf_facilities['required_area_sqm'] = np.maximum(min_area_sqm, gdf_facilities['visits_per_shift'] * area_per_visit_sqm).astype(int)
    else:
        print("  Колонка 'load_persons' отсутствует")
        gdf_facilities['visits_per_shift'] = 0
        gdf_facilities['required_area_sqm'] = min_area_sqm
    return metrics, gdf_facilities, gdf_demand_assigned

worker_data = {}

# Глобальная переменная для данных воркера
worker_data = {}

def init_worker(init_args):
    """
    Инициализатор воркера для параллельных вычислений.
    Загружает данные графа (узлы, ребра), строит графы NetworkX (пешеходный и автомобильный),
    фильтрует их по крупнейшим компонентам связности, получает данные о кандидатах и спросе
    (переданные как GeoDataFrame), и сохраняет все необходимое в глобальный словарь worker_data.
    """
    global worker_data
    pid = os.getpid()  # Получение ID текущего процесса для логов
    print(f"Инициализация воркера {pid}")
    start_init_time = time.time()

    # ----- 1. Распаковка аргументов -----
    (nodes_shp, edges_shp, gdf_candidates_from_args, demand_data_minimal_from_args, target_crs, node_id_col, edge_start_n_col,
     edge_end_n_col, edge_time_ped_col, edge_time_car_col, edge_allow_car_col, edge_oneway_col, points_pop_col, walk_speed_mps) = init_args

    worker_data = {'error': False}  # Инициализация статуса ошибки (False - нет ошибки)

    try:
        # ----- 2. Загрузка всех узлов графа из файла -----
        gdf_nodes_all = gpd.read_file(nodes_shp)

        # Базовая обработка узлов
        gdf_nodes_all[node_id_col] = gdf_nodes_all[node_id_col].astype('int32')
        gdf_nodes_all.dropna(subset=['geometry', node_id_col],
                             inplace=True)
        gdf_nodes_all = gdf_nodes_all.drop_duplicates(subset=[node_id_col],
                                                      keep='first')

        # Установка индекса и проверка/установка CRS
        gdf_nodes_all.set_index(node_id_col,
                                inplace=True,
                                drop=False)
        if str(gdf_nodes_all.crs).lower() != target_crs.lower():
            print(f"  [init {pid}] Приведение CRS узлов к {target_crs}")
            gdf_nodes_all = gdf_nodes_all.to_crs(target_crs)

        # ----- 3. Загрузка ребер графа из файла -----
        gdf_edges = gpd.read_file(edges_shp)

        # Проверка/установка CRS ребер
        if str(gdf_edges.crs).lower() != target_crs.lower():
            print(f"  [init {pid}] Приведение CRS ребер к {target_crs}")
            gdf_edges = gdf_edges.to_crs(target_crs)

        # Преобразование типов колонок ребер и обработка пропусков/ошибок
        gdf_edges[edge_time_ped_col] = pd.to_numeric(gdf_edges[edge_time_ped_col], errors='coerce').fillna(np.inf).astype('float32')
        gdf_edges[edge_time_car_col] = pd.to_numeric(gdf_edges[edge_time_car_col], errors='coerce').fillna(np.inf).astype('float32')
        gdf_edges[edge_allow_car_col] = pd.to_numeric(gdf_edges[edge_allow_car_col], errors='coerce').fillna(0).astype('uint8')
        gdf_edges[edge_start_n_col] = pd.to_numeric(gdf_edges[edge_start_n_col], errors='coerce').fillna(-1).astype('int32')
        gdf_edges[edge_end_n_col] = pd.to_numeric(gdf_edges[edge_end_n_col], errors='coerce').fillna(-1).astype('int32')

        # Очистка геометрии ребер
        gdf_edges.dropna(subset=['geometry'],
                         inplace=True)
        gdf_edges = gdf_edges[gdf_edges.geometry.notna() & gdf_edges.geometry.geom_type.isin(['LineString', 'MultiLineString'])] # Допускание MultiLineString

        # Фильтрация ребер: оставление только тех, чьи узлы есть в gdf_nodes_all
        valid_nodes_set = set(gdf_nodes_all.index)
        edge_node_mask = gdf_edges[edge_start_n_col].isin(valid_nodes_set) & gdf_edges[edge_end_n_col].isin(valid_nodes_set)
        gdf_edges = gdf_edges[edge_node_mask].copy()

        # Удаление ребер, бесполезных для обоих режимов (бесконечное время)
        gdf_edges = gdf_edges[ (gdf_edges[edge_time_ped_col] < np.inf) | (gdf_edges[edge_time_car_col] < np.inf) ].copy()

        # ----- 4. Построение исходных графов NetworkX -----
        G_walk_orig = nx.Graph()
        G_car_orig = nx.DiGraph() # Автомобильный граф - ориентированный
        all_node_ids_list = gdf_nodes_all.index.tolist() # Получение списка ID узлов из индекса

        # Добавление всех валидных узлов в оба графа
        G_walk_orig.add_nodes_from(all_node_ids_list)
        G_car_orig.add_nodes_from(all_node_ids_list)

        # Добавление ребер для пешеходного графа (неориентированный)
        edges_walk_df = gdf_edges[[edge_start_n_col, edge_end_n_col, edge_time_ped_col]]

        # Удаление ребер с бесконечным временем пешком
        edges_walk_df = edges_walk_df[edges_walk_df[edge_time_ped_col] < np.inf]

        # Добавление ребер с весом
        G_walk_orig.add_weighted_edges_from(edges_walk_df.values, weight=edge_time_ped_col)

        # Добавление ребер для автомобильного графа (ориентированный)
        weight_attr_car = edge_time_car_col # Имя атрибута веса ребра

        # Выбор ребер, доступных для авто и с конечным временем
        edges_car_df = gdf_edges[(gdf_edges[edge_allow_car_col] == 1) & (gdf_edges[edge_time_car_col] < np.inf)
        ][[edge_start_n_col, edge_end_n_col, edge_time_car_col, edge_oneway_col]]

        # Итеративное добавление ребер в G_car с учетом одностороннего движения
        for row in edges_car_df.itertuples(index=False):
             u, v, time_car, oneway_val = row[0], row[1], row[2], row[3]
             oneway_str = str(oneway_val).lower().strip()

             # Добавление ребер в прямом направлении (u -> v)
             if G_car_orig.has_node(u) and G_car_orig.has_node(v):
                 G_car_orig.add_edge(u, v, **{weight_attr_car: time_car})

                 # Добавление ребер в обратном направлении (v -> u), если движение не одностороннее
                 if oneway_str not in ['yes', 'true', '1', 't', 'y', '-1']:
                      if G_car_orig.has_node(v) and G_car_orig.has_node(u):
                          G_car_orig.add_edge(v, u, **{weight_attr_car: time_car})

        # Очистка памяти от временных DataFrame ребер
        del gdf_edges, edges_walk_df, edges_car_df, all_node_ids_list
        gc.collect()

        # ----- 5. Фильтрация G_walk по крупнейшей компоненте связности -----
        if G_walk_orig.number_of_nodes() > 0 and not nx.is_connected(G_walk_orig):
            print(f"  Воркер {pid}: G_walk не связный. Фильтрация по крупнейшей компоненте")
            largest_cc_walk_nodes = max(nx.connected_components(G_walk_orig), key=len)

            # Создание подграфа только с узлами из крупнейшей компоненты
            G_walk = G_walk_orig.subgraph(largest_cc_walk_nodes).copy()
            print(f"  [init {pid}] G_walk отфильтрован: {G_walk.number_of_nodes()} узлов, {G_walk.number_of_edges()} ребер")
            del largest_cc_walk_nodes
        else:
            # Если граф уже связный или пустой, используем его как есть
            G_walk = G_walk_orig
            status = "связный" if G_walk.number_of_nodes() > 0 else "пустой"
            print(f"  [init {pid}] G_walk {status}, используется как есть")
        # Сохраняем итоговый граф в данные воркера
        worker_data['G_walk'] = G_walk
        del G_walk_orig  # Удаляем исходный граф G_walk_orig
        gc.collect()

        # ----- 6. Фильтрация G_car по крупнейшей компоненте СЛАБОЙ связности -----
        gdf_nodes_car_only = gpd.GeoDataFrame() # Инициализируем пустой GDF для узлов авто-графа
        car_nodes_ids_set = set() # Множество ID узлов итогового G_car

        if G_car_orig.number_of_nodes() > 0:
            if not nx.is_weakly_connected(G_car_orig):
                print(f"  Воркер {pid}: G_car не связный (слабо). Фильтрация по крупнейшей компоненте")
                largest_cc_car_nodes = max(nx.weakly_connected_components(G_car_orig), key=len)

                # Создаем подграф только с узлами из крупнейшей компоненты
                G_car = G_car_orig.subgraph(largest_cc_car_nodes).copy()
                car_nodes_ids_set = largest_cc_car_nodes # Сохраняем узлы компоненты
                print(f"  [init {pid}] G_car отфильтрован: {G_car.number_of_nodes()} узлов, {G_car.number_of_edges()} ребер")
                del largest_cc_car_nodes
            else:
                # Если граф уже слабо связный, используем его как есть
                G_car = G_car_orig
                print(f"  [init {pid}] G_car слабо связный, используется как есть")
                car_nodes_ids_set = set(G_car.nodes()) # Все узлы графа

            # Создаем GeoDataFrame с геометрией узлов, входящих в итоговый G_car
            gdf_nodes_car_only = gdf_nodes_all.loc[list(car_nodes_ids_set), ['geometry']].copy() # Используем .loc для выбора по индексу
        else:
            # Если исходный G_car был пуст
            print(f"  [init {pid}] Исходный G_car пуст")
            G_car = G_car_orig # Оставляем пустой граф

        # Сохраняем итоговый граф и GeoDataFrame его узлов
        worker_data['G_car'] = G_car
        worker_data['gdf_nodes_car_only'] = gdf_nodes_car_only

        # Создаем пространственный индекс для узлов G_car
        if not gdf_nodes_car_only.empty and rtree_available:
            try:
                 worker_data['nodes_car_sindex'] = gdf_nodes_car_only.sindex
                 print(f"  [init {pid}] Простр. индекс для {len(gdf_nodes_car_only)} авто-узлов успешно создан")
            except Exception as e_sindex:
                 print(f"  [init {pid}] Ошибка создания sindex для авто-узлов: {e_sindex}")
                 worker_data['nodes_car_sindex'] = None
        else:
            worker_data['nodes_car_sindex'] = None # Индекс не создан
            if not gdf_nodes_car_only.empty and not rtree_available:
                print(f"  [init {pid}] Предупреждение: rtree не найден, пространственный индекс для авто-узлов не создан")
            elif gdf_nodes_car_only.empty:
                 print(f"  [init {pid}] Нет авто-узлов для создания индекса")

        del G_car_orig, gdf_nodes_car_only # Удаляем исходный граф G_car_orig и временный GDF
        if 'car_nodes_ids_set' in locals():
            del car_nodes_ids_set
        gc.collect()

        # ----- 7. Получение данных кандидатов (из аргументов) -----
        print(f"  Воркер {pid}: получение данных кандидатов из аргументов")

        # Используем GeoDataFrame, переданный напрямую
        gdf_candidates = gdf_candidates_from_args

        # Проверка полученных данных кандидатов
        if not isinstance(gdf_candidates, gpd.GeoDataFrame):
             raise TypeError(f"  Ожидался GeoDataFrame для кандидатов, получен {type(gdf_candidates)}")
        if gdf_candidates.empty:
             # Это может быть нормально, если в основном процессе все отфильтровались, но расчет тогда будет бессмысленным
             print(f"  [init {pid}] Получен пустой GeoDataFrame кандидатов.")
        if 'geometry' not in gdf_candidates.columns:
             raise ValueError("  В переданном GeoDataFrame кандидатов отсутствует колонка 'geometry'")

        # Проверка и установка CRS
        if str(gdf_candidates.crs).lower() != target_crs.lower():
             print(f"  [init {pid}] Предупреждение: CRS кандидатов ({gdf_candidates.crs}) отличается от целевого ({target_crs}). Приведение к целевому CRS")
             gdf_candidates = gdf_candidates.to_crs(target_crs)

        # Критическая проверка индекса: он используется как dest_node_id
        if not isinstance(gdf_candidates.index, pd.Index) or gdf_candidates.index.name != node_id_col:
             raise ValueError(f"  Переданный GeoDataFrame кандидатов должен иметь индекс с именем '{node_id_col}'. Текущий индекс: {gdf_candidates.index}")

        # Сохраняем в данные воркера только геометрию, но с правильным индексом
        worker_data['gdf_candidates'] = gdf_candidates[['geometry']] # Индекс копируется вместе со срезом
        print(f"  Воркер {pid}: данные кандидатов ({len(worker_data['gdf_candidates'])} строк) с индексом '{worker_data['gdf_candidates'].index.name}' успешно обработаны")

        # ----- 8. Получение данных спроса (из аргументов) -----
        print(f"  Воркер {pid}: получение данных спроса из аргументов")

        # Используем GeoDataFrame, переданный напрямую
        worker_data['gdf_demand'] = demand_data_minimal_from_args

        # Проверка полученных данных спроса
        if not isinstance(worker_data['gdf_demand'], gpd.GeoDataFrame):
             raise TypeError(f"  Ожидался GeoDataFrame для спроса, получен {type(worker_data['gdf_demand'])}")

        if worker_data['gdf_demand'].empty:
             print(f"  [init {pid}] Получен пустой GeoDataFrame спроса")

        # Проверяем наличие необходимых колонок
        required_demand_cols = ['geometry', 'nearest_node_id']
        if not all(col in worker_data['gdf_demand'].columns for col in required_demand_cols):
             raise ValueError(f"  В переданном GeoDataFrame спроса отсутствуют необходимые колонки: {required_demand_cols}"
                              f"  Найдены: {list(worker_data['gdf_demand'].columns)}")
        print(f"  Воркер {pid}: данные спроса ({len(worker_data['gdf_demand'])} строк) успешно получены")

        # ----- 9. Сохранение констант/параметров в данные воркера -----
        worker_data['WALK_SPEED_MPS'] = walk_speed_mps
        worker_data['edge_time_ped_col'] = edge_time_ped_col
        worker_data['edge_time_car_col'] = edge_time_car_col

        # ----- 10. Финальная проверка содержимого worker_data -----
        required_keys = ['gdf_demand', 'gdf_candidates', 'G_walk', 'G_car', 'gdf_nodes_car_only', 'nodes_car_sindex']
        missing_keys = []
        empty_data = []
        null_data = []

        for k in required_keys:
            if k not in worker_data:
                missing_keys.append(k)
                continue

            # Проверка на None (кроме sindex, который может быть None)
            if worker_data[k] is None and k != 'nodes_car_sindex':
                 null_data.append(k)

            # Проверка на пустоту для DataFrame/GeoDataFrame
            elif isinstance(worker_data[k], (pd.DataFrame, gpd.GeoDataFrame)) and worker_data[k].empty and k != 'gdf_nodes_car_only':
                 empty_data.append(k)

            # Проверка на пустоту для графов
            elif isinstance(worker_data[k], (nx.Graph, nx.DiGraph)) and worker_data[k].number_of_nodes() == 0:
                 if k in ['G_walk', 'G_car', 'gdf_demand', 'gdf_candidates']:
                     if k in ['gdf_demand', 'gdf_candidates'] and not worker_data[k].empty:
                         pass
                     elif k in ['gdf_demand', 'gdf_candidates']:
                         empty_data.append(f"{k} (graph/gdf)")
                     else:
                         print(f"  [init {pid}] Граф {k} пуст")

        # Формируем сообщение об ошибке, если что-то не так
        error_messages = []
        if missing_keys: error_messages.append(f"  Отсутствуют ключи: {missing_keys}")
        if null_data: error_messages.append(f"  Значения None для ключей: {null_data}")
        if empty_data: error_messages.append(f"  Пустые данные для ключей: {empty_data}")

        if error_messages:
            # Если отсутствуют критические данные (спрос, кандидаты), считаем это ошибкой
            if any(k in ['gdf_demand', 'gdf_candidates'] for k in missing_keys + null_data + empty_data):
                 raise RuntimeError(f"  Критические ошибки инициализации worker_data: {'; '.join(error_messages)}")
            else:
                 # Если отсутствуют только графы или индекс, выводим предупреждение
                 print(f"  [init {pid}] Предупреждения при проверке worker_data: {'; '.join(error_messages)}")

        # ----- 11. Очистка временных объектов -----
        del gdf_nodes_all, gdf_candidates, gdf_candidates_from_args, demand_data_minimal_from_args

        # valid_nodes_set может быть большим, удалим его тоже
        if 'valid_nodes_set' in locals():
            del valid_nodes_set
        gc.collect()

        print(f"  Воркер {pid} успешно инициализирован за {time.time() - start_init_time:.2f} сек.")

    except Exception as e_init:

        # Ловим любую ошибку во время инициализации
        print(f"  Ошибка инициализации воркера {pid}: {e_init}!")
        traceback.print_exc() # Печатаем полный traceback для диагностики
        worker_data = {'error': True} # Устанавливаем флаг ошибки для этого воркера

# ----- Основной Пайплайн -----
if __name__ == "__main__":
    warnings.filterwarnings(action='ignore',
                            category=FutureWarning)
    pd.options.mode.chained_assignment = None
    pd.set_option('display.float_format', '{:.2f}'.format)

    # ----- 1. Загрузка и подготовка данных -----
    print("----- 1. Загрузка и подготовка данных -----")
    section_start_time = time.time()
    data = {}
    try:
        print("  Загрузка узлов-кандидатов")
        gdf_candidates = gpd.read_file(candidate_nodes_thinned_path)
        gdf_candidates[node_id_col] = gdf_candidates[node_id_col].astype('int32')
        gdf_candidates = gdf_candidates.drop_duplicates(subset=[node_id_col],keep='first')
        gdf_candidates.set_index(node_id_col, inplace=True)
        if str(gdf_candidates.crs).lower() != target_crs.lower(): gdf_candidates = gdf_candidates.to_crs(target_crs)
        data['gdf_candidates'] = gdf_candidates
        print(f"  Загружено {len(gdf_candidates)} кандидатов")

        print("  Загрузка точек спроса (домов)")
        gdf_demand = gpd.read_file(points_shp_path)
        gdf_demand[points_pop_col] = pd.to_numeric(gdf_demand[points_pop_col], errors='coerce').fillna(0).astype('int32')
        initial_demand_count = len(gdf_demand)
        gdf_demand = gdf_demand[gdf_demand[points_pop_col] > 0].copy()
        if gdf_demand.empty:
            raise ValueError("Нет точек спроса!")
        if max_demand_points:
            gdf_demand = gdf_demand.sample(n=max_demand_points,random_state=1)
        gdf_demand.reset_index(drop=True,inplace=True)
        gdf_demand['demand_idx'] = gdf_demand.index.astype('int32')
        if str(gdf_demand.crs).lower() != target_crs.lower(): gdf_demand = gdf_demand.to_crs(target_crs)

        print("  Загрузка всех узлов графа")
        gdf_nodes_all = gpd.read_file(graph_nodes_all_shp)
        gdf_nodes_all[node_id_col] = gdf_nodes_all[node_id_col].astype('int32')
        gdf_nodes_all.dropna(subset=['geometry', node_id_col], inplace=True)
        gdf_nodes_all = gdf_nodes_all.drop_duplicates(subset=[node_id_col], keep='first')
        gdf_nodes_all.set_index(node_id_col, inplace=True)
        if str(gdf_nodes_all.crs).lower() != target_crs.lower():
            gdf_nodes_all = gdf_nodes_all.to_crs(target_crs)

        print("  Поиск ближайших узлов графа для точек спроса")
        nodes_geom_for_search = gdf_nodes_all[['geometry']]
        print(f"    Выполнение sjoin_nearest")
        gdf_demand_with_nodes = gpd.sjoin_nearest(gdf_demand, nodes_geom_for_search, how='left')
        print(f"    sjoin_nearest завершен")
        print(f"    Колонки после sjoin_nearest: {gdf_demand_with_nodes.columns.tolist()}")
        nearest_node_col_name = None
        if node_id_col in gdf_demand_with_nodes.columns and node_id_col not in gdf_demand.columns:
            nearest_node_col_name = node_id_col
        elif f"{node_id_col}_right" in gdf_demand_with_nodes.columns:
            nearest_node_col_name = f"{node_id_col}_right"
        else:
             right_cols = [col for col in gdf_demand_with_nodes.columns if col not in gdf_demand.columns]
             if len(right_cols) == 1:
                 nearest_node_col_name = right_cols[0]; print(f"    Предупреждение: найдена одна новая колонка '{nearest_node_col_name}'!")
             else:
                 raise KeyError(f"    Не удалось определить колонку с ID узла. Новые колонки: {right_cols}")
        print(f"    Используем колонку '{nearest_node_col_name}' как ID")
        missing_nodes_count = gdf_demand_with_nodes[nearest_node_col_name].isnull().sum()
        if missing_nodes_count > 0:
            print(f"    Предупреждение: для {missing_nodes_count} точек спроса не найден узел!")
            gdf_demand_with_nodes.dropna(subset=[nearest_node_col_name], inplace=True)
            print(f"    Точки без узлов удалены. Осталось: {len(gdf_demand_with_nodes)}")
        if gdf_demand_with_nodes.empty:
            raise ValueError("Не осталось точек спроса!")
        gdf_demand_with_nodes.rename(columns={nearest_node_col_name: 'nearest_node_id'}, inplace=True)
        gdf_demand_with_nodes['nearest_node_id'] = gdf_demand_with_nodes['nearest_node_id'].astype('int32')
        data['gdf_demand'] = gdf_demand_with_nodes
        data['demand_point_indices'] = data['gdf_demand']['demand_idx'].tolist()
        print(f"    Ближайшие узлы найдены для {len(data['gdf_demand'])} точек спроса")
        del gdf_nodes_all, nodes_geom_for_search, gdf_demand
        gc.collect()

        print("  Проверка путей к файлам графа")
        if not os.path.exists(graph_nodes_all_shp):
            raise FileNotFoundError(f"Файл узлов не найден: {graph_nodes_all_shp}")
        if not os.path.exists(graph_edges_shp):
            raise FileNotFoundError(f"Файл ребер не найден: {graph_edges_shp}")
        print("  Пути к файлам графа проверены.")

        # Фильтрация кандидатов по компоненте G_car ---
        print("  Фильтрация кандидатов по основной компоненте G_car")

        # Строим G_car только для определения компоненты
        temp_gdf_nodes = gpd.read_file(graph_nodes_all_shp)
        temp_gdf_nodes[node_id_col] = temp_gdf_nodes[node_id_col].astype('int32')
        temp_gdf_nodes.dropna(subset=['geometry', node_id_col], inplace=True)
        temp_gdf_nodes = temp_gdf_nodes.drop_duplicates(subset=[node_id_col], keep='first')
        temp_gdf_nodes.set_index(node_id_col, inplace=True)
        temp_valid_nodes_set = set(temp_gdf_nodes.index)

        temp_gdf_edges = gpd.read_file(graph_edges_shp)
        temp_gdf_edges[edge_start_n_col] = pd.to_numeric(temp_gdf_edges[edge_start_n_col], errors='coerce').fillna(-1).astype('int32')
        temp_gdf_edges[edge_end_n_col] = pd.to_numeric(temp_gdf_edges[edge_end_n_col], errors='coerce').fillna(-1).astype('int32')
        temp_edge_mask = temp_gdf_edges[edge_start_n_col].isin(temp_valid_nodes_set) & temp_gdf_edges[edge_end_n_col].isin(temp_valid_nodes_set)
        temp_gdf_edges = temp_gdf_edges[temp_edge_mask].copy()
        temp_gdf_edges[edge_time_car_col] = pd.to_numeric(temp_gdf_edges[edge_time_car_col], errors='coerce').fillna(np.inf).astype('float32')
        temp_gdf_edges[edge_allow_car_col] = pd.to_numeric(temp_gdf_edges[edge_allow_car_col], errors='coerce').fillna(0).astype('uint8')

        temp_G_car = nx.DiGraph()
        temp_G_car.add_nodes_from(list(temp_valid_nodes_set))
        temp_edges_car = temp_gdf_edges[temp_gdf_edges[edge_allow_car_col] == 1][[edge_start_n_col, edge_end_n_col, edge_time_car_col, edge_oneway_col]]
        temp_edges_car = temp_edges_car[temp_edges_car[edge_time_car_col] != np.inf]
        for row in temp_edges_car.itertuples(index=False):
            u, v, w, ow_obj = row[0], row[1], row[2], row[3]
            ow = str(ow_obj).lower()
            if temp_G_car.has_node(u) and temp_G_car.has_node(v):
                temp_G_car.add_edge(u, v, **{edge_time_car_col: w})
                if ow != 'yes' and ow != 'true' and ow != '1':
                    temp_G_car.add_edge(v, u, **{edge_time_car_col: w})

        # Находим узлы крупнейшей компоненты
        main_component_car_nodes = set()
        if temp_G_car.number_of_nodes() > 0:
            if nx.is_weakly_connected(temp_G_car):
                main_component_car_nodes = set(temp_G_car.nodes())
                print("  G_car связный, используются все кандидаты")
            else:
                largest_cc_car = max(nx.weakly_connected_components(temp_G_car), key=len)
                main_component_car_nodes = largest_cc_car
                print(f"  Найдена крупнейшая компонента G_car: {len(main_component_car_nodes):,} узлов")
        else:
             print("  Предупреждение: автомобильный граф пуст!")

        # Фильтруем gdf_candidates
        initial_candidates_count = len(data['gdf_candidates'])
        data['gdf_candidates'] = data['gdf_candidates'][data['gdf_candidates'].index.isin(main_component_car_nodes)]
        filtered_candidates_count = len(data['gdf_candidates'])
        print(f"  Кандидаты отфильтрованы по компоненте G_car: {filtered_candidates_count} из {initial_candidates_count} ({initial_candidates_count - filtered_candidates_count} удалено)")
        if filtered_candidates_count == 0:
             raise ValueError("  После фильтрации по компоненте G_car не осталось кандидатов!")

        # Очищаем временные объекты
        del temp_gdf_nodes, temp_gdf_edges, temp_G_car, temp_edges_car, temp_valid_nodes_set, temp_edge_mask, main_component_car_nodes
        if 'largest_cc_car' in locals(): del largest_cc_car

        gc.collect()
        print(f"----- Загрузка данных завершена за {time.time() - section_start_time:.2f} сек. -----")

    except MemoryError as e_mem:
        print(f"\nОшибка памяти на этапе 1: {e_mem}!")
        exit()
    except Exception as e:
        print(f"\nОшибка на этапе 1: {e}!")
        traceback.print_exc()
        exit()

    # ----- 3. Расчет матрицы времени доступа -----
    print("----- 3. Расчет матрицы времени доступа -----")

    section_start_time = time.time()
    num_demand = len(data['gdf_demand'])
    num_candidates = len(data['gdf_candidates'])
    print(f"  Расчет матрицы {num_demand:,} (дома) x {num_candidates:,} (кандидаты после фильтрации)")
    cost_matrix = np.full((num_demand, num_candidates), np.inf, dtype='float32')
    mode_matrix = np.full((num_demand, num_candidates), -1, dtype='int8')
    start_matrix_time = time.time()

    # Предварительная фильтрация по расстоянию
    tasks = []
    error_count_distance = 0
    initial_possible_pairs = num_demand * num_candidates
    if max_euclidean_distance_meters is not None and max_euclidean_distance_meters > 0:
        print(f"  Применение предварительной фильтрации по радиусу: {max_euclidean_distance_meters} м.")
        gdf_demand_geoms = data['gdf_demand'].geometry.values
        candidates_sindex = data['gdf_candidates'].sindex
        gdf_candidates_geoms = data['gdf_candidates'].geometry.values
        print("    Начало итерации по домам для фильтрации")
        demand_iterator = tqdm(range(num_demand),
                               desc="Предв. фильтрация",
                               total=num_demand,
                               unit="дом") if tqdm_available else range(num_demand)
        for i in demand_iterator:
            demand_geom = gdf_demand_geoms[i]
            if demand_geom is None or demand_geom.is_empty: continue
            search_bounds = demand_geom.buffer(max_euclidean_distance_meters).bounds
            for j_iloc in candidates_sindex.intersection(search_bounds):
                 candidate_geom = gdf_candidates_geoms[j_iloc]
                 if candidate_geom is None or candidate_geom.is_empty:
                     continue
                 try:
                     distance = demand_geom.distance(candidate_geom)
                     if distance <= max_euclidean_distance_meters: tasks.append((i, j_iloc))
                 except Exception:
                     error_count_distance += 1
                     pass
        print("  Завершена итерация по домам для фильтрации.")
        if error_count_distance > 0:
            print(f"  Предупреждение: произошло {error_count_distance} ошибок при расчете расстояния.")
        num_tasks_after_filter = len(tasks)
        print(
            f"   Осталось задач после фильтрации по расстоянию: {num_tasks_after_filter:,} (из {initial_possible_pairs:,} возможных)")
        if initial_possible_pairs > 0: filter_perc = (
                                                             1 - num_tasks_after_filter / initial_possible_pairs) * 100 if num_tasks_after_filter <= initial_possible_pairs else 0; print(
            f"   Процент отфильтрованных пар: {filter_perc:.2f}%")
        del candidates_sindex  # Удаляем индекс, он больше не нужен
    else:
        print("   Предварительная фильтрация по расстоянию отключена.")
        tasks = [(i, j_iloc) for i in range(num_demand)
                for j_iloc in range(num_candidates)]
        print(f"   Всего задач: {len(tasks):,}")
    gc.collect()

    if not tasks:
        print("  Предупреждение: после фильтрации не осталось задач!")
        exit()
    else:
        print("  Подготовка аргументов для инициализации воркеров")
        if 'nearest_node_id' not in data['gdf_demand'].columns:
            raise ValueError(
            "Колонка 'nearest_node_id' отсутствует!")

        # --- Подготовка данных для передачи ---
        demand_data_minimal = data['gdf_demand'][['geometry', 'nearest_node_id']].copy()

        # Данные кандидатов: только геометрия, но с сохранением индекса (node_id)
        # Важно: создаем копию, чтобы избежать проблем с владением данными
        # Индекс должен быть уже установлен и назван node_id_col в data['gdf_candidates']
        candidates_minimal_geodf = data['gdf_candidates'][['geometry']].copy()
        if candidates_minimal_geodf.index.name != node_id_col:
            # Эта проверка критична, т. к. индекс используется в воркере
            raise ValueError(f"Индекс GeoDataFrame кандидатов ('{candidates_minimal_geodf.index.name}') "
                             f"не совпадает с ожидаемым именем '{node_id_col}' перед передачей в воркеры")
        print(
            f"  Подготовлен GDF кандидатов для передачи ({len(candidates_minimal_geodf)} строк, индекс: '{candidates_minimal_geodf.index.name}')")
        gc.collect()

        # --- Формирование кортежа аргументов ---
        # Передаем пути к файлам графа и сами объекты GeoDataFrame для спроса и кандидатов
        parallel_data_init_args = (
            graph_nodes_all_shp,  # Путь к файлу всех узлов
            graph_edges_shp,  # Путь к файлу ребер
            candidates_minimal_geodf,  # GeoDataFrame кандидатов (с индексом!)
            demand_data_minimal,  # !!! GeoDataFrame спроса
            target_crs,  # Целевая СК
            node_id_col,  # Имя колонки/индекса ID узла
            edge_start_n_col,  # Имя колонки начала ребра
            edge_end_n_col,  # Имя колонки конца ребра
            edge_time_ped_col,  # Имя колонки времени пешком
            edge_time_car_col,  # Имя колонки времени на авто
            edge_allow_car_col,  # Имя колонки доступности авто
            edge_oneway_col,  # Имя колонки одностороннего движения
            points_pop_col,  # Имя колонки населения (для справки, в init_worker не используется)
            walk_speed_mps  # Скорость пешехода (для справки, в init_worker не используется)
        )

        # Очищаем временные переменные GeoDataFrame после формирования аргументов
        del demand_data_minimal, candidates_minimal_geodf
        gc.collect()
        print("  Аргументы для воркеров подготовлены.")

        actual_num_processes = min(num_processes, cpu_count())
        # Последовательное выполнение для отладки
        if actual_num_processes == 1:
            print("  ВЫПОЛНЕНИЕ В ПОСЛЕДОВАТЕЛЬНОМ РЕЖИМЕ (NUM_PROCESSES = 1)!")
            init_worker(parallel_data_init_args)  # Вызов инициализатора
            results_list = []
            tasks_iterator = tqdm(tasks,
                                  desc="Calculating Access Time (Sequential)",
                                  unit="task") if tqdm_available else tasks
            for task_args in tasks_iterator: results_list.append(calculate_access_time_single(task_args))
            processed_tasks_count = len(results_list)
            errors_in_results = sum(1 for r in results_list if r is None or r[2] == np.inf)
            print(f"  Последовательный расчет завершен.")
        else:  # Параллельное выполнение
            print(f"  Запуск {actual_num_processes} параллельных процессов для {len(tasks):,} задач")
            start_pool_time = time.time()
            processed_tasks_count = 0
            errors_in_results = 0
            mode_map = {'walk': 0, 'car': 1}
            try:
                # Создание пула с передачей аргументов инициализатору
                with Pool(processes=actual_num_processes,
                          initializer=init_worker,
                          initargs=(parallel_data_init_args,)) as pool:
                    total_tasks_to_run = len(tasks)
                    chunk_size = max(100, min(1000, total_tasks_to_run // (actual_num_processes * 10)))
                    print(f"      Размер блока (chunksize): {chunk_size:,}")
                    result_iterator = pool.imap_unordered(calculate_access_time_single, tasks, chunksize=chunk_size)
                    result_iterator_tqdm = tqdm(result_iterator, total=total_tasks_to_run,
                                                desc="Calculating Access Time",
                                                unit="task") if tqdm_available else result_iterator
                    # Сбор результатов
                    for result in result_iterator_tqdm:
                        processed_tasks_count += 1
                        if result:
                            i, j_iloc, min_time, best_mode = result
                            if i < num_demand and j_iloc < num_candidates:
                                cost_matrix[i, j_iloc] = min_time if pd.notna(min_time) else np.inf
                                mode_matrix[i, j_iloc] = mode_map.get(best_mode, -1)
                            else:
                                errors_in_results += 1  # Индексы вышли за пределы матрицы
                        else:
                            errors_in_results += 1  # Получен None или некорректный результат
                    if tqdm_available and hasattr(result_iterator_tqdm, 'close'):
                        result_iterator_tqdm.close()
                print(f"  Параллельный расчет завершен за {time.time() - start_pool_time:.2f} сек.")
            except MemoryError as e_mem_pool: print(f"    ОШИБКА ПАМЯТИ ПРИ ПАРАЛЛЕЛЬНОМ РАСЧЕТЕ: {e_mem_pool} !!!"); exit()
            except Exception as e_pool: print(f"    ОШИБКА ПРИ ПАРАЛЛЕЛЬНОМ РАСЧЕТЕ: {e_pool} !!!")

            print(f"   Всего обработано задач пулом: {processed_tasks_count} (из {total_tasks_to_run})")
            if errors_in_results > 0:
                print(f"Предупреждение: Обнаружено {errors_in_results} некорректных/пустых результатов.")

        # Очистка
        if 'tasks' in locals():
            del tasks
        if 'parallel_data_init_args' in locals():
            del parallel_data_init_args
        if 'result_iterator' in locals():
            del result_iterator
        if 'result_iterator_tqdm' in locals():
            del result_iterator_tqdm

    # --- Конец расчета матрицы ---
    print(f"  Матрица времени ({cost_matrix.shape[0]}x{cost_matrix.shape[1]}) и моды ({mode_matrix.shape[0]}x{mode_matrix.shape[1]}) рассчитана за {(time.time() - start_matrix_time) / 60:.2f} минут")
    inf_cells = np.sum(np.isinf(cost_matrix))
    total_cells = cost_matrix.size
    if total_cells > 0:
        print(f"  Количество ячеек с бесконечным временем: {inf_cells:,} / {total_cells:,} ({inf_cells/total_cells*100:.2f}%)")
    else:
        print("  Матрица времени пуста")
    inf_rows = np.all(np.isinf(cost_matrix), axis=1) if total_cells > 0 else np.array([True]*num_demand)
    num_inf_rows = np.sum(inf_rows)
    if num_inf_rows > 0:
        print(f"   Предупреждение: {num_inf_rows:,} домов не имеют доступа ни к одному кандидату!")
    print(f"--- Расчет матрицы доступности завершен за {(time.time() - section_start_time) / 60:.2f} минут ---")

    # ----- 4. Решение задачи оптимизации (PuLP) -----
    print("----- 4. Решение задачи оптимизации (через PuLP) -----")
    section_start_time = time.time()
    gdf_optimal_facilities = gpd.GeoDataFrame()
    allocation_map = {}
    optimal_candidate_ilocs = []
    try:
        if cost_matrix.size == 0:
            raise ValueError("Матрица стоимости пуста")
        if np.all(np.isinf(cost_matrix)):
             print("  Внимание! Все значения в матрице доступности бесконечны! Оптимизация не будет запущена")
             gdf_optimal_facilities = gpd.GeoDataFrame(columns=[node_id_col, 'geometry'],
                                                       geometry='geometry',
                                                       crs=target_crs).set_index(node_id_col)
             metrics = {}
        else:
            population_weights = data['gdf_demand'][points_pop_col].values.astype('float32')
            large_penalty = np.float32(1e10)
            print("    Подготовка матрицы стоимости для PuLP")
            cost_matrix_finite = np.where(np.isinf(cost_matrix), large_penalty, cost_matrix).astype('float32')
            cost_matrix_finite = np.nan_to_num(cost_matrix_finite,
                                               nan=large_penalty,
                                               posinf=large_penalty,
                                               neginf=large_penalty)
            cost_matrix_finite[cost_matrix_finite < 0] = large_penalty

            gc.collect()
            if np.isnan(population_weights).any():
                raise ValueError("NaN в весах!")

            num_demand_pulp, num_candidates_pulp = cost_matrix.shape

            print(f"  Формулировка задачи p-median для N = {n_polyclinics} поликлиник ({num_demand_pulp} спрос x {num_candidates_pulp} кандидатов)")
            start_pulp_model_time = time.time()
            prob = pulp.LpProblem("p-median-polyclinics", pulp.LpMinimize)
            demand_indices_range = range(num_demand_pulp)
            candidate_indices_range = range(num_candidates_pulp) # Используем актуальные размеры

            print("  Создание переменных PuLP")
            assign_vars = pulp.LpVariable.dicts("Y", ((i, j) for i in demand_indices_range for j in candidate_indices_range), cat='Binary')
            facility_vars = pulp.LpVariable.dicts("X", candidate_indices_range, cat='Binary'); print(f"  Переменные созданы за {time.time() - start_pulp_model_time:.2f} сек.")

            print("  Формулировка целевой функции")
            start_obj_time = time.time()
            objective = pulp.LpAffineExpression()
            for i in demand_indices_range:
                for j in candidate_indices_range:
                    objective += float(population_weights[i]) * float(cost_matrix_finite[i, j]) * assign_vars[(i, j)]

            prob += objective, "Total_Weighted_Access_Time"
            print(f"  Целевая функция сформулирована за {time.time() - start_obj_time:.2f} сек.")
            print("  Добавление ограничений...")
            start_constr_time = time.time()
            for i in demand_indices_range: prob += pulp.lpSum(assign_vars[(i, j)] for j in candidate_indices_range) == 1, f"Demand_{i}"
            for i in demand_indices_range:
                for j in candidate_indices_range:
                    prob += assign_vars[(i, j)] <= facility_vars[j], f"Assign_{i}_{j}"

            actual_n_polyclinics = min(n_polyclinics, num_candidates_pulp)
            if actual_n_polyclinics != n_polyclinics:
                 print(f"  Запрошенное число поликлиник ({n_polyclinics}) больше числа доступных кандидатов ({num_candidates_pulp}). Используется {actual_n_polyclinics}")
            prob += pulp.lpSum(facility_vars[j] for j in candidate_indices_range) == actual_n_polyclinics, "Facility_Count"
            print(f"  Ограничения добавлены за {time.time() - start_constr_time:.2f} сек.")
            del cost_matrix_finite
            gc.collect()
            print("  Очищена память от матрицы стоимости для PuLP")

            print(f"   Запуск решателя PuLP (CBC)..."); solver = pulp.getSolver('PULP_CBC_CMD', msg=1, timeLimit=1800); start_solve_time = time.time(); prob.solve(solver); print(f"   Решение PuLP заняло {time.time() - start_solve_time:.2f} сек.")
            status = pulp.LpStatus[prob.status]; print(f"   Статус решения: {status}")
            if prob.status != pulp.LpStatusOptimal: print("   Предупреждение: Оптимальное решение НЕ найдено!")
            if prob.objective is not None: print(f"   Значение целевой функции: {pulp.value(prob.objective):,.2f}")
            else: print("   Значение целевой функции недоступно.")

            print("   Извлечение результатов..."); optimal_candidate_ilocs = [j for j in candidate_indices_range if facility_vars[j].varValue > 0.9]
            if not optimal_candidate_ilocs: raise ValueError("Решатель не выбрал локации!")
            print(f"   Решателем выбрано {len(optimal_candidate_ilocs)} локаций (позиционные индексы)")

            print("   Формирование карты приписки..."); allocation_map = {}
            # !!! Используем ОТФИЛЬТРОВАННЫЙ gdf_candidates из data !!!
            candidate_indices = data['gdf_candidates'].index
            for i in demand_indices_range:
                real_dem_idx = data['demand_point_indices'][i]; best_facility_node_id = -1; min_time_for_demand = np.inf
                for j_iloc in optimal_candidate_ilocs:
                    current_time = cost_matrix[i, j_iloc]
                    if current_time < min_time_for_demand: min_time_for_demand = current_time; best_facility_node_id = candidate_indices[j_iloc]
                if best_facility_node_id != -1: allocation_map[real_dem_idx] = best_facility_node_id
            print(f"   Создана карта приписки для {len(allocation_map)} домов."); unassigned_count_post = num_demand_pulp - len(allocation_map)
            if unassigned_count_post > 0:
                print(f"  Предупреждение: {unassigned_count_post} домов не удалось приписать")

            # Получаем gdf_optimal_facilities из ОТФИЛЬТРОВАННОГО gdf_candidates
            optimal_facility_node_ids = candidate_indices[optimal_candidate_ilocs].tolist()
            gdf_optimal_facilities = data['gdf_candidates'][data['gdf_candidates'].index.isin(optimal_facility_node_ids)].copy()
            print(f"   Найдено {len(gdf_optimal_facilities)} оптимальных локаций в GeoDataFrame.")
            if len(gdf_optimal_facilities) != len(optimal_candidate_ilocs):
                print(f"   Предупреждение: Расхождение числа локаций!")
            del prob, assign_vars, facility_vars, objective
            gc.collect()

    except MemoryError as e_mem_pulp:
        print(f"\n!!! ОШИБКА ПАМЯТИ НА ЭТАПЕ 4 (PuLP): {e_mem_pulp} !!!")
        exit()
    except pulp.PulpError as e_pulp:
        print(f"   Ошибка PuLP: {e_pulp}")
        exit()
    except Exception as e_solve:
        print(f"   Критическая Ошибка на этапе 4: {e_solve}")
        traceback.print_exc(); exit()

    print(f"--- Решение задачи оптимизации завершено за {(time.time() - section_start_time) / 60:.2f} минут ---")

    # ----- 5. Пост-Оптимизационный Анализ -----
    print("----- Пост-Оптимизационный Анализ -----")
    section_start_time = time.time()
    metrics = {}
    if 'gdf_demand' in data and not gdf_optimal_facilities.empty and allocation_map:
        gdf_demand_assigned = data['gdf_demand'].copy()
        try:
            print("  Добавление данных о приписке")
            gdf_demand_assigned['facility_node_id'] = gdf_demand_assigned['demand_idx'].map(allocation_map).fillna(-1).astype('int32')
            num_unassigned_in_gdf = (gdf_demand_assigned['facility_node_id'] == -1).sum()
            if num_unassigned_in_gdf > 0:
                print(f"  В итоговом GDF {num_unassigned_in_gdf} неприписанных домов")

            print("  Извлечение времени доступа и моды")
            access_times = np.full(num_demand_pulp, np.inf, dtype='float32')
            access_modes_int = np.full(num_demand_pulp, -1, dtype='int8')
            node_id_to_iloc_map = {node_id: i for i, node_id in enumerate(data['gdf_candidates'].index)}
            for i, fac_node_id in enumerate(gdf_demand_assigned['facility_node_id']):
                 if fac_node_id != -1:
                      fac_iloc = node_id_to_iloc_map.get(fac_node_id)
                      if fac_iloc is not None and fac_iloc < num_candidates_pulp:
                           access_times[i] = cost_matrix[i, fac_iloc]
                           access_modes_int[i] = mode_matrix[i, fac_iloc]

            gdf_demand_assigned['access_time_sec'] = access_times
            mode_reverse_map = {-1: 'unassigned',
                                0: 'walk',
                                1: 'car'}; gdf_demand_assigned['access_mode'] = pd.Series(access_modes_int).map(mode_reverse_map).astype('category')

            if 'cost_matrix' in locals():
                del cost_matrix
            if 'mode_matrix' in locals():
                del mode_matrix
            del access_times, access_modes_int, node_id_to_iloc_map

            gc.collect()

            print("    Очищена память от матриц.")

            print("    Расчет нагрузки")
            if points_pop_col in gdf_demand_assigned.columns:
                 load_per_facility = gdf_demand_assigned[gdf_demand_assigned['facility_node_id'] != -1].groupby('facility_node_id')[points_pop_col].sum().astype('int32').rename('load_persons')
                 gdf_optimal_facilities = gdf_optimal_facilities.merge(load_per_facility, left_index=True, right_index=True, how='left')
                 gdf_optimal_facilities['load_persons'].fillna(0, inplace=True)
                 print(f"  Нагрузка рассчитана для {len(load_per_facility)} поликлиник")
            else: print(f"  Предупреждение: колонка '{points_pop_col}' не найдена")
            gdf_optimal_facilities['load_persons'] = 0

            print("  Расчет метрик")
            metrics, gdf_optimal_facilities, gdf_demand_assigned = calculate_metrics(gdf_demand_assigned, gdf_optimal_facilities, pop_to_visits_per_shift_coeff, points_pop_col)

        except MemoryError as e_mem_post:
            print(f"Ошибка памяти на этапе 5: {e_mem_post}!")
            exit()
        except Exception as e_post:
            print(f"Ошибка на этапе 5: {e_post}")
            traceback.print_exc()
    else:
        print("  Пропуск пост-анализа: нет данных")
        gdf_demand_assigned = data.get('gdf_demand', gpd.GeoDataFrame())

        if 'facility_node_id' not in gdf_demand_assigned.columns:
            gdf_demand_assigned['facility_node_id'] = -1
        if 'access_time_sec' not in gdf_demand_assigned.columns:
            gdf_demand_assigned['access_time_sec'] = np.inf
        if 'access_mode' not in gdf_demand_assigned.columns:
            gdf_demand_assigned['access_mode'] = 'unassigned'
    print(f"--- Пост-оптимизационный анализ завершен за {(time.time() - section_start_time) / 60:.2f} минут ---")

    # ----- 6. Вывод и сохранение результатов -----
    print("----- 6. Результаты и сохранение -----")
    section_start_time = time.time()
    print("Основные метрики качества")
    if metrics:
        for key, value in metrics.items():
            if not isinstance(value, dict):
                if isinstance(value, (int, float, np.number)):
                    if 'time_sec' in key:
                        metric_name = key.replace('_sec','_min')
                        print(f"   {metric_name}: {float(value) / 60:.2f}")
                    elif isinstance(value, (float, np.floating)):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
                else:
                    print(f"   {key}: {value}")
        if 'mode_stats_house_perc' in metrics:
            print("Статистика по моде (% Домов):")
            [print(f"      {mode}: {perc:.2f}%") for mode, perc in metrics['mode_stats_house_perc'].items()]
        if 'mode_stats_pop_perc' in metrics:
            print("\n   Статистика по моде (% Населения):")
            [print(f"      {mode}: {perc:.2f}%") for mode, perc in metrics['mode_stats_pop_perc'].items()]
    else:
        print("   Метрики не рассчитаны")

    os.makedirs(output_dir, exist_ok=True)
    try:
        print(f"Сохранение оптимальных локаций: {optimal_locations_gpkg}")
        if 'gdf_optimal_facilities' in locals() and not gdf_optimal_facilities.empty:
            gdf_to_save_fac = gdf_optimal_facilities.reset_index()
            for col in gdf_to_save_fac.select_dtypes(include=['float64']).columns:
                gdf_to_save_fac[col] = gdf_to_save_fac[col].astype('float32')
            for col in gdf_to_save_fac.select_dtypes(include=['int64']).columns:
                gdf_to_save_fac[col] = gdf_to_save_fac[col].astype('int32')
            for col in gdf_to_save_fac.select_dtypes(include=['object','category']).columns:
                gdf_to_save_fac[col] = gdf_to_save_fac[col].astype(str)
            gdf_to_save_fac.to_file(optimal_locations_gpkg, layer='optimal_polyclinics', driver='GPKG', engine='fiona')
            print("Сохранено")
        else:
            print("Нет данных для сохранения")
    except Exception as e:
        print(f"Ошибка сохранения локаций: {e}")

    try:
        print(f"Сохранение распределения спроса: {demand_allocation_gpkg}")
        if 'gdf_demand_assigned' in locals() and not gdf_demand_assigned.empty:
            cols_to_save = ['demand_idx', points_pop_col, 'facility_node_id', 'access_time_sec', 'access_mode', 'geometry']
            gdf_to_save = gdf_demand_assigned[[c for c in cols_to_save if c in gdf_demand_assigned.columns]].copy()
            if 'access_time_sec' in gdf_to_save:
                gdf_to_save['access_time_sec'] = gdf_to_save['access_time_sec'].astype('float32')
            if 'facility_node_id' in gdf_to_save:
                gdf_to_save['facility_node_id'] = gdf_to_save['facility_node_id'].astype('int32')
            if 'demand_idx' in gdf_to_save:
                gdf_to_save['demand_idx'] = gdf_to_save['demand_idx'].astype('int32')
            if points_pop_col in gdf_to_save:
                gdf_to_save[points_pop_col] = gdf_to_save[points_pop_col].astype('int32')
            if 'access_mode' in gdf_to_save:
                gdf_to_save['access_mode'] = gdf_to_save['access_mode'].astype(str)
            gdf_to_save.to_file(demand_allocation_gpkg, layer='demand_allocation', driver='GPKG', index=False, engine='fiona')
            print("Сохранено")
        else:
            print("Нет данных для сохранения.")
    except Exception as e:
        print(f"Ошибка сохранения спроса: {e}")

    try:
        print(f"Сохранение метрик: {metrics_csv}")
        if metrics:
            metrics_s = pd.Series(metrics).map(lambda x: x if not isinstance(x, dict) else str(x))
            metrics_df = metrics_s.reset_index()
            metrics_df.columns = ['Metric', 'Value']
            for idx, row in metrics_df.iterrows():
                 if 'time_sec' in row['Metric']:
                     try:
                         metrics_df.loc[idx, 'Value'] = f"{float(row['Value']) / 60:.4f}"
                         metrics_df.loc[idx, 'Metric'] = row['Metric'].replace('_sec', '_min')
                     except:
                         pass
                 elif isinstance(row['Value'], (float, np.floating)):
                     metrics_df.loc[idx, 'Value'] = f"{row['Value']:.4f}"
            metrics_df.to_csv(metrics_csv, index=False, sep=';', decimal='.')
            print("Сохранено.")
        else:
            print("Нет метрик для сохранения.")
    except Exception as e:
        print(f"Ошибка сохранения метрик: {e}")

    print(f"Сохранение результатов завершено за {time.time() - section_start_time:.2f} сек.")
    print(f"Моделирование завершено за {(time.time() - start_global_time) / 60:.2f} мин.")