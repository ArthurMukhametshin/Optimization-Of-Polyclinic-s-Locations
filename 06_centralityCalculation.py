# ----- Импорт библиотек -----

import time
import warnings
import networkx as nx
import geopandas as gpd

# ----- Параметры -----

graph_nodes_all_shp = r'Data\modelData\roads_nodes_filtered.shp'
graph_edges_shp = r'Data\modelData\roads_edges_filtered.shp'

output_nodes_with_centrality_shp = r'Data\modelData\roads_nodes_filtered_with_centrality.shp'   # Путь сохранения узлов с центральностью

# Параметры
target_crs = "EPSG:32637" # Целевая СК
centrality_graph_type = 'car' # По какому графу считать ('walk' или 'car')

# Имена колонок
node_id_col = 'NODE_ID'
edge_start_n_col = 'START_N'
edge_end_n_col = 'END_N'
edge_allow_car_col = 'ALLOW_CAR' # Если centrality_graph_type = 'car', то 'ALLOW_CAR'; Если centrality_graph_type = 'walk', то 'ALLOW_PED'

# ----- Загрузка узлов и ребер -----

start_time = time.time()

print("Загрузка данных")

gdf_nodes_all = gpd.read_file(graph_nodes_all_shp)
gdf_edges = gpd.read_file(graph_edges_shp)

print(f"  Загружено: {len(gdf_nodes_all)} узлов, {len(gdf_edges)} ребер")

# Проверка и установка CRS
if gdf_nodes_all.crs is None or gdf_edges.crs is None:
    raise ValueError("Отсутствует CRS у узлов или ребер")

if str(gdf_nodes_all.crs).lower() != target_crs.lower():
    gdf_nodes_all = gdf_nodes_all.to_crs(target_crs)

if str(gdf_edges.crs).lower() != target_crs.lower():
    gdf_edges = gdf_edges.to_crs(target_crs)

print(f"  CRS проверены ({target_crs})")

# Очистка узлов (NaN геометрия, дубликаты ID)
gdf_nodes_all.dropna(subset=['geometry'],
                     inplace=True)

gdf_nodes_all = gdf_nodes_all[~gdf_nodes_all.geometry.is_empty]

if gdf_nodes_all[node_id_col].duplicated().any():
    print("  Удаление дубликатов NODE_ID у узлов")
    gdf_nodes_all.drop_duplicates(subset=[node_id_col],
                                  keep='first',
                                  inplace=True)

gdf_nodes_all.set_index(node_id_col, inplace=True)

# ----- Построение графа NetworkX -----

print(f"Построение графа ({centrality_graph_type})")

if centrality_graph_type == 'walk':
    G_cent = nx.Graph()

elif centrality_graph_type == 'car':
    G_cent = nx.DiGraph()

else:
    raise ValueError("  Неверный centrality_graph_type")

all_node_ids_set = set(gdf_nodes_all.index)
G_cent.add_nodes_from(all_node_ids_set)

# Выбор ребер
if centrality_graph_type == 'walk':
    edges_df_cent = gdf_edges[[edge_start_n_col, edge_end_n_col]]
else:
    edges_df_cent = gdf_edges[gdf_edges[edge_allow_car_col] == 1][[edge_start_n_col, edge_end_n_col]]

# Добавление ребер, где оба конца существуют
edges_for_cent = [(u,v) for u,v in edges_df_cent.values.tolist() if u in all_node_ids_set and v in all_node_ids_set]
G_cent.add_edges_from(edges_for_cent)

print(f"  Граф построен: {G_cent.number_of_nodes()} узлов, {G_cent.number_of_edges()} ребер")

# ----- Расчет Degree Centrality -----

print("Расчет Degree Centrality")

degree_map = dict(G_cent.degree())
gdf_nodes_all['degree_cen'] = gdf_nodes_all.index.map(degree_map).fillna(0).astype(int)
print(f"  Центральность (Degree) рассчитана. Макс: {gdf_nodes_all['degree_cen'].max()}")

# ----- Сохранение результата -----

print(f"Сохранение узлов с центральностью в {output_nodes_with_centrality_shp}")

# Сброс индекса, чтобы NODE_ID стал колонкой
gdf_nodes_to_save = gdf_nodes_all.reset_index()

# Переименовывание degree_cen для Shapefile
gdf_nodes_to_save.rename(columns={'degree_cen': 'DEGREE_CEN'},
                         inplace=True)

# Выбор колонок для сохранения
cols_to_save = [node_id_col, 'DEGREE_CEN', 'geometry']

# Добавление COMP_ID, если он был в исходном файле узлов
if 'COMP_ID' in gdf_nodes_to_save.columns: cols_to_save.insert(__index=1,
                                                               __object='COMP_ID')

gdf_to_save_final = gdf_nodes_to_save[[col for col in cols_to_save if col in gdf_nodes_to_save.columns]]

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore",
                            category=UserWarning,
                            message="Column names longer than 10 characters will be truncated")
    gdf_to_save_final.to_file(output_nodes_with_centrality_shp,
                              driver='ESRI Shapefile',
                              encoding='utf-8')

print("  Сохранение завершено")

print(f"Расчет центральности завершен за {time.time() - start_time:.2f} секунд")