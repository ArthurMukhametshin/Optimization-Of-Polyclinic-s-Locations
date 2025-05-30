# ----- Импорт библиотек -----

import time
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# ----- Параметры -----

candidates_with_centrality_path = r'Data\modelData\roads_nodes_filtered_candidates.shp'

output_thinned_candidates_shp = r'Data\modelData\roads_nodes_filtered_candidates_byDegreeCentrality.shp' # Путь для сохранения финальных кандидатов

# Параметры
target_crs = "EPSG:32637"
thinning_grid_size = 500 # Шаг сетки в метрах

# Имена колонок в candidates_with_centrality_path
node_id_col = 'NODE_ID'
centrality_col = 'DEGREE_CEN'

# ----- Загрузка узлов -----

start_time = time.time()

print("Загрузка узлов")

# Загрузка кандидатов с центральностью
gdf_candidates = gpd.read_file(candidates_with_centrality_path)
print(f"  Загружено {len(gdf_candidates)} кандидатов")

# Проверка CRS
if str(gdf_candidates.crs).lower() != target_crs.lower():
    print(f"  Перепроецирование кандидатов в {target_crs}")
    gdf_candidates = gdf_candidates.to_crs(target_crs)

# Удаление дубликатов NODE_ID
if gdf_candidates[node_id_col].duplicated().any():
    print("  Удаление дубликатов NODE_ID")
    gdf_candidates.sort_values(by=centrality_col,
                               ascending=False,
                               inplace=True)
    gdf_candidates.drop_duplicates(subset=[node_id_col],
                                   keep='first',
                                   inplace=True)
    print(f"  Осталось {len(gdf_candidates)} уникальных кандидатов")

# ----- Создание сетки и привязка кандидатов -----

print(f"Создание сетки (шаг {thinning_grid_size}м) и привязка кандидатов")

# Использование охвата загруженных кандидатов
xmin, ymin, xmax, ymax = gdf_candidates.total_bounds

# Добавление буфера к границам для надежности
xmin -= thinning_grid_size
ymin -= thinning_grid_size
xmax += thinning_grid_size
ymax += thinning_grid_size

# Генерация координат ячеек
cols = list(np.arange(xmin, xmax + thinning_grid_size, thinning_grid_size))
rows = list(np.arange(ymin, ymax + thinning_grid_size, thinning_grid_size))

# Создание полигонов сетки
polygons = [Polygon([(cols[x],rows[y]),
                     (cols[x+1],rows[y]),
                     (cols[x+1],rows[y+1]),
                     (cols[x],rows[y+1])])

            for x in range(len(cols)-1) for y in range(len(rows)-1)]

gdf_grid = gpd.GeoDataFrame(data={'grid_id': range(len(polygons))},
                            geometry=polygons,
                            crs=target_crs)

print(f"  Создана сетка из {len(gdf_grid)} ячеек")

# Присоединение ID ячейки к кандидатам
print("  Присоединение ID ячеек к кандидатам")
gdf_candidates_with_grid = gpd.sjoin(gdf_candidates, gdf_grid[['grid_id', 'geometry']],
                                     how='left',
                                     predicate='within')

# Удаление кандидатов, не попавших ни в одну ячейку (если вдруг такие есть)
initial_count_sjoin = len(gdf_candidates_with_grid)
gdf_candidates_with_grid.dropna(subset=['index_right'],
                                inplace=True)
print(f"  Привязано {len(gdf_candidates_with_grid)} кандидатов к ячейкам (удалено {initial_count_sjoin - len(gdf_candidates_with_grid)} не попавших)")

# ----- Отбор лучшего кандидата по центральности в каждой ячейке -----

print("Отбор лучшего кандидата по центральности в каждой ячейке")

# Сортировка по grid_id и затем по убыванию центральности
gdf_candidates_with_grid.sort_values(by=['grid_id', centrality_col],
                                     ascending=[True, False],
                                     inplace=True)

# Оставление только первой (лучшей по центральности) записи для каждого grid_id
gdf_thinned_candidates = gdf_candidates_with_grid.drop_duplicates(subset=['grid_id'],
                                                                  keep='first')
print(f"  Отобрано {len(gdf_thinned_candidates)} кандидатов после разреживания")

# ----- Сохранение результата -----

print(f"Сохранение итогового набора кандидатов в {output_thinned_candidates_shp}")

gdf_final_candidates = gdf_thinned_candidates[[node_id_col, 'geometry']].copy()
gdf_final_candidates.to_file(output_thinned_candidates_shp)

print("  Сохранение завершено")

print(f"Разреживание кандидатов завершено за {time.time() - start_time:.2f} секунд")