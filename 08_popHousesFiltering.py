# ----- Импорт библиотек -----

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# ----- Параметры -----

input_points_path = r'Data\modelData\popHouses.shp' # Путь к исходному файлу с домами
output_aggregated_path = r'Data\modelData\popHouses_filtered.shp' # Путь для сохранения результата
grid_cell_size_meters = 200 # Размер ячейки сетки в метрах
target_crs = "EPSG:32637" # Целевая СК

# Имена колонок во входном файле
id_col_in = 'id'          # Имя колонки с ID во входном файле
pop_col_in = 'population' # Имя колонки с населением во входном файле

# Имена колонок в выходном файле
id_col_out = 'agg_id'      # Имя колонки ID в выходном файле
pop_col_out = 'population' # Имя колонки населения в выходном файле

# Допуск для охвата (чтобы точки не лежали на границе)
buffer_dist = 1.0

# ----- ФУНКЦИЯ -----

def aggregate_points_by_grid(points_filepath,
                             output_filepath,
                             grid_size,
                             target_crs,
                             input_id_col,
                             input_pop_col,
                             output_id_col,
                             output_pop_col,
                             buffer_dist=1.0):
    """
    Агрегирование точек по сетке путем создания центроидов с суммарным населением.
    Сохранение результата в Shapefile.

    Args:
        points_filepath (str): путь к исходному файлу точек.
        output_filepath (str): путь для сохранения агрегированных точек.
        grid_size (float): размер ячейки сетки в единицах системы координат.
        target_crs (str): целевая система координат (например, "EPSG:32637").
        input_id_col (str): имя колонки ID в исходных данных.
        input_pop_col (str): имя колонки населения в исходных данных.
        output_id_col (str): имя колонки ID в выходных данных.
        output_pop_col (str): имя колонки населения в выходных данных.
        buffer_dist (float): размер буфера для охвата точек.
    """


    # ----- Загрузка данных -----

    start_time = time.time()

    print("Загрузка исходных точек")
    points_gdf = gpd.read_file(points_filepath)
    print(f"  Загружено {len(points_gdf)} точек")

    # Проверка CRS
    if points_gdf.crs is None:
        warnings.warn("  CRS входного файла не определена. Устанавливается target_crs")
        points_gdf.set_crs(target_crs, allow_override=True)
    elif str(points_gdf.crs).lower() != str(target_crs).lower():
        print(f"  Перепроекция точек в {target_crs}")
        points_gdf = points_gdf.to_crs(target_crs)

    # Очистка данных: удаление строк без геометрии или с некорректным населением
    points_gdf = points_gdf[points_gdf.geometry.is_valid & points_gdf.geometry.notna()]
    points_gdf[input_pop_col] = pd.to_numeric(points_gdf[input_pop_col], errors='coerce').fillna(0)
    points_gdf = points_gdf[points_gdf[input_pop_col] >= 0]
    print(f"  Осталось {len(points_gdf)} точек после очистки геометрии и населения")

    # ----- Определение охвата и буферизация -----
    print(f"Определение охвата и создание буфера ({buffer_dist} м)")
    minx, miny, maxx, maxy = points_gdf.total_bounds
    minx -= buffer_dist
    miny -= buffer_dist
    maxx += buffer_dist
    maxy += buffer_dist

    # ----- Создание сетки -----
    print(f"Создание сетки с шагом {grid_size} м")
    cols = list(np.arange(minx, maxx + grid_size, grid_size))
    rows = list(np.arange(miny, maxy + grid_size, grid_size))

    polygons = []
    grid_id_counter = 0
    for x_start in cols[:-1]:
        for y_start in rows[:-1]:
            x_end = x_start + grid_size
            y_end = y_start + grid_size
            polygons.append(
                {
                    'grid_id': grid_id_counter,
                    'geometry': Polygon([(x_start, y_start),
                                         (x_start, y_end),
                                         (x_end, y_end),
                                         (x_end, y_start)])
                 }
            )
            grid_id_counter += 1

    grid_gdf = gpd.GeoDataFrame(polygons, crs=target_crs)
    print(f"  Создано {len(grid_gdf)} ячеек сетки")

    # ----- Пространственное соединение -----
    print("Соединение точек с сеткой")

    # Используем только нужные колонки
    points_for_join = points_gdf[[input_id_col, input_pop_col, 'geometry']].copy()

    # Освобождение памяти
    del points_gdf
    gc.collect()

    # Использование sindex для ускорения
    points_sindex = points_for_join.sindex

    # Соединение точек с сеткой
    joined_gdf = gpd.sjoin(points_for_join, grid_gdf,
                           how='inner',
                           predicate='within')
    print(f"  {len(joined_gdf)} точек попало в ячейки сетки")

    # Освобождение памяти
    del points_for_join, grid_gdf, points_sindex
    gc.collect()

    # ----- Агрегация по ячейкам -----
    print("Агрегация данных по ячейкам сетки")

    # Добавление координат X и Y для расчета среднего
    joined_gdf['x_coord'] = joined_gdf.geometry.x
    joined_gdf['y_coord'] = joined_gdf.geometry.y

    # Группировка и агрегирование
    aggregated_data = joined_gdf.groupby('grid_id').agg(
        sum_population=(input_pop_col, 'sum'),
        mean_x=('x_coord', 'mean'),
        mean_y=('y_coord', 'mean'),).reset_index()

    print(f"  Агрегировано {len(aggregated_data)} ячеек")

    # Освобождение памяти
    del joined_gdf
    gc.collect()

    # ----- Формирование результата -----
    print("Создание геометрии для агрегированных точек")

    aggregated_geometry = [Point(xy) for xy in zip(aggregated_data['mean_x'], aggregated_data['mean_y'])]

    # Создание итогового GeoDataFrame
    final_aggregated_gdf = gpd.GeoDataFrame(aggregated_data[['sum_population']],
                                            geometry=aggregated_geometry,
                                            crs=target_crs)

    # Переименовывание колонки населения
    final_aggregated_gdf.rename(columns={'sum_population': output_pop_col},
                                inplace=True)

    # Добавление нового ID
    final_aggregated_gdf.reset_index(drop=True,
                                     inplace=True)
    final_aggregated_gdf[output_id_col] = final_aggregated_gdf.index

    # Оставление только нужных колонок и в нужном порядке
    final_aggregated_gdf = final_aggregated_gdf[[output_id_col, output_pop_col, 'geometry']]

    # ----- Сохранение результата в Shapefile -----
    print(f"Сохранение результата в Shapefile: {output_filepath}")

    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Проверка длины имен колонок
    for col in final_aggregated_gdf.columns:
        if col != 'geometry' and len(col) > 10:
             warnings.warn(f"Имя колонки '{col}' длиннее 10 символов. Оно может быть усечено при сохранении в Shapefile.")

    final_aggregated_gdf.to_file(output_filepath, driver='ESRI Shapefile')
    print("  Результат успешно сохранен.")

    end_time = time.time()
    print(f"Агрегация завершена за {end_time - start_time:.2f} секунд")

# ----- Выполнение скрипта -----
if __name__ == "__main__":

    # Установка предупреждений pandas/geopandas
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None

    # Запуск основной функции
    aggregate_points_by_grid(
        points_filepath=input_points_path,
        output_filepath=output_aggregated_path,
        grid_size=grid_cell_size_meters,
        target_crs=target_crs,
        input_id_col=id_col_in,
        input_pop_col=pop_col_in,
        output_id_col=id_col_out,
        output_pop_col=pop_col_out,
        buffer_dist=buffer_dist
    )

    print("Скрипт завершил работу")