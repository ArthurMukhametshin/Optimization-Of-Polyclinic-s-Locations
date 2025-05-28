# ----- Импорт библиотек -----

import os
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.prepared import prep

# ----- Параметры -----

# Пути файлов
points_shp_path = 'Data/zhilBuildings_pnt.shp' # Точки жилых зданий
buildings_shp_path = 'Data/buildings_pol.shp' # Полигоны зданий
districts_shp_path = 'Data/districts_pol.shp' # Полигоны муниципальных районов Москвы
output_shp_path = 'Data/zhilBuildings_pop_pnt.shp' # Путь сохранения точек жилых зданий с населением

# Наименования полей
points_levels_col = 'levels' # Поле с этажностью в points_shp_path
points_type_col = 'file_name' # Поле для определения типа дома (МКД/ИЖС) в points_shp_path
districts_pop_col = 'pop_rostat' # Поле с населением в районах в districts_shp_path
districts_id_col = 'dist_id' # Поле с ID районов в districts_shp_path

# Коэффициенты и критерии
living_area_coeff = 0.847 # Коэффициент жилой площади
max_nearest_distance = 100 # Максимальное расстояние для поиска ближайшего полигона в метрах

izhs_type_values = ['частные дома'] # Значения в поле points_type_col, соответствующие ИЖС
min_pop_izhs = 2 # Минимальное население для ИЖС
min_pop_mkd = 1 # Минимальное население для МКД

# ----- Подавление предупреждений -----

warnings.filterwarnings(action='ignore',
                        message='GeoSeries.notna',
                        category=UserWarning)

# ----- Загрузка данных -----

start_time = time.time()

print("Загрузка данных")

gdf_points = gpd.read_file(points_shp_path)
print(f"  Загружено точек: {len(gdf_points)}")

gdf_buildings_orig = gpd.read_file(buildings_shp_path)
print(f"  Загружено полигонов зданий: {len(gdf_buildings_orig)}")

gdf_districts = gpd.read_file(districts_shp_path)
print(f"  Загружено районов: {len(gdf_districts)}")

# ----- Проверка и установка CRS -----

print("Проверка и установка CRS")

target_crs = gdf_points.crs
if str(gdf_buildings_orig.crs).lower() != str(target_crs).lower():
    gdf_buildings_orig = gdf_buildings_orig.to_crs(target_crs)
if str(gdf_districts.crs).lower() != str(target_crs).lower():
    gdf_districts = gdf_districts.to_crs(target_crs)

print(f"  CRS успешно синхронизированы ({target_crs})")

# ----- Предобработка полигонов зданий -----

print("Предобработка полигонов зданий")

print("  Выполнение разбиения мультиполигонов")
if 'MultiPolygon' in gdf_buildings_orig.geom_type.unique():
    gdf_buildings_exploded = gdf_buildings_orig.explode(index_parts=True)
    gdf_buildings_exploded = gdf_buildings_exploded[gdf_buildings_exploded.geometry.geom_type == 'Polygon'].copy()
else:
    gdf_buildings_exploded = gdf_buildings_orig.copy()
    gdf_buildings_exploded.index = pd.MultiIndex.from_arrays(arrays=[gdf_buildings_exploded.index, np.zeros(len(gdf_buildings_exploded),dtype=int)],
                                                             names=['level_0', 'level_1'])

print("  Исправление геометрии полигонов через построение буфера")
geom_fixed = gdf_buildings_exploded.geometry.buffer(0)
valid_mask = geom_fixed.is_valid & ~geom_fixed.is_empty & (geom_fixed.geom_type == 'Polygon')
gdf_buildings = gdf_buildings_exploded.loc[valid_mask].copy()
gdf_buildings.geometry = geom_fixed[valid_mask]

print("  Расчет площади полигонов")
gdf_buildings['area_fp_part'] = gdf_buildings.geometry.area
gdf_buildings = gdf_buildings[gdf_buildings['area_fp_part'] > 0].copy()

print("  Подготовка к сопоставлению")
gdf_buildings = gdf_buildings.reset_index(drop=True)

print("  Создание пространственного индекса зданий")
buildings_sindex = gdf_buildings.sindex
print("  Пространственный индекс создан")

# ----- Сопоставление точек и полигонов зданий -----

print("Сопоставление точек и полигонов зданий")

gdf_points['area_fp'] = 0.0
gdf_points['associated_part_idx'] = -1
processed_points = 0
total_points = len(gdf_points)

for index, point in gdf_points.iterrows():
    point_geom = point.geometry
    selected_area = 0.0
    selected_part_idx_final = -1
    possible_matches_idx = list(buildings_sindex.intersection(point_geom.bounds))

    if possible_matches_idx:
        possible_matches = gdf_buildings.iloc[possible_matches_idx]
        point_prepared = prep(point_geom)
        precise_matches = possible_matches[possible_matches.geometry.apply(lambda g: point_prepared.intersects(g))]

        if not precise_matches.empty:
             selected_part_idx = precise_matches['area_fp_part'].idxmin() if len(precise_matches) > 1 else precise_matches.index[0]
             selected_area = precise_matches.loc[selected_part_idx, 'area_fp_part']
             selected_part_idx_final = selected_part_idx

    if selected_part_idx_final == -1:
        nearest_indices = list(buildings_sindex.nearest(point_geom, return_all=False)[1])

        if nearest_indices:
            nearest_part_idx = nearest_indices[0]
            nearest_geom = gdf_buildings.geometry.iloc[nearest_part_idx]
            actual_distance = point_geom.distance(nearest_geom)

            if actual_distance <= max_nearest_distance:
                buffer_geom = point_geom.buffer(max_nearest_distance)
                possible_nearby_idx = list(buildings_sindex.intersection(buffer_geom.bounds))

                if possible_nearby_idx:
                    possible_nearby = gdf_buildings.iloc[possible_nearby_idx]
                    nearby_polygons = possible_nearby[possible_nearby.distance(point_geom) <= max_nearest_distance]

                    if not nearby_polygons.empty:
                        selected_part_idx = nearby_polygons['area_fp_part'].idxmin()
                        selected_area = nearby_polygons.loc[selected_part_idx, 'area_fp_part']
                        selected_part_idx_final = selected_part_idx

    gdf_points.loc[index, 'area_fp'] = selected_area
    gdf_points.loc[index, 'associated_part_idx'] = selected_part_idx_final
    processed_points += 1

    if processed_points % 10000 == 0:
        print(f"  Обработано точек: {processed_points}/{total_points}")

print(f"Сопоставление завершено. Точек с площадью (area_fp > 0): {len(gdf_points[gdf_points['area_fp'] > 0])}/{total_points}")

# ----- Расчет жилой площади -----

print("Расчет жилой площади")

gdf_points[points_levels_col] = pd.to_numeric(gdf_points[points_levels_col], errors='coerce').fillna(1).apply(lambda x: 1 if x <= 0 else x).astype(int)
gdf_points['area'] = gdf_points['area_fp'] * gdf_points[points_levels_col]
gdf_points['area_live'] = gdf_points['area'] * living_area_coeff

print(f"  Расчет жилой площади завершен. Суммарная расчетная жилая площадь (area_live): {gdf_points['area_live'].sum():.2f}")

# ----- Привязка точек к районам -----

print("Привязка точек к районам")

gdf_points_with_district = gpd.sjoin(gdf_points, gdf_districts[[districts_id_col, districts_pop_col, 'geometry']], how='left', predicate='within')

conflicting_cols = [col for col in [districts_id_col, districts_pop_col] if col in gdf_points.columns]
district_id_col_actual = districts_id_col
pop_col_actual = districts_pop_col
if districts_id_col in conflicting_cols:
    district_id_col_actual = f"{districts_id_col}_right"
if districts_pop_col in conflicting_cols:
    pop_col_actual = f"{districts_pop_col}_right"
unassigned_points_mask = gdf_points_with_district[district_id_col_actual].isnull()

print("Привязка точек к районам завершена")

# ----- Первичное дазиметрическое распределение населения по жилым зданиям -----

print("Первичное дазиметрическое распределение населения по жилым зданиям")

gdf_points_with_district['total_area_live_district'] = gdf_points_with_district.groupby(district_id_col_actual)['area_live'].transform('sum')
gdf_points_with_district['density_district'] = np.where(
    gdf_points_with_district['total_area_live_district'] > 0,
    gdf_points_with_district[pop_col_actual] / gdf_points_with_district['total_area_live_district'], 0)
gdf_points_with_district['density_district'] = gdf_points_with_district['density_district'].fillna(0)
gdf_points_with_district['pop_initial'] = gdf_points_with_district['area_live'] * gdf_points_with_district['density_district']
gdf_points_with_district['pop_initial'] = gdf_points_with_district['pop_initial'].fillna(0)

print("Первичное дазиметрическое распределение населения по жилым зданиям завершено")

# ----- Двухэтапное перераспределение -----

print("Выполнение двухэтапного перераспределения населения")

# Добавление временных колонок
gdf_points_with_district['pop_final'] = gdf_points_with_district['pop_initial']
gdf_points_with_district['is_izhs'] = gdf_points_with_district[points_type_col].isin(izhs_type_values) & (gdf_points_with_district['area_live'] > 0)
gdf_points_with_district['is_mkd'] = (~gdf_points_with_district[points_type_col].isin(izhs_type_values)) & (gdf_points_with_district['area_live'] > 0)
gdf_points_with_district['min_pop'] = 0
gdf_points_with_district.loc[gdf_points_with_district['is_izhs'], 'min_pop'] = min_pop_izhs
gdf_points_with_district.loc[gdf_points_with_district['is_mkd'], 'min_pop'] = min_pop_mkd
gdf_points_with_district['pop_adjusted'] = gdf_points_with_district['pop_initial']
gdf_points_with_district['deficit'] = 0.0
gdf_points_with_district['surplus'] = 0.0

districts_processed = 0
total_districts = gdf_points_with_district[district_id_col_actual].nunique()
results = []

# Итерация по районам
for district_id, group in gdf_points_with_district.groupby(district_id_col_actual):
    districts_processed += 1

    if districts_processed % 10 == 0: print(f"  Обработка района {districts_processed}/{total_districts}")
    group = group.copy()
    pop_rostat_district = group[pop_col_actual].iloc[0]

    # Применение минимумов и расчет дефицита
    mask_low_pop = (group['pop_adjusted'] < group['min_pop'])
    group.loc[mask_low_pop, 'deficit'] = group.loc[mask_low_pop, 'min_pop'] - group.loc[mask_low_pop, 'pop_adjusted']
    group.loc[mask_low_pop, 'pop_adjusted'] = group.loc[mask_low_pop, 'min_pop']
    total_deficit = group['deficit'].sum()

    # Определение доноров и расчет излишек
    mask_donor = (group['pop_adjusted'] > group['min_pop'])
    group.loc[mask_donor, 'surplus'] = group.loc[mask_donor, 'pop_adjusted'] - group.loc[mask_donor, 'min_pop']
    total_surplus = group['surplus'].sum()

    # Перераспределение
    group['pop_final'] = group['pop_adjusted']

    if total_deficit > 0 and total_surplus > 1e-6:
        reduction_ratio = min(total_deficit / total_surplus, 1.0)
        group.loc[mask_donor, 'pop_final'] = group.loc[mask_donor, 'pop_adjusted'] - (group.loc[mask_donor, 'surplus'] * reduction_ratio)
    elif total_deficit > 0:
         pass

    # Финальная корректировка суммы до pop_rostat
    current_sum = group['pop_final'].sum()
    final_diff = pop_rostat_district - current_sum

    if abs(final_diff) > 0.1 and current_sum > 1e-6:
         group['pop_final'] = group['pop_final'] + (group['pop_final'] / current_sum) * final_diff
         group['pop_final'] = np.maximum(group['pop_final'], 0)

    # Округление до целых чисел
    group['pop_final'] = group['pop_final'].round()

    # Корректировка округления для точного совпадения суммы
    final_sum_rounded = group['pop_final'].sum()
    rounding_diff = int(round(pop_rostat_district - final_sum_rounded))

    if rounding_diff != 0 and not group.empty:
         indices_to_adjust = group.index[:abs(rounding_diff)]
         adjustment = np.sign(rounding_diff)
         group.loc[indices_to_adjust, 'pop_final'] += adjustment
         group['pop_final'] = np.maximum(group['pop_final'], 0)

    results.append(group[['pop_final']])

# Объединение результатов и обновление основного DataFrame
print("Объединение результатов перераспределения")

if results:
     final_pop_df = pd.concat(results)
     gdf_points_with_district.update(final_pop_df)
     gdf_points_with_district['pop_estimated'] = gdf_points_with_district['pop_final'].fillna(0).astype(int)
else:
     print("  Не удалось обработать ни одного района для перераспределения")
     gdf_points_with_district['pop_estimated'] = gdf_points_with_district['pop_initial'].round().astype(int)

print("Перераспределение завершено")

# ----- Финализация и сохранение -----

print("Финализация и сохранение")

cols_to_remove = ['index_right', 'total_area_live_district', 'density_district', 'associated_part_idx', 'pop_col_actual',
                  'pop_initial', 'is_izhs', 'is_mkd', 'min_pop', 'pop_adjusted','deficit', 'surplus', 'pop_final']

if district_id_col_actual != districts_id_col:
    cols_to_remove.append(district_id_col_actual)

cols_to_drop_existing = [col for col in cols_to_remove if col in gdf_points_with_district.columns]
print(f"  Столбцы для удаления: {cols_to_drop_existing}")

gdf_final = gdf_points_with_district.drop(columns=cols_to_drop_existing)
gdf_final.rename(columns={'pop_estimated': 'population'},
                 inplace=True)

print(f"Столбцы в итоговом gdf_final: {gdf_final.columns.tolist()}")

# ----- Проверка результата -----

print("Проверка сумм населения по районам")

district_sums_original = gdf_districts.set_index(districts_id_col)[districts_pop_col]

if districts_id_col not in gdf_final.columns:
     original_id_col_name = [col for col in gdf_final.columns if col.startswith(districts_id_col)]

     if original_id_col_name:
         districts_id_col_for_grouping = original_id_col_name[0]
     else:
         raise ValueError(f"  Не найдена колонка ID района '{districts_id_col}' для группировки")

else:
    districts_id_col_for_grouping = districts_id_col

district_sums_estimated = gdf_final.groupby(districts_id_col_for_grouping)['population'].sum()
comparison = pd.DataFrame({'Original': district_sums_original, 'Estimated': district_sums_estimated}).fillna(0)
comparison['Difference'] = comparison['Original'] - comparison['Estimated']

print("  Сравнение сумм по районам:")
print(comparison[comparison['Difference'] != 0])
total_diff = comparison['Difference'].sum()

print(f"  Общее расчетное население: {comparison['Estimated'].sum()}")
print(f"  Общее исходное население: {comparison['Original'].sum()}")

print(f"  Итоговое расхождение: {total_diff}")

# ----- Сохранение результата -----

print(f"Сохранение результата в {output_shp_path}")

output_dir = os.path.dirname(output_shp_path)
gdf_final.to_file(output_shp_path, encoding='utf-8')

print(f"Результат успешно сохранен")

print(f"Скрипт завершен. Время выполнения: {time.time() - start_time:.2f} секунд")