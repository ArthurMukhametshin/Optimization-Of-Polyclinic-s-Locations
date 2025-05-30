# ----- Импорт библиотек -----

import time
import pandas as pd
import geopandas as gpd

# ----- Параметры -----

gtfs_trips_file = r'Data\GTFS\trips.txt'
gtfs_routes_file = r'Data\GTFS\routes.txt'
gtfs_calendar_file = r'Data\GTFS\calendar.txt'
gtfs_stop_times_file = r'Data\GTFS\stop_times.txt'

output_gpkg = r'Data\modelData\transportRoutes.gpkg'  # Путь сохранения маршрутов ОТ
output_layer_name = 'pt_segments' # Имя таблицы внутри GeoPackage

# Имена полей в GTFS
stop_times_trip_id = 'trip_id'
stop_times_arrival = 'arrival_time'
stop_times_departure = 'departure_time'
stop_times_stop_id = 'stop_id'
stop_times_sequence = 'stop_sequence'

trips_trip_id = 'trip_id'
trips_route_id = 'route_id'
trips_service_id = 'service_id'

routes_route_id = 'route_id'
routes_type = 'route_type'
routes_short_name = 'route_short_name'
routes_agency_id = 'agency_id'

calendar_service_id = 'service_id'

# Параметры
filter_by_service_day = True
target_day_of_week = 'monday'

# Максимально разумное время сегмента в секундах (для фильтрации ошибок GTFS)
max_reasonable_segment_time_sec = 6 * 3600 # 6 часов

# ----- Функции -----

def time_to_seconds(time_str):
    """Конвертирует HH:MM:SS в секунды от полуночи, обрабатывая >24 часов"""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, TypeError, AttributeError):
        return None

# ----- Загрузка данных GTFS -----

start_time = time.time()
print("Фаза 2: Создание сегментов ОТ из расписания")

print("Загрузка данных GTFS")

dtype_spec = {stop_times_trip_id: str,
              stop_times_stop_id: str,
              trips_trip_id: str,
              trips_route_id: str,
              trips_service_id: str,
              routes_route_id: str,
              routes_agency_id: str,
              calendar_service_id: str}

df_stop_times = pd.read_csv(gtfs_stop_times_file,
                            dtype=dtype_spec,
                            low_memory=False)
df_trips = pd.read_csv(gtfs_trips_file,
                       dtype=dtype_spec,
                       low_memory=False)
df_routes = pd.read_csv(gtfs_routes_file,
                        dtype=dtype_spec,
                        low_memory=False)

active_service_ids = None
if filter_by_service_day:
    df_calendar = pd.read_csv(gtfs_calendar_file,
                              dtype=dtype_spec,
                              low_memory=False)
    print(f"  Загружен calendar.txt ({len(df_calendar)})")

    active_service_ids = set(df_calendar[df_calendar[target_day_of_week] == 1][calendar_service_id])
    print(f"  Найдено {len(active_service_ids)} активных service_id для '{target_day_of_week}'")

print(f"  Загружено {len(df_stop_times)} stop_times, {len(df_trips)} trips, {len(df_routes)} routes")

# ----- Фильтрация рейсов -----

if filter_by_service_day and active_service_ids is not None:
    print("Фильтрация рейсов по service_id")
    initial_trips = len(df_trips)
    df_trips = df_trips[df_trips[trips_service_id].isin(active_service_ids)].copy()
    print(f"  Оставлено {len(df_trips)} рейсов из {initial_trips}")

# ----- Объединение данных -----

print("Объединение данных stop_times, trips, routes")

df_trips_reduced = df_trips[[trips_trip_id, trips_route_id]]
df_routes_reduced = df_routes[[routes_route_id, routes_type, routes_short_name, routes_agency_id]]
df_trips_routes = pd.merge(df_trips_reduced, df_routes_reduced,
                           on=routes_route_id,
                           how='inner')

# Добавление проверки на пустой результат после первого объединения
if df_trips_routes.empty:
    raise ValueError("  Нет совпадений между trips и routes по route_id!")
df_merged = pd.merge(df_stop_times, df_trips_routes,
                     on=trips_trip_id,
                     how='inner')

print(f"  Объединение завершено. Записей: {len(df_merged)}")

# ----- Обработка времени и сортировка -----

print("Обработка времени и сортировка")

df_merged['arrival_sec'] = df_merged[stop_times_arrival].apply(time_to_seconds)
df_merged['departure_sec'] = df_merged[stop_times_departure].apply(time_to_seconds)
initial_rows = len(df_merged)

# Проверка stop_sequence до удаления NaN по времени
if stop_times_sequence not in df_merged.columns:
    raise ValueError("  Колонка stop_sequence отсутствует")
df_merged[stop_times_sequence] = pd.to_numeric(df_merged[stop_times_sequence], errors='coerce')

# Удаление строки с некорректными ID, временем или sequence
df_merged.dropna(subset=['arrival_sec', 'departure_sec', stop_times_stop_id, trips_trip_id, stop_times_sequence],
                 inplace=True)

# Конвертация типов после удаления NaN
df_merged['arrival_sec'] = df_merged['arrival_sec'].astype(int)
df_merged['departure_sec'] = df_merged['departure_sec'].astype(int)
df_merged[stop_times_sequence] = df_merged[stop_times_sequence].astype(int)

print(f"  Удалено {initial_rows - len(df_merged)} строк с некорректным временем/sequence/ID")

df_merged.sort_values(by=[trips_trip_id, stop_times_sequence],
                      inplace=True)
print("  Данные отсортированы")

# ----- Создание сегментов и расчет времени -----

print("Создание сегментов и расчет времени")

df_merged['prev_stop_id'] = df_merged.groupby(trips_trip_id)[stop_times_stop_id].shift(1)
df_merged['prev_departure_sec'] = df_merged.groupby(trips_trip_id)['departure_sec'].shift(1)
df_segments_calc = df_merged.dropna(subset=['prev_stop_id', 'prev_departure_sec']).copy()

df_segments_calc['prev_departure_sec'] = df_segments_calc['prev_departure_sec'].astype(int)
df_segments_calc['time_sec'] = df_segments_calc['arrival_sec'] - df_segments_calc['prev_departure_sec']

# Обработка перехода через полночь
day_seconds = 24 * 3600
midnight_threshold = -20 * 3600
rollover_mask = df_segments_calc['time_sec'] < 0
df_segments_calc.loc[rollover_mask & (df_segments_calc['time_sec'] > midnight_threshold), 'time_sec'] += day_seconds

# Фильтрация некорректного времени
valid_time_mask = (df_segments_calc['time_sec'] >= 0) & (df_segments_calc['time_sec'] < max_reasonable_segment_time_sec)
df_segments_final = df_segments_calc[valid_time_mask].copy()
num_invalid_time = len(df_segments_calc) - len(df_segments_final)

if num_invalid_time > 0:
    print(f"  Удалено {num_invalid_time} сегментов с некорректным временем")

if df_segments_final.empty:
    print("  Не осталось сегментов после фильтрации по времени!")
else:
    print(f"  Рассчитано {len(df_segments_final)} валидных сегментов ОТ")

    # Отбор и переименование колонок
    df_segments_final = df_segments_final[[routes_route_id,
                                           routes_short_name,
                                           routes_type,
                                           routes_agency_id,
                                           trips_trip_id,
                                           'prev_stop_id',
                                           stop_times_stop_id,
                                           stop_times_sequence,
                                           'time_sec']]
    df_segments_final.rename(columns={routes_route_id: 'route_id',
                                      routes_short_name: 'route_ref',
                                      routes_type: 'route_type',
                                      routes_agency_id: 'agency_id',
                                      trips_trip_id: 'trip_id',
                                      'prev_stop_id': 'from_stop_id',
                                      stop_times_stop_id: 'to_stop_id',
                                      stop_times_sequence: 'end_stop_seq',
                                      'time_sec': 'time_sec'},
                             inplace=True)

    # Добавление sequence сегмента
    df_segments_final['sequence'] = df_segments_final.groupby('trip_id').cumcount()

    # Конвертация типов
    df_segments_final['time_sec'] = df_segments_final['time_sec'].round(3)
    df_segments_final['route_type'] = df_segments_final['route_type'].astype(int)
    df_segments_final['sequence'] = df_segments_final['sequence'].astype(int)
    df_segments_final['end_stop_seq'] = df_segments_final['end_stop_seq'].astype(int)

    for col in ['route_id', 'agency_id', 'trip_id', 'from_stop_id', 'to_stop_id', 'route_ref']:
         if col in df_segments_final.columns:
             df_segments_final[col] = df_segments_final[col].astype(str)

# ----- Сохранение результата -----

print(f"Сохранение результата: {output_gpkg}, слой: {output_layer_name}")

gdf_to_save = gpd.GeoDataFrame(df_segments_final)
gdf_to_save.to_file(output_gpkg,
                    layer=output_layer_name,
                    driver="GPKG",
                    index=False)

print(f"  Таблица сегментов сохранена ({len(gdf_to_save)} строк)")

end_time_phase2 = time.time()
print(f"Фаза 2 завершена за {end_time_phase2 - start_time:.2f} секунд")