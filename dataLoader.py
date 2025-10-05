#%% md
# 🏁 Predicción de Ganadores F1 — Dataset FastF1 (Quali + Carrera)
# Este notebook genera automáticamente un dataset con datos combinados de clasificación
# y carrera de todas las temporadas elegidas usando la API FastF1.

#%%

import fastf1 as ff1
import pandas as pd
from tqdm import tqdm

# Habilitar caché (evita re-descargas)
ff1.Cache.enable_cache('./cache')

#%% md
# ## ⚙️ Funciones auxiliares

#%%

def extract_session_data(session, session_type):
    results = session.results
    rows = []

    # Clima agregado seguro
    weather = session.weather_data

    # Inicializamos valores por defecto
    AvgTemp = RainFlag = AvgHumidity = WindSpeed = None

    if not weather.empty:
        if 'Temperature' in weather.columns:
            AvgTemp = weather['Temperature'].mean()
        if 'RainIntensity' in weather.columns:
            RainFlag = 1 if (weather['RainIntensity'] > 0).any() else 0
        if 'Humidity' in weather.columns:
            AvgHumidity = weather['Humidity'].mean()
        if 'WindSpeed' in weather.columns:
            WindSpeed = weather['WindSpeed'].mean()

    for _, res in results.iterrows():
        drv_code = res['Abbreviation']
        laps = session.laps.pick_driver(drv_code)

        base = {
            'Year': session.event['EventDate'].year,
            'RoundNumber': session.event['RoundNumber'],
            'Race': session.event['EventName'],
            'Driver': res['FullName'],
            'Code': drv_code,
            'Team': res['TeamName'],
            # Clima
            'AvgTemp': AvgTemp,
            'RainFlag': RainFlag,
            'AvgHumidity': AvgHumidity,
            'WindSpeed': WindSpeed
        }

        if session_type == 'R':
            row = {
                **base,
                'Grid': res['GridPosition'],
                'FinalPos': res['Position'],
                'Status': res['Status'],
                'Points': res['Points'],
                'AvgLapTime': laps['LapTime'].mean().total_seconds() if not laps.empty else None,
                'BestLap': laps['LapTime'].min().total_seconds() if not laps.empty else None,
                'PitStops': laps['PitOutTime'].count(),
                'TyreStints': laps['Compound'].nunique(),
                'TyreCompounds': ','.join(laps['Compound'].dropna().unique()) if not laps.empty else None,
                'Winner': 1 if res['Position'] == 1 else 0,
                'Top3': 1 if res['Position'] <= 3 else 0,
            }
        elif session_type == 'Q':
            qtime = None
            if not laps.empty:
                qtime = laps['LapTime'].min()
                qtime = qtime.total_seconds() if pd.notnull(qtime) else None
            row = {
                **base,
                'QualiPos': res['Position'],
                'QualiTime': qtime,
            }

        rows.append(row)

    return pd.DataFrame(rows)

def build_race_dataset(year, gp_name, session_type='R'):
    """Carga una sesión de FastF1 (Race o Quali) y devuelve DataFrame resumen."""
    try:
        session = ff1.get_session(year, gp_name, session_type)
        session.load()
        return extract_session_data(session, session_type)
    except Exception as e:
        print(f"❌ Error cargando {session_type} de {gp_name} {year}: {e}")
        return pd.DataFrame()


def build_season_dataset(year):
    """Genera dataset de una temporada completa (Q + R combinadas)."""
    print(f"\n📅 Descargando temporada {year}")
    schedule = ff1.get_event_schedule(year)
    races = schedule[schedule['EventFormat'] == 'conventional']  # filtra tests y no-carreras

    all_races = []
    for gp_name in tqdm(races['EventName'], desc=f"Cargando {year}"):
        quali_df = build_race_dataset(year, gp_name, 'Q')
        race_df = build_race_dataset(year, gp_name, 'R')

        if quali_df.empty or race_df.empty:
            continue

        # merge: por Year + Race + Code
        merged = pd.merge(
            race_df,
            quali_df[['Year', 'Race', 'Code', 'QualiPos', 'QualiTime']],
            on=['Year', 'Race', 'Code'],
            how='left'
        )

        all_races.append(merged)

    if all_races:
        return pd.concat(all_races, ignore_index=True)
    return pd.DataFrame()


#%% md
# ## 🧾 Descarga de múltiples temporadas y combinación total

#%%

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]  # podés agregar más si están disponibles

all_years = []
for y in YEARS:
    df_season = build_season_dataset(y)
    if not df_season.empty:
        all_years.append(df_season)

df_all = pd.concat(all_years, ignore_index=True)
df_all.to_csv('f1_fastf1_dataset_2021_2024.csv', index=False)

print("✅ Dataset final generado con éxito")
print(f"Total filas: {df_all.shape[0]}")
print("Columnas:", df_all.columns.tolist())

df_all.head()

#%% md
# ## 📊 Estructura del dataset final
# - Year, Race, RoundNumber
# - Driver, Code, Team
# - Grid, FinalPos, Points, Status
# - AvgLapTime, BestLap, PitStops, TyreCompounds
# - Winner, Top3
# - QualiPos, QualiTime
