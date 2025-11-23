import os

import streamlit as st
import pandas as pd
import joblib
import kagglehub

# --- Cargar modelo ---
model = joblib.load('f1_predictor_model.joblib')
le = joblib.load("label_encoder.joblib")

# --- Cargar dataset de circuitos ---
path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
fname = "circuits.csv"
fpath = os.path.join(path, fname)
if os.path.exists(fpath):
    circuits = pd.read_csv(fpath, na_values="\\N")
else:
    print(f"⚠️ No encontrado: {fname}")

# Crear etiqueta amigable
circuits["label"] = circuits["name"] + " (" + circuits["country"] + ")"
circuit_map = dict(zip(circuits["label"], circuits["circuitId"]))

features = [
    'grid',
    'elo_driver_before','elo_constructor_before',
    'driver_points_before','driver_wins_before',
    'constructor_points_before','constructor_wins_before',
    'qual_position','qualifying_gap','qual_gap_vs_teammate',
    'race_gap_vs_teammate',
    'year','circuitId',
    'AirTemp','TrackTemp','Humidity','Rainfall','WindSpeed'
]

display_to_real = {
    "Posición de salida (grid)": "grid",
    "ELO del piloto": "elo_driver_before",
    "ELO del constructor": "elo_constructor_before",
    "Puntos acumulados del piloto": "driver_points_before",
    "Victorias del piloto": "driver_wins_before",
    "Puntos acumulados del constructor": "constructor_points_before",
    "Victorias del constructor": "constructor_wins_before",
    "Posición en clasificación": "qual_position",
    "Gap en clasificación (vs líder)": "qualifying_gap",
    "Gap en clasificación (vs compañero)": "qual_gap_vs_teammate",
    "Gap en carrera (vs compañero)": "race_gap_vs_teammate",
    "Año": "year",
    "Circuito": "circuitId",
    "Temperatura del aire": "AirTemp",
    "Temperatura de pista": "TrackTemp",
    "Humedad (%)": "Humidity",
    "Lluvia (mm)": "Rainfall",
    "Velocidad del viento (km/h)": "WindSpeed"
}

default_values = {
    'grid': 1,
    'elo_driver_before': 2203.081727,
    'elo_constructor_before': 744.012201,
    'driver_points_before': 25,
    'driver_wins_before': 1,
    'constructor_points_before': 43,
    'constructor_wins_before': 1,
    'qual_position': 1,
    'qualifying_gap': 0,
    'qual_gap_vs_teammate': -1,
    'race_gap_vs_teammate': -1,
    'year': 2023,
    'circuitId': 3,
    'AirTemp': 27.431677,
    'TrackTemp': 31.011801,
    'Humidity': 21.496894,
    'Rainfall': 0.0,
    'WindSpeed': 0.68323
}


st.title("🏁 Predicción de resultados en Fórmula 1")

inputs = {}

# --- Circuito con selector ---
circuit_choice = st.selectbox("Circuito", list(circuit_map.keys()))
inputs["circuitId"] = circuit_map[circuit_choice]

# --- Resto de inputs ---
for display_name, real_name in display_to_real.items():

    if real_name == "circuitId":
        continue

    if real_name == "year":
        val = st.number_input(
            "Año de la carrera",
            min_value=1950,
            max_value=3000,
            value=2023,
            step=1
        )
        inputs[real_name] = int(val)
        continue

    if real_name == "grid":
        val = st.number_input(
            "Posición de salida (grid)",
            value=1,
            step=1
        )
        inputs[real_name] = int(val)
        continue

    val = st.number_input(display_name, value=float(default_values.get(real_name, 0.0)))
    inputs[real_name] = float(val)

# --- Predicción ---
if st.button("Predecir resultado"):
    ejemplo_df = pd.DataFrame([inputs])
    pred = model.predict(ejemplo_df)[0]
    clase = le.inverse_transform([pred])[0]

    st.success(f"🏆 Resultado predicho: **{clase}**")