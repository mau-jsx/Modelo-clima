import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import os
import pickle

# ===== CONFIGURACIÓN =====
CSV_PATH = "formosa_weather_complete.csv"
LAT, LON = -26.1775, -58.1781
API_URL = "https://api.open-meteo.com/v1/forecast"
SEQUENCE_LENGTH = 7 * 24  # 7 días de datos horarios
MODEL_PATH = "lstm_model.keras"
SCALERS_PATH = "scalers.pkl"
RE_TRAIN_HOURS = 3 # Re-entrenar el modelo cada 3 horas

# Variables que se usarán como entrada y salida
FEATURES_LIST = ["temperature", "precipitation", "wind_speed", "humidity", "cloudcover", "surface_pressure"]
TARGETS_LIST = ["rain", "storm", "fire_risk", "temperature", "cloudcover"]

# ===== FUNCIÓN RIESGO DE INCENDIO =====
def calcular_riesgo_incendio(temp, humedad, viento, lluvia):
    temp_factor = max(0, min(1, (temp - 20) / 20))
    humedad_factor = max(0, min(1, (50 - humedad) / 50))
    viento_factor = max(0, min(1, viento / 40))
    lluvia_factor = max(0, min(1, 1 - (lluvia / 10)))
    riesgo = (0.4 * temp_factor +
              0.3 * humedad_factor +
              0.2 * viento_factor +
              0.1 * lluvia_factor) * 100
    return round(riesgo, 1)

# ===== 1. CARGAR Y PREPARAR DATOS HISTÓRICOS =====
def preparar_datos_historicos():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    else:
        df = pd.DataFrame(columns=["date", "temperature", "precipitation", "wind_speed", "humidity",
                                   "cloudcover", "surface_pressure", "storm", "rain", "fire_risk"])
    return df

def descargar_datos_recientes(df):
    start_date = (df["date"].max() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M") if not df.empty else "2020-01-01T00:00"
    end_date = datetime.now().strftime("%Y-%m-%dT%H:%M")
    
    if start_date > end_date:
        return df

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date[:10],
        "end_date": end_date[:10],
        "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,cloudcover,surface_pressure",
        "timezone": "America/Argentina/Buenos_Aires"
    }

    try:
        r = requests.get(API_URL, params=params)
        r.raise_for_status()
        data = r.json()
        if "hourly" in data:
            hourly = pd.DataFrame(data["hourly"])
            hourly["date"] = pd.to_datetime(hourly["time"])
            hourly = hourly.rename(columns={
                "temperature_2m": "temperature",
                "precipitation": "precipitation",
                "relative_humidity_2m": "humidity",
                "wind_speed_10m": "wind_speed",
                "surface_pressure": "surface_pressure"
            })
            
            hourly["storm"] = (hourly["precipitation"] > 10).astype(int)
            hourly["rain"] = (hourly["precipitation"] > 2).astype(int)
            hourly["fire_risk"] = hourly.apply(lambda row: calcular_riesgo_incendio(
                row["temperature"], row["humidity"], row["wind_speed"], row["precipitation"]), axis=1)

            hourly = hourly[["date"] + FEATURES_LIST + ["storm", "rain", "fire_risk"]]
            df = pd.concat([df, hourly]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            df.to_csv(CSV_PATH, index=False)
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con la API: {e}")
    
    return df

# ===== 2. ENTRENAR O CARGAR EL MODELO =====
def entrenar_o_cargar_modelo(df):
    if (os.path.exists(MODEL_PATH) and os.path.exists(SCALERS_PATH)):
        last_modified = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        if (datetime.now() - last_modified) < timedelta(hours=RE_TRAIN_HOURS):
            print("Cargando modelo y scalers existentes...")
            model = load_model(MODEL_PATH)
            with open(SCALERS_PATH, 'rb') as f:
                scalers = pickle.load(f)
            features_scaler = scalers['features']
            targets_scaler = scalers['targets']
            return model, features_scaler, targets_scaler
        else:
            print("El modelo ha caducado, re-entrenando...")

    print("Entrenando un nuevo modelo...")
    if df is None or len(df) < SEQUENCE_LENGTH + 1:
        print("No hay suficientes datos para entrenar el modelo LSTM.")
        return None, None, None

    features = df[FEATURES_LIST].values
    targets = df[TARGETS_LIST].values

    features_scaler = MinMaxScaler(feature_range=(0, 1))
    targets_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = features_scaler.fit_transform(features)
    scaled_targets = targets_scaler.fit_transform(targets)
    
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_features)):
        X.append(scaled_features[i - SEQUENCE_LENGTH:i, :])
        y.append(scaled_targets[i, :]) 
    
    X, y = np.array(X), np.array(y)
    
    model = Sequential()
    # Aumento de unidades LSTM a 100
    model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(100))
    model.add(Dense(len(TARGETS_LIST)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=15, batch_size=25, verbose=0)
    
    model.save(MODEL_PATH)
    with open(SCALERS_PATH, 'wb') as f:
        pickle.dump({'features': features_scaler, 'targets': targets_scaler}, f)
    
    print("Modelo entrenado y guardado con éxito.")
    return model, features_scaler, targets_scaler

# ===== 3. OBTENER PRONÓSTICO HOY y MAÑANA =====
def obtener_pronostico(model, features_scaler, targets_scaler, df):
    hoy = datetime.now().date()
    manana = hoy + timedelta(days=1)
    
    # Obtener pronóstico diario desde la API
    params_diarios = {
        "latitude": LAT,
        "longitude": LON,
        "forecast_days": 2,
        "daily": "temperature_2m_max,temperature_2m_min,weather_code",
        "timezone": "America/Argentina/Buenos_Aires"
    }
    r_diarios = requests.get(API_URL, params=params_diarios)
    data_diarios = r_diarios.json()

    # Mapeo de códigos de clima (WMO Weather interpretation codes) a descripciones
    weather_codes = {
        0: "Despejado (soleado)",
        1: "Principalmente despejado",
        2: "Parcialmente nublado",
        3: "Nublado",
        45: "Niebla",
        48: "Niebla con escarcha",
        51: "Llovizna ligera",
        53: "Llovizna moderada",
        55: "Llovizna densa",
        61: "Lluvia ligera",
        63: "Lluvia moderada",
        65: "Lluvia intensa",
        66: "Lluvia helada ligera",
        67: "Lluvia helada intensa",
        71: "Nevada ligera",
        73: "Nevada moderada",
        75: "Nevada intensa",
        77: "Granos de nieve",
        80: "Chubascos de lluvia ligeros",
        81: "Chubascos de lluvia moderados",
        82: "Chubascos de lluvia violentos",
        95: "Tormenta",
        96: "Tormenta con granizo ligero",
        99: "Tormenta con granizo intenso"
    }

    # Pronóstico para HOY
    temp_min_hoy = data_diarios["daily"]["temperature_2m_min"][0]
    temp_max_hoy = data_diarios["daily"]["temperature_2m_max"][0]
    weather_code_hoy = data_diarios["daily"]["weather_code"][0]
    clima_hoy = weather_codes.get(weather_code_hoy, "Desconocido")
    
    # Pronóstico para MAÑANA (con el modelo LSTM)
    last_sequence_data = df[FEATURES_LIST].tail(SEQUENCE_LENGTH).values
    scaled_features = features_scaler.transform(last_sequence_data)
    
    X_pred = scaled_features.reshape(1, SEQUENCE_LENGTH, len(FEATURES_LIST))
    predicted_scaled = model.predict(X_pred)
    predicted_unscaled = targets_scaler.inverse_transform(predicted_scaled)
    
    # Extraer las predicciones del modelo LSTM
    predicciones = predicted_unscaled[0]
    rain_pred = predicciones[TARGETS_LIST.index('rain')]
    storm_pred = predicciones[TARGETS_LIST.index('storm')]
    fire_risk_pred = predicciones[TARGETS_LIST.index('fire_risk')]
    
    rain_prob = max(0, rain_pred * 100)
    storm_prob = max(0, storm_pred * 100)
    fire_risk_pred = predicciones[TARGETS_LIST.index('fire_risk')]
    temp_max_manana = data_diarios["daily"]["temperature_2m_max"][1]
    temp_min_manana = data_diarios["daily"]["temperature_2m_min"][1]
    
    # Lógica para la descripción de temperatura de mañana
    temp_promedio_manana = (temp_max_manana + temp_min_manana) / 2
    
    sensacion_termica_manana = "templado"
    if temp_promedio_manana < 5:
        sensacion_termica_manana = "muy frío"
    elif temp_promedio_manana < 15:
        sensacion_termica_manana = "frío"
    elif temp_promedio_manana > 30:
        sensacion_termica_manana = "calor"
    elif temp_promedio_manana > 25:
        sensacion_termica_manana = "caluroso"

    return {
        "hoy": {
            "fecha": hoy.strftime("%d de %B de %Y"),
            "min_temp": temp_min_hoy,
            "max_temp": temp_max_hoy,
            "clima": clima_hoy
        },
        "manana": {
            "fecha": manana.strftime("%d de %B de %Y"),
            "min_temp": temp_min_manana,
            "max_temp": temp_max_manana,
            "sensacion": sensacion_termica_manana,
            "lluvia_prob": rain_prob,
            "tormenta_prob": storm_prob,
            "riesgo_incendio": fire_risk_pred
        }
    }

# ===== EJECUCIÓN PRINCIPAL =====
if __name__ == "__main__":
    df = preparar_datos_historicos()
    df_actualizado = descargar_datos_recientes(df)
    
    model, features_scaler, targets_scaler = entrenar_o_cargar_modelo(df_actualizado)
    
    if model:
        pronostico = obtener_pronostico(model, features_scaler, targets_scaler, df_actualizado)
        
        # Imprimir el pronóstico de HOY
        print(f"\nPronóstico para hoy, {pronostico['hoy']['fecha']} en Formosa Capital ☀️")
        print(f"Rango de temperatura: Min. {pronostico['hoy']['min_temp']:.1f}°C / Max. {pronostico['hoy']['max_temp']:.1f}°C")
        print(f"El cielo estará: {pronostico['hoy']['clima']}")
        
        print("\n" + "-"*30 + "\n")
        
        # Imprimir el pronóstico de MAÑANA
        print(f"Predicción para mañana, {pronostico['manana']['fecha']} en Formosa Capital 🌤️")
        print(f"Rango de temperatura: Min. {pronostico['manana']['min_temp']:.1f}°C / Max. {pronostico['manana']['max_temp']:.1f}°C")
        print(f"Sensación térmica estimada: {pronostico['manana']['sensacion']}")
        print(f"🌧️ Probabilidad de lluvia: {pronostico['manana']['lluvia_prob']:.0f}%")
        print(f"⛈️ Probabilidad de tormenta: {pronostico['manana']['tormenta_prob']:.0f}%")
        print(f"🔥 Riesgo de incendio: {pronostico['manana']['riesgo_incendio']:.0f}%")