# LINK A RENDER - https://wine-qa-streamlit.onrender.com

import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_wine
import os

# Cargar el modelo
model_path = os.path.join(os.path.dirname(__file__), "wine_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Nombres de clase reales y amigables
wine_data = load_wine()
class_names = wine_data.target_names
friendly_names = {
    'class_0': 'Vino tipo A (ej: Barolo)',
    'class_1': 'Vino tipo B (ej: Chianti)',
    'class_2': 'Vino tipo C (ej: Beaujolais)'
}

# Encabezado
st.title("🍷 Clasificador de Vinos")

# Entradas del usuario
st.markdown("Por favor, ingresa las características químicas del vino:")

feature_names = [
    "Alcohol", "Ácido málico", "Ceniza", "Alcalinidad de la ceniza", "Magnesio",
    "Fenoles totales", "Flavonoides", "Fenoles no flavonoides", "Proantocianinas",
    "Intensidad de color", "Tonalidad (Hue)", "OD280/OD315 (diluidos)", "Prolina"
]

# Capturar entradas del usuario
features = []
for name in feature_names:
    value = st.number_input(name, step=1)
    features.append(value)

# Botón de predicción
if st.button("Predecir"):
    try:
        prediction = model.predict([features])[0]
        class_name = class_names[prediction]
        st.success(f"Predicción: {friendly_names[class_name]}")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
