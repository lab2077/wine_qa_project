from flask import Flask, request, render_template
from sklearn.datasets import load_wine
import pickle
import numpy as np

# Inicializar la app Flask
app = Flask(__name__)

# Cargar el modelo entrenado
with open("wine_model.pkl", "rb") as file:
    model = pickle.load(file)

# Cargar nombres de las clases del dataset original
wine_data = load_wine()
class_names = wine_data.target_names  # ['class_0', 'class_1', 'class_2']

# Nombres amigables para mostrar al usuario
friendly_names = {
    'class_0': 'Vino tipo A (ej: Barolo)',
    'class_1': 'Vino tipo B (ej: Chianti)',
    'class_2': 'Vino tipo C (ej: Beaujolais)'
}

# Ruta principal con formulario
@app.route("/", methods=["GET", "POST"])
def predict():
    prediction_text = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["alcohol"]),
                float(request.form["malic_acid"]),
                float(request.form["ash"]),
                float(request.form["alcalinity_of_ash"]),
                float(request.form["magnesium"]),
                float(request.form["total_phenols"]),
                float(request.form["flavanoids"]),
                float(request.form["nonflavanoid_phenols"]),
                float(request.form["proanthocyanins"]),
                float(request.form["color_intensity"]),
                float(request.form["hue"]),
                float(request.form["od280_od315"]),
                float(request.form["proline"]),
            ]
            prediction = model.predict([features])[0]
            class_label = class_names[prediction]  # e.g., 'class_1'
            prediction_text = f"Predicción: {friendly_names[class_label]}"
        except Exception as e:
            prediction_text = f"Error en la predicción: {e}"
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
