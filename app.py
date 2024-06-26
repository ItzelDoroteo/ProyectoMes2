from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado y el escalador
model = joblib.load('modelo_emissions.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request (JSON)
        data = request.get_json()

        fuel = float(data['fuel'])
        fuel_combustion = float(data['fuel_combustion'])
        unnamed9 = float(data['unnamed9'])
        unnamed10 = float(data['unnamed10'])
        # Connection_type_DSL = float(data['Connection_type_DSL'])

        # Escalar los datos de entrada
        input_data = [[fuel, fuel_combustion, unnamed9, unnamed10, ]]
        scaled_data = scaler.transform(input_data)
        
        # Crear un DataFrame con los datos escalados
        data_df = pd.DataFrame(scaled_data, columns=['ENGINE_SIZE', 'FUEL_CONSUMPTION*', 'Unnamed: 9', 'Unnamed: 10'])
        
        # Imprimir el DataFrame para verificar los datos
        print("Datos recibidos (escalados):")
        print(data_df)
        
        # Realizar la predicción
        prediction = model.predict(data_df)[0]
        
        # Devolver la predicción como respuesta JSON
        return jsonify({'CO2_EMISSIONS': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
