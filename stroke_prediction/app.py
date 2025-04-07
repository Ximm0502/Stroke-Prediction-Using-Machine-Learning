from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__, static_folder='static')
model = joblib.load('stroke_model.pkl')
scaler = joblib.load('scaler.pkl')

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        
        # Tạo DataFrame từ dữ liệu đầu vào
        input_data = pd.DataFrame([data])
        
        # Xử lý Blood Pressure Levels
        if 'Blood Pressure Levels' not in input_data.columns or not isinstance(input_data['Blood Pressure Levels'].iloc[0], str):
            raise ValueError("Blood Pressure Levels must be a string in format 'Systolic/Diastolic'")
        blood_pressure = input_data['Blood Pressure Levels'].str.split('/', expand=True)
        if blood_pressure.shape[1] != 2:
            raise ValueError("Blood Pressure Levels must be in format 'Systolic/Diastolic' (e.g., '120/80')")
        input_data['Systolic Pressure'] = blood_pressure[0].astype(int)
        input_data['Diastolic Pressure'] = blood_pressure[1].astype(int)
        
        # Xử lý Cholesterol Levels
        if 'Cholesterol Levels' not in input_data.columns or not isinstance(input_data['Cholesterol Levels'].iloc[0], str):
            raise ValueError("Cholesterol Levels must be a string in format 'HDL: value, LDL: value'")
        input_data['HDL'] = input_data['Cholesterol Levels'].str.extract(r'HDL: (\d+)').astype(float)
        input_data['LDL'] = input_data['Cholesterol Levels'].str.extract(r'LDL: (\d+)').astype(float)
        if input_data['HDL'].isna().any() or input_data['LDL'].isna().any():
            raise ValueError("Cholesterol Levels must be in format 'HDL: value, LDL: value' (e.g., 'HDL: 50, LDL: 120')")
        
        # Các cột cần thiết theo mô hình
        numeric_cols = ["Average Glucose Level", "Body Mass Index (BMI)", "Stress Levels",
                        "Systolic Pressure", "Diastolic Pressure", "HDL", "LDL"]
        categorical_cols = ["Gender", "Marital Status", "Work Type", "Residence Type",
                            "Smoking Status", "Alcohol Intake", "Physical Activity",
                            "Stroke History", "Family History of Stroke", "Dietary Habits"]
        symptom_cols = ["Blurred Vision", "Confusion", "Difficulty Speaking", "Dizziness",
                        "Headache", "Loss of Balance", "Numbness", "Seizures", "Severe Fatigue", "Weakness"]
        
        # Đảm bảo tất cả các cột triệu chứng đều có mặt, nếu không có thì gán 0
        for col in symptom_cols:  # Xóa [''] khỏi danh sách
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Kiểm tra các cột bắt buộc
        required_cols = numeric_cols + categorical_cols + symptom_cols
        missing_cols = [col for col in required_cols if col not in input_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Chuẩn hóa dữ liệu số
        app.logger.debug(f"Before scaling: {input_data[numeric_cols]}")
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
        app.logger.debug(f"After scaling: {input_data[numeric_cols]}")
        
        # Sắp xếp cột theo đúng thứ tự của dữ liệu huấn luyện
        X_cols = numeric_cols + categorical_cols + symptom_cols
        input_data = input_data[X_cols]
        
        # Kiểm tra danh sách cột
        app.logger.debug(f"Input columns: {input_data.columns.tolist()}")
        
        # Dự đoán
        prediction = model.predict(input_data)[0]
        app.logger.debug(f"Prediction: {prediction}")
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)