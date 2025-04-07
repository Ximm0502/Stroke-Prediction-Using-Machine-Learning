import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu
df = pd.read_csv("stroke_prediction_dataset.csv", delimiter=";")

# Tiền xử lý
df[["Systolic Pressure", "Diastolic Pressure"]] = df["Blood Pressure Levels"].str.split("/", expand=True)
df["Systolic Pressure"] = df["Systolic Pressure"].astype(int)
df["Diastolic Pressure"] = df["Diastolic Pressure"].astype(int)
df["HDL"] = df["Cholesterol Levels"].str.extract(r"HDL: (\d+)").astype(float)
df["LDL"] = df["Cholesterol Levels"].str.extract(r"LDL: (\d+)").astype(float)
df["Symptoms"] = df["Symptoms"].fillna("")
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
symptoms_encoded = pd.DataFrame(mlb.fit_transform(df["Symptoms"].apply(lambda x: x.split(", ") if x else [])),
                                columns=mlb.classes_, index=df.index)
df = pd.concat([df, symptoms_encoded], axis=1)

serious_symptoms = ["Difficulty Speaking", "Loss of Balance", "Seizures", "Numbness", "Blurred Vision", "Confusion"]
df["Has_Serious_Symptom"] = df[serious_symptoms].sum(axis=1) > 0
conditions = [
    (df["Age"] > 40).astype(int),
    (df["Hypertension"] == 1).astype(int),
    (df["Heart Disease"] == 1).astype(int),
    (df["Average Glucose Level"] > 140).astype(int),
    ((df["Systolic Pressure"] > 140) | (df["Diastolic Pressure"] > 90)).astype(int),
    df["Has_Serious_Symptom"].astype(int),
    (df["Stroke History"] == 1).astype(int)
]
df["Diagnosis_New"] = (sum(conditions) >= 2).astype(int)

# Thêm nhiễu (tùy chọn, có thể bỏ nếu không cần)
np.random.seed(42)
noise_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
df.loc[noise_indices, "Diagnosis_New"] = 1 - df.loc[noise_indices, "Diagnosis_New"]

numeric_cols = ["Average Glucose Level", "Body Mass Index (BMI)", "Stress Levels",
                "Systolic Pressure", "Diastolic Pressure", "HDL", "LDL"]
categorical_cols = ["Gender", "Marital Status", "Work Type", "Residence Type",
                    "Smoking Status", "Alcohol Intake", "Physical Activity",
                    "Stroke History", "Family History of Stroke", "Dietary Habits"]
symptom_cols = mlb.classes_.tolist()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df[numeric_cols + categorical_cols + symptom_cols]
y = df["Diagnosis_New"]

scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Huấn luyện mô hình
best_xgb = XGBClassifier(
    colsample_bytree=1.0, learning_rate=0.05, max_depth=9, n_estimators=200,
    reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, eval_metric="logloss", random_state=42
)
best_xgb.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = best_xgb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Lưu mô hình và scaler
joblib.dump(best_xgb, "stroke_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(symptom_cols, "symptom_cols.pkl")  # Lưu danh sách triệu chứng để đồng bộ với Flask