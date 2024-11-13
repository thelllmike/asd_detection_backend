from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Define input schema
class ScreeningData(BaseModel):
    A1: int
    A2: int
    A3: int
    A4: int
    A5: int
    A6: int
    A7: int
    A8: int
    A9: int
    A10: int
    Age_Mons: int
    Sex: str
    Ethnicity: str
    Jaundice: str
    Family_mem_with_ASD: str
    Who_completed_the_test: str

app = FastAPI()

# Load model and preprocessing objects
model = joblib.load('ASD_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define categories for categorical fields if missing
categories = {
    "Sex": ["Male", "Female"],
    "Ethnicity": ["Asian", "White-European", "Black", "Hispanic", "Latino", "Others"],
    "Jaundice": ["yes", "no"],
    "Family_mem_with_ASD": ["yes", "no"],
    "Who_completed_the_test": ["Self", "Health professional", "Family member", "Others"]
}

# Ensure all required encoders are in place
for col, classes in categories.items():
    if col not in label_encoders:
        le = LabelEncoder()
        le.fit(classes)
        label_encoders[col] = le

@app.post("/predict")
async def predict(data: ScreeningData):
    input_data = data.dict()

    # Apply label encoding to categorical fields
    categorical_columns = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who_completed_the_test"]
    transformed_data = {}

    for col in categorical_columns:
        if col in input_data:
            if col in label_encoders:
                if input_data[col] not in label_encoders[col].classes_:
                    input_data[col] = label_encoders[col].classes_[0]
                transformed_data[col] = label_encoders[col].transform([input_data[col]])[0]
            else:
                return {"error": f"Label encoder for {col} is missing after initialization."}
    
    input_data.update(transformed_data)
    
    try:
        # Prepare feature values with a placeholder for the 17th feature
        feature_values = [
            input_data["A1"], input_data["A2"], input_data["A3"], input_data["A4"],
            input_data["A5"], input_data["A6"], input_data["A7"], input_data["A8"],
            input_data["A9"], input_data["A10"], input_data["Age_Mons"],
            input_data["Sex"], input_data["Ethnicity"], input_data["Jaundice"],
            input_data["Family_mem_with_ASD"], input_data["Who_completed_the_test"],
            0  # Placeholder for the missing 17th feature
        ]

        # Convert feature values to float
        feature_values = [float(value) for value in feature_values]

        # Scale numerical features if required
        features_array = np.array(feature_values).reshape(1, -1)
        numerical_features = [10]
        features_array[:, numerical_features] = scaler.transform(features_array[:, numerical_features])

        # Make prediction
        prediction = model.predict(features_array)
        result = "Yes" if prediction[0] == 1 else "No"

        return {"ASD_Traits": result}
    
    except ValueError as e:
        return {"error": f"ValueError during feature processing: {str(e)}"}
