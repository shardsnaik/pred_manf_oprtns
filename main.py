from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import uvicorn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pydantic import BaseModel

app = FastAPI()


class PredictionValues(BaseModel):
    Hydraulic_Pressurebar: float
    Coolant_Pressurebar: float
    Air_System_Pressurebar: float
    Coolant_Temperature: float
    Hydraulic_Oil_Temperature: float
    Spindle_Bearing_Temperature: float
    Spindle_Vibration: float
    Tool_Vibration: float
    Spindle_SpeedRPM: float
    Voltagevolts: float
    TorqueNm: float
    CuttingkN: float


data_set = None
model = None
scaler = None


@app.get('/')
def home():
    """
    Home endpoint providing API information.
    """
    return {
        "message": "Welcome to the API",
        "endpoints": [
            {"path": "/upload",
             "description": "Upload a CSV file to train the model"},
            {"path": "/train",
             "description": "Train the model using the uploaded dataset"},
            {"path": "/predict",
             "description": "Make predictions using the trained model"}
        ]


    }


@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    """
    Upload a CSV file to be used for training the model.
    """
    global data_set
    if file.filename.endswith('.csv'):
        try:
            data = pd.read_csv(file.file)
            # Ensure no string values in the dataset
            for col in data.columns:
                if data[col].dtypes == 'object':
                    raise HTTPException(
                        status_code=400,
                        detail="File must not contain string values")
            # Fill missing values with column mean
            data.fillna(data.mean(), inplace=True)
            data_set = data
            return {"message": "File uploaded successfully"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(
            status_code=400, detail="File must be in CSV format")


@app.post('/train')
async def train():
    """
    Train the model using the uploaded dataset.
    """
    global data_set, model, scaler

    if data_set is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Use the /upload endpoint first.")

    try:
        X = data_set.iloc[:, :-1]
        Y = data_set.iloc[:, -1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, Y, test_size=0.2, random_state=42)

        model = GradientBoostingClassifier(
            learning_rate=0.1, max_depth=5, min_samples_leaf=3,
            min_samples_split=10, n_estimators=300, subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        return {
            "message": "Model trained successfully!",
            "f1_score": f1,
            "accuracy": accuracy
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/predict')
async def prediction(request: PredictionValues):
    """
    Make predictions using the trained model.
    """
    global model

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Use the /train endpoint first.")

    try:
        scaler = joblib.load("scaler.pkl")
        inp = [[
            request.Hydraulic_Pressurebar,
            request.Coolant_Pressurebar,
            request.Air_System_Pressurebar,
            request.Coolant_Temperature,
            request.Hydraulic_Oil_Temperature,
            request.Spindle_Bearing_Temperature,
            request.Spindle_Vibration,
            request.Tool_Vibration,
            request.Spindle_SpeedRPM,
            request.Voltagevolts,
            request.TorqueNm,
            request.CuttingkN,
        ]]
        pred = scaler.transform(inp)
        prediction = model.predict(pred)[0]
        confidence = max(model.predict_proba(pred)[0])
        confidence = f'{confidence:.2f}'

        return {
            "Downtime": "Yes" if int(prediction) == 1 else "No",
            "confidence": float(confidence)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
