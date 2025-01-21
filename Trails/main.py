from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import uvicorn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib


app = FastAPI()

data_set = None
model = None

@app.get('/')
def home():
    return {"message": "Welcome to the API",
            "endpoints": [
            {"path": "/upload", "description": "Upload a CSV file to train the model"},
            {"path": "/train", "description": "Train the model using the uploaded dataset"},
            {"path": "/predict", "description": "Make predictions using the trained model"}
            ]}
          


@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    global data_set
    if file.filename.endswith('.csv'):
         try:
             data = pd.read_csv(file.file)
             for col in  data.columns:
                 if data[col].dtypes == 'object':
                     raise HTTPException(status_code=400, detail="File must not contain string values")
             data.fillna(data.mean(), inplace=True)

             data_set = data
             print(data_set.head())
             return {"message": "File uploaded successfully"}
         except Exception as e:
             raise HTTPException(status_code=400, detail=str(e))
    
    
    else: 
        raise HTTPException(status_code=400, detail="File must be in CSV format")
    
@app.post('/train')
async def train():
    global data_set, model

    if data_set is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Use the /upload endpoint first.")
    
    try:
        X = data_set.iloc[:, :-1]
        Y = data_set.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        joblib.dump(model, "model.pkl")

        return {"accuracy": accuracy, "f1_score": f1}

    except Exception as e:  
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)