# Manufacturing Downtime Prediction API 🎛️

## Overview 📜
This API is designed to predict machine downtime in manufacturing environments using machine learning models. The application is built using **FastAPI** and supports dataset upload, model training, and downtime prediction. The endpoints have been tested using Postman and FastAPI's built-in documentation. 🚀

## Dataset 📂
The dataset used for this project is sourced from [Kaggle](#). Three categorical columns were removed, and the target feature `Downtime` was converted into numeric format to simplify the model's processing. The refined dataset is used and for testing the model use our dataset provided in [Dataset] folder ensures for compatibility with numerical input for keeping it simple model predictions. 📊

## Endpoints 🌐
The API includes the following endpoints:

### 1. Welcome Message 🎉
**Endpoint:**
```
GET http://127.0.0.1:8000/
```
**Expected Output:**
```json
{
    "message": "Welcome to the API",
    "endpoints": [
        {
            "path": "/upload",
            "description": "Upload a CSV file to train the model"
        },
        {
            "path": "/train",
            "description": "Train the model using the uploaded dataset"
        },
        {
            "path": "/predict",
            "description": "Make predictions using the trained model"
        }
    ]
}
```

### 2. Upload Dataset 📤
**Endpoint:**
```
POST http://127.0.0.1:8000/upload
```
**Procedure:**
1. Navigate to the **Body** section in Postman.
2. Select **form-data**.
3. Add a key-value pair:
   - **Key:** file (ensure the type is set to "File").
   - **Value:** Select the CSV file to upload.
4. Send the request.

**Expected Output:**
```json
{
    "message": "File uploaded successfully"
}
```

### 3. Train the Model 🏋️‍♂️
**Endpoint:**
```
POST http://127.0.0.1:8000/train
```
**Expected Output:**
```json
{
    "message": "Model trained successfully!",
    "f1_score": 0.996,
    "accuracy": 0.996
}
```

### 4. Predict Downtime 🔮
**Endpoint:**
```
POST http://127.0.0.1:8000/predict
```
**Procedure:**
1. Navigate to the **Body** section in Postman.
2. Select **raw** and set the format to **JSON**.
3. Provide the input values in JSON format.

**Sample Input:**
```json
{
    "Hydraulic_Pressurebar": 70,
    "Coolant_Pressurebar": 6,
    "Air_System_Pressurebar": 7.54,
    "Coolant_Temperature": 30,
    "Hydraulic_Oil_Temperature": 80,
    "Spindle_Bearing_Temperature": 30,
    "Spindle_Vibration": 1.3,
    "Tool_Vibration": 53,
    "Spindle_SpeedRPM": 1000,
    "Voltagevolts": 330,
    "TorqueNm": 27,
    "CuttingkN": 3.50
}
```

**Expected Output:**
```json
{
    "prediction": "Yes",
    "confidence": 0.99
}
```

## Instructions to Set Up and Run the API 🔧

### Prerequisites
- Python 3.9 or above installed on your machine. 🐍
- Clone this repository to your local system.

### Steps
1. Navigate to the project directory:
   ```bash
   cd manufacturing-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```

4. Open your browser or Postman and navigate to:
   - **Documentation:** `http://127.0.0.1:8000/docs`
   - **Welcome Message Endpoint:** `http://127.0.0.1:8000/`

### Example Requests and Responses

#### Welcome Endpoint 🎉
**Request:**
```
GET http://127.0.0.1:8000/
```
**Response:**
```json
{
    "message": "Welcome to the API",
    "endpoints": [
        {
            "path": "/upload",
            "description": "Upload a CSV file to train the model"
        },
        {
            "path": "/train",
            "description": "Train the model using the uploaded dataset"
        },
        {
            "path": "/predict",
            "description": "Make predictions using the trained model"
        }
    ]
}
```

#### Train Endpoint 🏋️‍♂️
**Request:**
```
POST http://127.0.0.1:8000/train
```
**Response:**
```json
{
    "message": "Model trained successfully!",
    "f1_score": 0.996,
    "accuracy": 0.996
}
```

#### Predict Endpoint 🔮
**Request:**
```
POST http://127.0.0.1:8000/predict
```
**Sample Input:**
```json
{
    "Hydraulic_Pressurebar": 70,
    "Coolant_Pressurebar": 6,
    "Air_System_Pressurebar": 7.54,
    "Coolant_Temperature": 30,
    "Hydraulic_Oil_Temperature": 80,
    "Spindle_Bearing_Temperature": 30,
    "Spindle_Vibration": 1.3,
    "Tool_Vibration": 53,
    "Spindle_SpeedRPM": 1000,
    "Voltagevolts": 330,
    "TorqueNm": 27,
    "CuttingkN": 3.50
}
```
**Response:**
```json
{
    "prediction": "Yes",
    "confidence": 0.99
}
```

## Limitations 🛑
- The model may slightly overfit due to the limited dataset size. 🐾
- Missing features may reduce prediction accuracy. 📉

## Improvements ✨
1. Enhance the `main.py` script to dynamically handle any uploaded dataset with appropriate data types. 🧰
2. Expand the dataset to include more features and records. 📈
3. Implement cross-validation to further reduce overfitting. 🔄


## Technical Notes 🛠️
- The project relies on:
  - Python 3.9 🐍
  - FastAPI ⚡
  - Scikit-learn 🤖
  - Pandas 🐼
- All dependencies are listed in the `requirements.txt` file. 📋

## Installation 🖥️
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd manufacturing-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

