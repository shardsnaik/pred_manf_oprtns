import joblib
import model
# model = joblib.load('C:\\Users\\Public\\MACHINE LEARNING\\pred_manf_oprtns\\model.pkl')
model = joblib.load('C:\\Users\\Public\\MACHINE LEARNING\\pred_manf_oprtns\\models\\machine_downtime.pkl')
scaler = joblib.load('C:\\Users\\Public\\MACHINE LEARNING\\pred_manf_oprtns\\scaler.pkl')

sample_ds = [[100.0, 50.0, 75.0, 60.0, 70.0, 80.0, 0.5, 0.3, 1500.0, 220.0, 100.0, 200.0]]

def predict(data):
    normalized_data = scaler.transform(data)
    prediction = model.predict(normalized_data)
    confidence = model.predict_proba(normalized_data)
    return {
        "downtime": "Yes" if prediction[0] == 1 else "No",
        'confidence':f'{max(confidence[0]):.2f}'
    }


print(predict(sample_ds))