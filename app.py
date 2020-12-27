from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('house_price_prediction.pkl')
scaler = joblib.load('standard_scaler.pkl')

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/prediction', methods = ['POST'])
def prediction():

    Quality = request.form['Quality']
    YearBuilt = request.form['YearBuilt']
    BasementArea = request.form['BasementArea']
    Area = request.form['Area']
    BathroomNos = request.form['BathroomNos']
    GarageCars = request.form['GarageCars']
    GarageArea = request.form['GarageArea']
    Encoded_Foundation = request.form['Foundation']
    Encoded_CentralAC = request.form['CentralAC']
    Encoded_GarageType = request.form['GarageType']
    
    Quality, YearBuilt, BasementArea, Area, BathroomNos, GarageCars, GarageArea, Encoded_Foundation, Encoded_CentralAC, Encoded_GarageType = int(Quality), int(YearBuilt), int(BasementArea), int(Area), int(BathroomNos), int(GarageCars), int(GarageArea), int(Encoded_Foundation), int(Encoded_CentralAC), int(Encoded_GarageType)  
    scaled_values = scaler.transform([[Quality, YearBuilt, BasementArea, Area, BathroomNos, GarageCars, GarageArea, Encoded_Foundation, Encoded_CentralAC, Encoded_GarageType]])
    result = model.predict(scaled_values)

    string = 'House Price: ' + str(result[0])

    return render_template('index.html', prediction_text = string)

# running the app
if __name__ == '__main__':
    app.run(port =5000, threaded = True)