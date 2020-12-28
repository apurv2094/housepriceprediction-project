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
    print("yes")
    Quality = request.form['Quality']
    YearBuilt = request.form['YearBuilt']
    BasementArea = request.form['BasementArea']
    print("yes")
    Area = request.form['Area']
    print("yes")
    BathroomNos = request.form['BathroomNos']
    print("yes")
    GarageCars = request.form['GarageCars']
    print("yes")
    GarageArea = request.form['GarageArea']
    print("yes")
    Encoded_Foundation = request.form['Foundation']
    print("yes")
    Encoded_CentralAC = request.form.get('CentralAC')
    print("yes")
    Encoded_GarageType = request.form.get('GarageType')
    print("yes")
    
    Quality, YearBuilt, BasementArea, Area, BathroomNos, GarageCars, GarageArea, Encoded_Foundation, Encoded_CentralAC, Encoded_GarageType = int(Quality), int(YearBuilt), int(BasementArea), int(Area), int(BathroomNos), int(GarageCars), int(GarageArea), int(Encoded_Foundation), int(Encoded_CentralAC), int(Encoded_GarageType)  
    scaled_values = scaler.transform([[Quality, YearBuilt, BasementArea, Area, BathroomNos, GarageCars, GarageArea, Encoded_Foundation, Encoded_CentralAC, Encoded_GarageType]])
    result = model.predict(scaled_values)

    string = 'House Price: ' + str(result[0])

    return render_template('index.html', prediction_text = string)

# running the app
if __name__ == '__main__':
    app.run(port =5000, threaded = True)