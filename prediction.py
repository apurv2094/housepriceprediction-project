import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('https://d24c4belkzmc2t.cloudfront.net/cdn-uploads/1608527992_housing_price.csv')

df.rename(columns={'BasementArea (ft2)': 'BasementArea', 'Area (ft2)': 'Area', 'GarageArea (ft2)': 'GarageArea'}, inplace = True)
print(df.head())

df['HouseType'] = df['HouseType'].replace(np.nan, value='Flat')
df['GarageType'] = df['GarageType'].replace(np.nan, value='No_Garage')

street_encoder = LabelEncoder()
htype_encoder = LabelEncoder()
foundation_encoder = LabelEncoder()
central_encoder = LabelEncoder()
garagetype_encoder = LabelEncoder()

df['Encoded_Street'] = street_encoder.fit_transform(df['Street'].values.reshape(-1,1))
df['Encoded_House Type'] = htype_encoder.fit_transform(df['HouseType'].values.reshape(-1,1))
df['Encoded_Foundation'] = foundation_encoder.fit_transform(df['Foundation'].values.reshape(-1,1))
df['Encoded_CentralAC'] = central_encoder.fit_transform(df['CentralAC'].values.reshape(-1,1))
df['Encoded_GarageType'] = garagetype_encoder.fit_transform(df['GarageType'].values.reshape(-1,1))

x = df[['Quality', 'YearBuilt', 'BasementArea', 'Area', 'BathroomNos', 'GarageCars', 'GarageArea', 'Encoded_Foundation', 'Encoded_CentralAC', 'Encoded_GarageType']]
y = df['SalePrice']

x_train, x_test, y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train,y_train)

print(model.score(x_train, y_train))

joblib.dump(model, 'house_price_prediction.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')