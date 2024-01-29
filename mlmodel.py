import numpy as np
import pandas as pd  
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Charger les données
fuel_df = pd.read_csv('FuelConsumption.csv')

# Sélectionner les caractéristiques nécessaires
x = fuel_df[['MODELYEAR','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y = fuel_df['CO2EMISSIONS']

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15 , shuffle=False, random_state = 0)
# print(x_test.shape, y_test.shape)
# Créer un modèle de régression polynomiale de degré 4
poly_features = PolynomialFeatures(degree=4, include_bias=False)
std_scaler = StandardScaler()
lin_reg = LinearRegression()
polynomial_regression = make_pipeline(poly_features, std_scaler, lin_reg)

# Entraîner le modèle
polynomial_regression.fit(x_train, y_train)

# Afficher les scores de corrélation
print('Correlation Train =', polynomial_regression.score(x_train, y_train))
print('Correlation Test =', polynomial_regression.score(x_test, y_test))

import pickle

filename = 'fuel.pickle'
pickle.dump(polynomial_regression, open(filename, 'wb'))

#installer pipreqsnb
#pip install pipreqsnb
#faire $pipreqs  .
#pour générer le fichier requirements.txt
