#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request
from flask_cors import CORS
import configparser
import os
import json
import pandas as pd
import numpy as np
import joblib #save model
import shutil
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
import sys
sys.path.append('../') # 若在根目錄請註解
# sys.path.append(r'C:\Users\Alice\Desktop\API\AIupsAlgorithm-main') # 若在根目錄請註解
import os
# os.chdir(r'C:\Users\Alice\Desktop\API\AIupsAlgorithm-main')
from Algorithm import __AlgoList__
warnings.simplefilter(action='ignore', category=FutureWarning)



app = Flask(__name__)
CORS(app)


@app.route('/corr', methods=['GET','POST'])
def corr():
    data = request.get_json()
    print(data)

    df_temp = pd.DataFrame.from_dict(data)#[0], orient='index')

    y_metrology = df_temp.loc['Metrology','Metrology']

    drop_index = []
    for i in df_temp.index :

        if 'Metrology' in i or 'METROLOGY' in i:
            if i != y_metrology:
                drop_index.append(i)

    df = df_temp.drop(['Metrology'],axis=1)
    df = df.transpose()


    X_drop_id = sensorID.split(',')

    X_drop_name = []
    for i in df.columns:
        words = i.split('_')
        number = words[-1]
        for j in X_drop_id:
            if j == number:
                X_drop_name.append(i)

    X = df.drop(drop_index + X_drop_name, axis = 1)
    #y = df[y_metrology]

    X_corr = X.corr()
    X_corr['abs_y'] = np.abs(X_corr[y_metrology])
    X_corr = X_corr.sort_values(by='abs_y',ascending=False)
    X_corr = X_corr.dropna(how='all')
    test = str(list(X_corr['abs_y'][1:].index))
    print(test)
    return(test)




@app.route('/', methods=['GET','POST'])
def main():
    data = request.get_json()
    print(data)

    df_temp = pd.DataFrame.from_dict(data)#[0], orient='index')

    y_metrology = df_temp.loc['Metrology','Metrology']

    drop_index = []
    for i in df_temp.index :
        if 'Metrology' in i or 'METROLOGY' in i:
            drop_index.append(i)

    df = df_temp.drop(['Metrology'],axis=1)
    df = df.transpose()


    X_drop_id = sensorID.split(',')

    X_drop_name = []
    for i in df.columns:
        words = i.split('_')
        number = words[-1]
        for j in X_drop_id:
            if j == number:
                X_drop_name.append(i)

    X = df.drop(drop_index + X_drop_name, axis = 1)


    # 可用演算法
    __AlgoList__


    y = df[y_metrology]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    XGBoost = __AlgoList__["regression"]["xgboost"]
    model = XGBoost.Regressor(max_depth=6)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"(MSE): {mse}")
    print(model)
    # Importances
    print(f"(Importances): {dict(zip(X.columns,model.feature_importances_))}")
    df_feature_importance = pd.DataFrame([model.feature_importances_],columns= X.columns,index=[y_metrology])

    df_feature_importance = df_feature_importance.transpose()

    df_sort = df_feature_importance.sort_values(by=y_metrology, ascending=False)

    test = str(list(df_sort.index))
    return (test )





if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read('./server_setting.ini', encoding='utf-8')  # './server_setting.ini' ini路徑
    ip = conf.get('server', 'ip')
    port = conf.get('server', 'port')
    sensorID= conf.get('data', 'sensorID')


    print(ip, port)
    app.run(ip, port=port)

