from curses.ascii import CR
from distutils.cygwinccompiler import CygwinCCompiler
from config import api_key
from sqlalchemy import create_engine
from flask import Flask, jsonify, render_template
import pandas as pd
from sqlalchemy import create_engine
import requests
import numpy as np
import time
import os
from flask_sqlalchemy import SQLAlchemy
from models import create_classes
import datetime
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#################################################
# Flask Setup
#################################################
app = Flask(__name__)



#################################################
# Database Setup
#################################################
DATABASE_URL = os.environ.get('DATABASE_URL', '').replace("://", "ql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
# Remove tracking modifications
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Connects to the database using the app config
db = SQLAlchemy(app)

# Comment out for local testing with postgres
# engine = create_engine(f"postgresql://postgres:{pwd}@localhost:5432/Crypto_db")
# reflect an existing database into a new model
# Base = automap_base()
# reflect the tables
# Base.prepare(engine, reflect=True)
# Save reference to the table
# CryptoCurr = Base.classes.crypto_price

CryptoCurr = create_classes(db)
engine = create_engine(DATABASE_URL, echo = False)

#################################################
# Helper Functions for model predictions
#################################################

def predict_past_year(db, db_table, coin, model, scaler):
    """Function to make predictions for the past year in one day increments for a given coin and model
    Args:
        db (object): sqlalchemy database object
        db_table (object): database table to pull data from
        coin (string): [coin that is going to be predicted]
        model ([loaded LSTM model]): [trained  model loaded in from directory]
        scaler (pickle file): saved MinMaxScaler
    Returns:
        past_year_dict [dict]: [dictionary containing dates, predictions, real prices]
    """
    
    look_back = 60
    one_year_ago = datetime.date.today() - datetime.timedelta(days=(365 + look_back))
    
    results = db.session.query(db_table.timestamp_date, db_table.close).filter(db_table.coin == coin).filter(db_table.timestamp_date >= one_year_ago).order_by(db_table.timestamp_date).all()
    
    dates = [str(x[0]) for x in results]
    close_prices = [float(x[1]) for x in results]
    
    inputs = np.array(close_prices).reshape(-1,1)

    inputs_transformed = scaler.transform(inputs)

    X_test = []

    for i in range(look_back, len(inputs_transformed)):
        
        X_test.append(inputs_transformed[i-look_back:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    
    past_year_dict = {
        'dates': dates[look_back:],
        'real_prices': close_prices[look_back:],
        'predictions': [float(x) for x in list(predicted_stock_price[:,0])]
    }
    
    return past_year_dict


def predict_dates(most_recent_date, predict_days):
    
    prediction_dates = pd.date_range(most_recent_date, periods=predict_days+1).tolist()[1:]
    
    return prediction_dates


def predict_X_days(db, db_table, coin, model, predict_days, scaler):
    """Predict 'days' into the future by feeding back daily predictions into model
    Args:
        session (object): connection to sql db
        coin (string): [coin that is going to be predicted]
        model ([loaded LSTM model]): [trained  model loaded in from directory]
        predict_days ([int]): [Number of days to predict into the future]
    Returns:
        x_days_dict [dict]: [dictionary containing dates, predictions, real prices]
    """
    
    look_back = 60
    ninety_days_back = datetime.date.today() - datetime.timedelta(days=90)
    
    results = db.session.query(db_table.timestamp_date, db_table.close).filter(db_table.coin == coin).filter(db_table.timestamp_date >= ninety_days_back).order_by(db_table.timestamp_date).all()
    
    dates = [x[0] for x in results]
    close_prices = [float(x[1]) for x in results]
    
    inputs_list = close_prices[-60:]
    inputs = np.array(inputs_list).reshape(-1,1)
    
    sliding_inputs = scaler.transform(inputs)
    
    predictions = []

    for _ in range(predict_days):
        
        prediction = model.predict(sliding_inputs.reshape(1,-1,1))
        predictions.append(prediction[:,0][0])
        
        sliding_inputs = sliding_inputs.ravel().tolist()
        del sliding_inputs[0]
        sliding_inputs.append(prediction[:,0][0])
        
        sliding_inputs = np.array(sliding_inputs).reshape(1,-1,1)
        
    predictions = np.array(predictions).reshape(-1,1)
    predicted_stock_price = scaler.inverse_transform(predictions)
    
    pred_dates = predict_dates(dates[-1], predict_days)
        
    forecast_dict = {
        'real_price_dates': dates,
        'pred_dates': pred_dates,
        'real_prices': close_prices,
        'predictions': list(predicted_stock_price[:,0])
    }
    
    return forecast_dict


def predict_past_year_acc(db, db_table, coin, model, scaler, look_back):
    """Function to make predictions for the past year in one day increments

    Args:
        session (object): connection to sql db
        coin (string): [coin that is going to be predicted]
        model ([loaded LSTM model]): [trained  model loaded in from directory]

    Returns:
        past_year_dict [dict]: [dictionary containing dates, predictions, real prices]
    """
    

    one_year_ago = datetime.date.today() - datetime.timedelta(days=(365 + look_back))
    
    results = db.session.query(db_table.timestamp_date, db_table.close).filter(db_table.coin == coin).filter(db_table.timestamp_date >= one_year_ago).order_by(db_table.timestamp_date).all()
        
    dates = [str(x[0]) for x in results]
    close_prices = [float(x[1]) for x in results]
    
    features_dict = {
        'close': close_prices
    }
    
    features_df = pd.DataFrame(features_dict)
    # create 20 and 50 day moving average columns
    features_df['ma_20_day'] = features_df['close'].rolling(20, min_periods=1).mean()
    features_df['ma_50_day'] = features_df['close'].rolling(50, min_periods=1).mean()

    # create close price velocity and accelaration columns
    features_df['close_velo'] = features_df['close'].diff()
    features_df['close_acc'] = features_df['close_velo'].diff()

    features_df = features_df.iloc[50:, :]

    num_features = 5
    inputs = np.array(features_df.values).reshape(-1,num_features)

    inputs_transformed = scaler.transform(inputs)

    # print(features_df.values)
    # print(inputs)
    X_test = []

    for i in range(look_back, inputs_transformed.shape[0]):
        
        X_test.append(inputs_transformed[i-look_back:i])

    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], look_back, num_features))
    
    
    predicted_stock_acc = model.predict(X_test)
    
    # Get something which has as many features as dataset
    predicted_stock_acc_extended = np.zeros((len(predicted_stock_acc),5))

    # Put the predictions there
    predicted_stock_acc_extended[:,4] = predicted_stock_acc[:,0]

    # # Inverse transform it and select the 4th column.
    predicted_stock_acc = scaler.inverse_transform(predicted_stock_acc_extended)[:,4]

    past_year_dict = {
        'dates': list(dates[50+look_back:]),
        'real_acc': list(features_df['close_acc'].values[look_back:]),
        'pred_acc': [float(x) for x in list(predicted_stock_acc)]
    }
    
    return past_year_dict
#################################################
# Flask Routes
#################################################

@app.route("/")
def index():
   
    coins = ['ETH', 'BTC']

    for coin in coins:

        # check if the db is empty
        if len(db.session.query(CryptoCurr.time).filter(CryptoCurr.coin == coin).limit(1).all()) == 0: 

            #API call
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin}&tsym=USD&limit=2000&toTs=-1&api_key={api_key}"
            r = requests.get(url)
            data = r.json()
            df_daily = pd.DataFrame(data['Data']['Data'])

            # cleaning
            Newdf_daily = df_daily[['time','high','low','open','volumefrom','volumeto','close']].copy()
            Newdf_daily.insert(2,"coin", coin)
            Newdf_daily.insert(2,"currency","USD")
            Newdf_daily.dropna()
            Newdf_daily['timestamp_date'] = pd.to_datetime(Newdf_daily['time'],unit = 's')
            Newdf_daily['timestamp_year'] = pd.to_datetime(Newdf_daily['timestamp_date'],errors = 'ignore').dt.year


            # add to db
            Newdf_daily.to_sql("crypto_price", con = engine, if_exists='append', index=False)

            oldest_timestamp_in_df = db.session.query(CryptoCurr.time).\
                filter(CryptoCurr.coin == coin).\
                order_by(CryptoCurr.time).\
                limit(1).all()[0][0]
            new_df_daily = pd.DataFrame()
           
            while oldest_timestamp_in_df > 1388563200:

                # API call
                oldest_timestamp_in_df -= 1
                url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin}&tsym=USD&limit=2000&toTs={oldest_timestamp_in_df}&api_key={api_key}"
                r = requests.get(url)
                data = r.json()
                df_daily = pd.DataFrame(data['Data']['Data'])
                
                # cleaning df
                df_daily = df_daily[['time','high','low','open','volumefrom','volumeto','close']].copy()
                df_daily.insert(2,"coin", coin)
                df_daily.insert(2,"currency","USD")
                df_daily.dropna()
                df_daily['timestamp_date'] = pd.to_datetime(df_daily['time'],unit = 's')
                df_daily['timestamp_year'] = pd.to_datetime(df_daily['timestamp_date'],errors = 'ignore').dt.year
                new_df_daily = new_df_daily.append(df_daily)
                oldest_timestamp_in_df = new_df_daily.iloc[0]['time']

            # only dates jan. 1 2014 and later
            new_df_daily = new_df_daily[new_df_daily['time'] >= 1388563200]
            # add to db
            new_df_daily.to_sql("crypto_price", con = engine, if_exists='append', index=False)

            new_df_daily = pd.DataFrame()
            
        # get the most recent date
        most_recent_timestamp_in_db = db.session.query(CryptoCurr.time).\
            filter(CryptoCurr.coin == coin).\
            order_by(CryptoCurr.time.desc()).\
            limit(1).all()[0][0]
        current_date = time.time()
        
        # Count num days to get date on, from most recent in db until today
        limit = current_date - most_recent_timestamp_in_db
        days = int(limit/60/60/24)
        
        price_daily_clean_df = pd.DataFrame()
        # api call to get more data
        if days > 1:

            # API call
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin}&tsym=USD&limit={days-1}&toTs=-1&api_key={api_key}"
            r = requests.get(url)
            data = r.json()
            price_df = pd.DataFrame(data['Data']['Data'])
            # drop today's data since it is not accurate (partial data: i.e. volume is only current to time of call)
            price_df = price_df.loc[price_df['time'] != price_df['time'].max()]

            # cleaning df
            price_daily_clean_df = price_df[['time','high','low','open','volumefrom','volumeto','close']].copy()
            price_daily_clean_df.insert(2,"coin", coin)
            price_daily_clean_df.insert(2,"currency","USD")
            price_daily_clean_df.dropna()
            price_daily_clean_df['timestamp_date'] = pd.to_datetime(price_daily_clean_df['time'],unit = 's')
            price_daily_clean_df['timestamp_year'] = pd.to_datetime(price_daily_clean_df['timestamp_date'],errors = 'ignore').dt.year
            
            price_daily_clean_df.to_sql("crypto_price", con = engine, if_exists='append', index=False)

        # load df into db
        # Newdf_daily.to_sql(name='crypto_daily_table', con=engine, if_exists='append', index=False) 
        # price_daily_clean_df.to_sql("crypto_price", con = engine, if_exists='append', index=False)


    return render_template("index.html")
 
 
#################################################
# Routes to render html files
#################################################
@app.route("/about")
def render_about():
    return render_template('about.html')

@app.route("/bitcoin")
def render_bitcoin():
    return render_template('bitcoin.html')
  
@app.route("/ETH")
def render_ETH():
    return render_template('ETH.html')
  
@app.route("/comparison")
def render_comparison():
    return render_template('comparison.html')

#JUST TEST ROUTE
@app.route("/first_five")
def firstfive():

    res = db.session.query(CryptoCurr.coin, CryptoCurr.time, CryptoCurr.close).\
        order_by(CryptoCurr.time).\
        limit(5).all()

    # Convert list of tuples to dict
    dict = {}
    for i in res:
        dict[i[0]] = float(i[1])

    return jsonify(dict)

@app.route("/hist_data")
def hist_data():
    """Route to gather and return historical data for all crypto currencies
    Returns:
        hist_data_json [json]: json object containing historical data
    """
    
    
    hist_data_dict = {}
    
    hist_data_json = jsonify(hist_data_dict)

    return hist_data_json


#################################################
# Routes to get coin data (no predictions)
#################################################

@app.route("/bitcoin_daily_data")
def get_bitcoin_daily():

    results = db.session.query(CryptoCurr.high, CryptoCurr.low, CryptoCurr.open, CryptoCurr.close, CryptoCurr.timestamp_date, CryptoCurr.volumefrom, CryptoCurr.volumeto).filter(CryptoCurr.coin == 'BTC').filter(CryptoCurr.timestamp_year > 2016).all()
    
    response = []
    
    for item in results:
        response.append(
            {"high": item[0],
             "low": item[1],
             "open": item[2],
             "close": item[3],
             "timestamp_date": str(item[4]),
             "volumefrom": item[5],
             "volumeto": item[6]}
        )

    bitcoin_daily_data = jsonify(response)
    
    return bitcoin_daily_data

@app.route("/ETH_daily_data")
def get_ETH_daily():
    
    results = db.session.query(CryptoCurr.high, CryptoCurr.low, CryptoCurr.open, CryptoCurr.close, CryptoCurr.timestamp_date, CryptoCurr.volumefrom, CryptoCurr.volumeto).filter(CryptoCurr.coin == 'ETH').filter(CryptoCurr.timestamp_year > 2016).all()
    response = []
    
    for item in results:
        response.append(
            {"high": item[0],
             "low": item[1],
             "open": item[2],
             "close": item[3],
             "timestamp_date": str(item[4]),
             "volumefrom": item[5],
             "volumeto": item[6]}
        )

    ETH_daily_data = jsonify(response)
    
    return ETH_daily_data

@app.route("/comparison_page_data")
def get_comp_page_data():
    pass


#################################################
# Model Routes to get model prediction data
#################################################
@app.route("/model_predictions_BTC")
def get_predictions_BTC():
    
    model_loaded = tf.keras.models.load_model('Model_Testing/Crypto_Models/Trained_model_2_daily_BTC_4L_50N_0p1D_trainUpTo2021.h5', compile = False)

    scaler = pickle.load(open('scaler.pkl', 'rb'))
    assert isinstance(scaler, MinMaxScaler)
    scaler.clip = False  # add this line
    
    coin = 'BTC'
    
    past_year_dict = predict_past_year(db, CryptoCurr, coin, model_loaded, scaler)

    BTC_model_preds_json = jsonify(past_year_dict)

    return BTC_model_preds_json

@app.route("/model_predictions_acc_BTC")
def get_predictions_BTC_acc():
    
    model_loaded = tf.keras.models.load_model('Model_Testing/Crypto_Models/TM_8_daily_acc_BTC_SEEPIC_trainUpTo2021.h5', compile = False)

    scaler = pickle.load(open('Model_Testing/Crypto_Models/Scalers/scaler_acc_8.pkl', 'rb'))
    assert isinstance(scaler, StandardScaler)
    scaler.clip = False  # add this line
    
    coin = 'BTC'
    
    past_year_dict = predict_past_year_acc(db, CryptoCurr, coin, model_loaded, scaler, 30)

    BTC_model_preds_acc_json = jsonify(past_year_dict)

    return BTC_model_preds_acc_json


@app.route("/model_predictions_ETH")
def get_predictions_ETH():
    model_loaded = tf.keras.models.load_model('Model_Testing/Crypto_Models/TM_14_MODEL1_ETH_4L_50N_0p1D_100epo_trainUptoLastYear.h5', compile = False)

    scaler = pickle.load(open('Model_Testing/Crypto_Models/Scalers/scaler_TM_14.pkl', 'rb'))
    assert isinstance(scaler, MinMaxScaler)
    scaler.clip = False  # add this line
    
    coin = 'ETH'
    
    past_year_dict = predict_past_year(db, CryptoCurr, coin, model_loaded, scaler)

    ETH_model_preds_json = jsonify(past_year_dict)

    return ETH_model_preds_json


@app.route("/model_predictions_acc_ETH")
def get_predictions_ETH_acc():
    
    model_loaded = tf.keras.models.load_model('Model_Testing/Crypto_Models/TM_16_MODEL2_ETH_SEEPIC_trainUpToLastYear.h5', compile = False)

    scaler = pickle.load(open('Model_Testing/Crypto_Models/Scalers/scaler_TM16_ETH_MODEL2.pkl', 'rb'))
    assert isinstance(scaler, StandardScaler)
    scaler.clip = False  # add this line
    
    coin = 'ETH'
    
    past_year_dict = predict_past_year_acc(db, CryptoCurr, coin, model_loaded, scaler, 30)

    ETH_model_preds_acc_json = jsonify(past_year_dict)

    return ETH_model_preds_acc_json   




if __name__ == "__main__":
    app.run(debug=True)