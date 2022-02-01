from config import api_key, pwd
from sqlalchemy import extract
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, inspect
from flask import Flask, json, jsonify, render_template, request
import pandas as pd
from sqlalchemy import create_engine, func
import requests
import numpy as np
import time
import os
from flask_sqlalchemy import SQLAlchemy
from models import create_classes



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

# conn = psycopg2.connect(DATABASE_URL, sslmode='require')

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
# Flask Routes
#################################################
@app.route("/")
def index():
   
    # check if the db is empty
    if len(db.session.query(CryptoCurr.time).limit(1).all() == 0): 

        # API call
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&toTs=-1&api_key={api_key}"
        r = requests.get(url)
        data = r.json()
        df_daily = pd.DataFrame(data['Data']['Data'])
        # cleaning
        df_daily = df_daily[df_daily['time'] >= 1388563200]
        Newdf_daily = df_daily[['time','high','low','open','volumefrom','volumeto','close','conversionType']].copy()
        Newdf_daily['Datetime'] = pd.to_datetime(Newdf_daily['time'],unit = 's')
        Newdf_daily['Year'] = pd.to_datetime(Newdf_daily['Datetime'],errors = 'ignore').dt.year
        # add to db
        Newdf_daily.to_sql("crypto_price", con = engine, if_exists='replace', index=False)

    oldest_timestamp_in_db = db.session.query(CryptoCurr.time).limit(1).all()
    current_time = int(time.time())

    while current_time > oldest_timestamp_in_db:
        # api call to get more data
        oldest_timestamp_in_db -= 1
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&toTs=-1&api_key={api_key}"
        r = requests.get(url)
        data = r.json()
        oldest_timestamp_in_db = db.session.query(CryptoCurr.time).limit(1).all()

        price_df = pd.DataFrame(data['Data']['Data'])
        df_daily = price_df.append(df_daily)
        
    # cleaning df
    df_daily = df_daily[df_daily['time'] >= 1388563200]
    Newdf_daily = df_daily[['time','high','low','open','volumefrom','volumeto','close','conversionType']].copy()
    Newdf_daily['Datetime'] = pd.to_datetime(Newdf_daily['time'],unit = 's')
    Newdf_daily['Year'] = pd.to_datetime(Newdf_daily['Datetime'],errors = 'ignore').dt.year

    # load df into db
    # Newdf_daily.to_sql(name='crypto_daily_table', con=engine, if_exists='append', index=False) 
    Newdf_daily.to_sql("crypto_price", con = engine, if_exists='replace', index=False)

    return render_template("index.html")
 


@app.route("/first_five")
def firstfive():

    # session = Session(engine)
    # res = db.session.query(CryptoCurr.time, CryptoCurr.close).\
    #     order_by(CryptoCurr.time.desc()).\
    #     limit(5).all()
    res = db.session.query(CryptoCurr.time, CryptoCurr.close).\
        limit(5).all()
    # session.close()

    # Convert list of tuples to dict
    dict = {}
    for i in res:
        dict[i[0]] = float(i[1])

    return jsonify(dict)



if __name__ == "__main__":
    app.run(debug=True)
