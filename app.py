from config import api_key, pw
from sqlalchemy import extract
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, inspect
from flask import Flask, json, jsonify
import pandas as pd
from sqlalchemy import create_engine, func
import requests
import numpy as np



#################################################
# Flask Setup
#################################################
app = Flask(__name__)



#################################################
# Database Setup
#################################################
engine = create_engine(f"postgresql://postgres:{pw}@localhost:5432/crypto")

# api call and data cleaning
# url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&toTs=-1&api_key={api_key}"
# r = requests.get(url)
# data = r.json()
# df_daily = pd.DataFrame(data['Data']['Data'])

# maxTs = df_daily.iloc[0]['time']
# limit = 2000
# histo = 'histoday'
# oldestTs = 1388563200
# count = 0

# while maxTs >= oldestTs:

#     maxTs = df_daily.iloc[0]['time']
    
#     maxTs -= 1
    
#     # API call with the oldest ts
#     url = f"https://min-api.cryptocompare.com/data/v2/{histo}?fsym=BTC&tsym=USD&limit={limit}&toTs={maxTs}&api_key={api_key}"
#     r = requests.get(url)
#     data = r.json()
#     price_df = pd.DataFrame(data['Data']['Data'])
#     df_daily = price_df.append(df_daily)
#     count += 1
    
# df_daily = df_daily[df_daily['time'] >= 1388563200]
# Newdf_daily = df_daily[['time','high','low','open','volumefrom','volumeto','close','conversionType']].copy()
# Newdf_daily['Datetime'] = pd.to_datetime(Newdf_daily['time'],unit = 's')
# Newdf_daily['Year'] = pd.to_datetime(Newdf_daily['Datetime'],errors = 'ignore').dt.year

# # load df into db
# Newdf_daily.to_sql(name='crypto_daily_table', con=engine, if_exists='replace', index=False) 
# # set primary key
# with engine.connect() as con:
#     con.execute('ALTER TABLE crypto_daily_table ADD PRIMARY KEY (time);')

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

# Save reference to the table
daily_ref = Base.classes.crypto_daily_table




#################################################
# Flask Routes
#################################################
@app.route("/")
def index():
    # check if the current timestamp is greater than the last timestamp in db
     with open('templates/index.html') as f:
        return f.read()


@app.route("/first_five")
def firstfive():
    session = Session(engine)

    res = session.query(daily_ref.time, daily_ref.close).limit(5).all()
    # Convert list of tuples to dict
    dict = {}
    for i in res:
        dict[i[0]] = i[1]

    session.close()
    return jsonify(dict)


if __name__ == "__main__":
    # app.run(debug=True)
    app.run()