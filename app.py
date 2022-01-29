from config import api_key, pwd
from sqlalchemy import extract
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, inspect
from flask import Flask, json, jsonify
import pandas as pd
from sqlalchemy import create_engine, func
import requests
import numpy as np
import time




#################################################
# Flask Setup
#################################################
app = Flask(__name__)



#################################################
# Database Setup
#################################################
engine = create_engine(f"postgresql://postgres:{pwd}@localhost:5432/Crypto_db")
# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)
# Save reference to the table
CryptoCurr = Base.classes.crypto_price


#################################################
# Flask Routes
#################################################
@app.route("/")
def index():

    print('index loaded')
    # if latest_timestamp_in_db < current_time:  
    #     # api call to get more data
    #     url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&toTs=-1&api_key={api_key}"
    #     r = requests.get(url)
    #     data = r.json()
    #     df_daily = pd.DataFrame(data['Data']['Data'])

    #     while current_time >= latest_timestamp_in_db:

    #         latest_timestamp_in_db = session.query(daily_ref.time).limit(1).all()
            
    #         latest_timestamp_in_db -= 1
            
    #         # API call with the oldest ts
    #         url = f"https://min-api.cryptocompare.com/data/v2/{histo}?fsym=BTC&tsym=USD&limit={limit}&toTs={latest_timestamp_in_db}&api_key={api_key}"
    #         r = requests.get(url)
    #         data = r.json()
    #         price_df = pd.DataFrame(data['Data']['Data'])
    #         df_daily = price_df.append(df_daily)
        
    # # cleaning df
    # df_daily = df_daily[df_daily['time'] >= 1388563200]
    # Newdf_daily = df_daily[['time','high','low','open','volumefrom','volumeto','close','conversionType']].copy()
    # Newdf_daily['Datetime'] = pd.to_datetime(Newdf_daily['time'],unit = 's')
    # Newdf_daily['Year'] = pd.to_datetime(Newdf_daily['Datetime'],errors = 'ignore').dt.year

    # # load df into db
    # Newdf_daily.to_sql(name='crypto_daily_table', con=engine, if_exists='append', index=False) 
    # session.close()

    # check if the current timestamp is greater than the last timestamp in db
    with open('templates/index.html') as f:
        return f.read()


@app.route("/first_five")
def firstfive():

    session = Session(engine)
    res = session.query(CryptoCurr.time, CryptoCurr.close).limit(5).all()
    # Convert list of tuples to dict
    dict = {}
    for i in res:
        dict[i[0]] = float(i[1])

    session.close()
    return jsonify(dict)



if __name__ == "__main__":
    app.run(debug=True)
