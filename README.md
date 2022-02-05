# Price-Prediction-Model

This project’s goal is to develop a machine learning model that can predict a cryptocurrency's future market price. 

A LSTM model is trained on historical price data that is pulled in through an API and stored in a relational database. 

The model attempts to predict prices for a chosen time window, for both Bitcoin and Ethereum.

Our app is deployed using Heroku:

https://price-prediction-model.herokuapp.com/


# Dataset:
The daily crypto price data has been pulled in through an API on CryptoCompare:

https://min-api.cryptocompare.com/documentation?key=Historical&cat=dataHistoday

The pricing information includes: timestamp, high, low, open, volumefrom, volumeto, and close. We will most likely save all the data, but only use one of the pricing metrics to train the model.

# ETL proccess
The API data includes timestamp, high, low, open, volumefrom, volumeto, and close. in addition to these columns, we've created a coin, 

# Data storage
* We used Heroku Postgres to store data for our app.
* The database updates only when needed, based on the current and last unix timestamp in the db
Database updates up to once daily, when index page loads, based on 00:00 GMT time zone.
* Time units were daily only.
* Data for both coins was stored in 1 table, due to limitations of a free Heroku Postgres database.

# Long Short Term Memory (LSTM) Model

# Visuals












