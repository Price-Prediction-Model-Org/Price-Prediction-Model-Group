-- Table: public.crypto_price

-- DROP TABLE public.crypto_price;

CREATE TABLE crypto_price(
    id SERIAL PRIMARY KEY,
    time INT NOT NULL,
    high DECIMAL NOT NULL,
    Currency VARCHAR(20) NOT NULL,
    Coin VARCHAR(20) NOT NULL,
    low DECIMAL NOT NULL,
    open DECIMAL NOT NULL,
    volumefrom DECIMAL NOT NULL,
    volumeto DECIMAL NOT NULL,
    close DECIMAL NOT NULL,
    conversionType VARCHAR(20) NOT NULL,
    timestamp_date DATE NOT NULL,
    timestamp_Year INT NOT NULL
);
