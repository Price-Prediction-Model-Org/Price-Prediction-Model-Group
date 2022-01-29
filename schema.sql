-- Table: public.crypto_price

-- DROP TABLE public.crypto_price;

CREATE TABLE IF NOT EXISTS public.crypto_price
(
    id integer NOT NULL DEFAULT nextval('crypto_price_id_seq'::regclass),
    "time" integer,
    high double precision,
    "Currency" character varying COLLATE pg_catalog."default",
    "Coin" character varying COLLATE pg_catalog."default",
    low double precision,
    open double precision,
    volumefrom double precision,
    volumeto double precision,
    close double precision,
    "conversionType" character varying COLLATE pg_catalog."default",
    timestamp_date date,
    "timestamp_Year" integer,
    CONSTRAINT crypto_price_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.crypto_price
    OWNER to postgres;