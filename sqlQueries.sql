Select * from cryptoprice
limit 5;

/*How has Bitcoin’s price in USD varied over time*/
Select DATE_PART('Year',timestamp_date) as Year,  Max(close-open) as Variance
from cryptoprice
group by Year
order by year ;

/*How has Bitcoin’s trading volume increased or decreased over time*/
Select DATE_PART('Year',timestamp_date) as year, SUM(volumefrom) as bitcoinVol 
from cryptoprice
group by Year
order by bitcoinVol desc ;

/*Which day was Bitcoin most profitable?*/
Select timestamp_date as Day,  Max(close-open) as Profit
from cryptoprice
group by day
order by Profit desc;




/*Latest Close Price */
select (timestamp_date) as closingdate,Max(close) as closing_price

from cryptoprice
group by timestamp_date
Order by timestamp_date desc

