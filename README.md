# 2015 US Flight Delays Analysis

I wanted to find out why flights get delayed in the US and whether delays can be predicted before a flight takes off. I used the 2015 flight data from the US Department of Transportation and worked with a random sample of 4 million flights.

My notebook: https://www.kaggle.com/code/bardiasajedi/2015-flight-delays-complete-analysis

## Data

The dataset is on Kaggle: https://www.kaggle.com/datasets/usdot/flight-delays

It has three files. flights.csv has about 5.8 million flights, and the other two list the airlines and airports. I joined them together so each flight has the airline name and the airport locations.

One problem I ran into: in October the airports are listed as numbers instead of codes like JFK or LAX, so those flights can't be matched to airports. I removed them, which is why October is missing from the charts.

## What I did

First I cleaned the data. The times are stored as numbers like 1430 for 2:30 pm, so I converted them. I also checked missing values and removed outliers.

Then I looked at delays from different angles:
* which airlines are the most and least on time
* how delays change by hour, weekday, month and around holidays
* which airports and routes have the most delays, shown on maps
* what causes most delays (weather, the airline, late aircraft and so on)

I also ran some statistical tests and regressions to check which differences are real, and used SARIMA to forecast how many flights there are per day.

At the end I trained a few models (logistic regression, random forest and gradient boosting) to predict whether a flight will arrive 15 or more minutes late.

## Libraries

pandas, NumPy, Matplotlib, Seaborn, SciPy, statsmodels, Plotly, scikit-learn and missingno

## Running it

The easiest way is on Kaggle. Open the notebook link above, or upload the .ipynb file to a new Kaggle notebook. Then add the "2015 Flight Delays and Cancellations" dataset as input and click Run All. The full run takes a while because of the 4 million rows.

To run it on your own computer, install the libraries first:

```
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn plotly missingno pyarrow kagglehub
```

You also need a Kaggle API token so the notebook can download the data.
