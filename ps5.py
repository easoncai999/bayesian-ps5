import requests
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from private import API_KEY # Store API Key in a separate file for privacy

# Make sure set up your free Alpha Vantage API key to reproduce all code
symbol = "SPY"

# Construct the API URL for monthly adjusted time series data (free endpoint)
url = (
    f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}"
    f"&apikey={API_KEY}"
)
response = requests.get(url)

# Parse the JSON response
data_raw = response.json()

# Convert the dictionary to a DataFrame
ts_data = data_raw["Monthly Adjusted Time Series"]
df = pd.DataFrame.from_dict(ts_data, orient='index')
df.index = pd.to_datetime(df.index)

# Rename columns for clarity
df = df.rename(columns={
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. adjusted close": "adjusted_close",
    "6. volume": "volume",
    "7. dividend amount": "dividend_amount"
})

# Ensure the data is sorted by date increasing order for plotting
df = df.sort_index()
cols_name = ["open", "high", "low", "close", "adjusted_close", "volume", "dividend_amount"]
for i in cols_name:
    df[i] = pd.to_numeric(df[i])

df.head()

df = df.loc[df.index >= '2004-01-01']

plt.figure(figsize=(12,6))
plt.plot(df.index, df['adjusted_close'], color='#0D2D40')
plt.title('SPY Adjusted Close Price (Proxy for S&P 500)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')

# Mark the 2008 market crash
plt.axvline(x=pd.to_datetime('2008-09-01'), color='#2C7F63', linestyle='dashed', linewidth=1)
plt.text(pd.to_datetime('2008-09-01'), df['adjusted_close'].max(), '2008 Market Crash', color='#2C7F63', verticalalignment='top')

# Mark the 2022 covid market impact
plt.axvline(x=pd.to_datetime('2022-01-01'), color='#939BBB', linestyle='dashed', linewidth=1)
plt.text(pd.to_datetime('2022-01-01'), df['adjusted_close'].min(), '2022 COVID Impact', color='#939BBB', verticalalignment='bottom')

plt.show()

def prior_function(macro_data, news_sentiment, prior_sd=0.05):
    """
    Combine macroeconomic data and news sentiment into a prior distribution.
    Returns a dictionary representing a normal distribution with a calculated mean and a specified standard deviation.
    """
    macro_score = np.mean(macro_data)
    sentiment_score = np.mean(news_sentiment)
    # Weighted sum (70% macro, 30% sentiment)
    prior_mean = 0.7 * macro_score + 0.3 * sentiment_score
    return {"distribution": "normal", "mean": prior_mean, "sd": prior_sd}

macro_example = [0.01, 0.02, 0.015, 0.03]     # monthly growth rates (in decimal)
news_example = [0.012, 0.011, 0.013, 0.012]   # news sentiment scores (in decimal)

prior_dist = prior_function(macro_example, news_example, prior_sd=0.05)
print("Prior Distribution:", prior_dist)

def likelihood_function(price_data, inflate_sd=1):
    """
    Calculate log returns from price data and return a likelihood distribution
    (assumed to be normal) with estimated mean and standard deviation.
    The inflation factor makes the likelihood less (or more) precise.
    """
    price_data = np.array(price_data)
    returns = np.diff(np.log(price_data))
    mu = np.mean(returns)
    sigma = np.std(returns) * inflate_sd  # Using no inflation (inflate_sd=1) so that likelihood is more informative.
    return {"distribution": "normal", "mean": mu, "sd": sigma}

# For the likelihood, we use the last 100 data points from adjusted_close
price_data_example = df['adjusted_close'].tail(100).values  
likelihood_dist = likelihood_function(price_data_example, inflate_sd=1)
print("Likelihood Distribution:", likelihood_dist)

def posterior_function(prior, likelihood):
    """
    Combine a normal prior and a normal likelihood (conjugate case)
    to form a normal posterior.
    """
    prior_mean = prior['mean']
    prior_sd = prior['sd']
    likelihood_mean = likelihood['mean']
    likelihood_sd = likelihood['sd']
    
    posterior_mean = (likelihood_sd**2 * prior_mean + prior_sd**2 * likelihood_mean) / (prior_sd**2 + likelihood_sd**2)
    posterior_sd = np.sqrt((prior_sd**2 * likelihood_sd**2) / (prior_sd**2 + likelihood_sd**2))
    
    return {"distribution": "normal", "mean": posterior_mean, "sd": posterior_sd}

# Compute the posterior.
posterior_dist = posterior_function(prior_dist, likelihood_dist)
print("Posterior Distribution:", posterior_dist)

# Create an x-axis grid that covers the range of interest.
x = np.linspace(-0.2, 0.25, 1000)

plt.figure(figsize=(12, 6))
plt.plot(x, norm.pdf(x, prior_dist['mean'], prior_dist['sd']), label="Prior", color='#9AD5F8')
plt.plot(x, norm.pdf(x, likelihood_dist['mean'], likelihood_dist['sd']), label="Likelihood", color='#5E74DD')
plt.plot(x, norm.pdf(x, posterior_dist['mean'], posterior_dist['sd']), label="Posterior", color='#0D2D40')
plt.title("Integrating Qualitative Insights and Price Dynamics via Bayesian Inference")
plt.xlabel("Average Log Return of SPY")
plt.ylabel("Density")
plt.legend()
plt.show()