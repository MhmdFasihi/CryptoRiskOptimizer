import numpy as np 
import pandas as pd 
from scipy.stats import norm 
import time 
import matplotlib.pyplot as plt 
import yfinance as yf 
from datetime import date 
from statsmodels.stats.stattools import jarque_bera
##calculate the number of days between 2 dates in python
d0=date(2020, 11, 1)
d1=date(2023, 6, 12)
delta=d1-d0
mu = 0.01
sigma = 0.10
value_invested=10000
num_assets=2
##Historicall Value at Risk function 
def value_at_risk(value_invested, returns, weights, alpha=0.95, lookback_days=delta.days):
    returns = returns.fillna(0.0)
    # Multiply asset returns by weights to get one weighted portfolio return
    portfolio_returns = returns.iloc[-lookback_days:].dot(weights)
    # Compute the correct percentile loss and multiply by value invested
    return np.percentile(portfolio_returns, 100 * (1-alpha)) * value_invested
##Cvar or Expected shortfall 
def cvar(value_invested, returns, weights, alpha=0.95, lookback_days=520):
    # Call out to our existing function
    var = value_at_risk(value_invested, returns, weights, alpha, lookback_days=lookback_days)
    returns = returns.fillna(0.0)
    portfolio_returns = returns.iloc[-lookback_days:].dot(weights)
    
    # Get back to a return rather than an absolute loss
    var_pct_loss = var / value_invested
    
    return value_invested * np.nanmean(portfolio_returns[portfolio_returns < var_pct_loss])
##based on normal distributiuon value at risk 
# Portfolio mean return is unchanged, but std has to be recomputed
# This is because independent variances sum, but std is sqrt of variance
#portfolio_std = np.sqrt( np.power(sigma, 2) * num_assets ) / num_assets

# manually 
#historicalvalueatrisk=(mu - portfolio_std * norm.ppf(0.95)) * value_invested
#print(historicalvalueatrisk)

#def value_at_risk_N(mu=0, sigma=1.0, alpha=0.95):
#    return mu - sigma*norm.ppf(alpha)


#x = np.linspace(-3*sigma,3*sigma,1000)
#y = norm.pdf(x, loc=mu, scale=portfolio_std)
#plt.plot(x,y)
#plt.axvline(value_at_risk_N(mu = 0.01, sigma = portfolio_std, alpha=0.95), color='red', linestyle='solid');
#plt.legend(['Return Distribution', 'VaR for Specified Alpha as a Return'])
#plt.title('VaR in Closed Form for a Normal Distribution')





##Download data from yahoo finance and calculate the returns and define the weights of the each assets, define the value invested 
oex=['BTC-USD','ETH-USD','BNB-USD']
startdate='2020-11-01'
enddate='2023-06-12'
data=yf.download(oex,start=startdate,end=enddate)
data=data['Adj Close']
returns=data.pct_change()
returns=returns-returns.mean(skipna=True)
weights = np.array([0.65, 0.25, .10])
a=value_at_risk(value_invested,returns,weights=weights,alpha=0.95,lookback_days=delta.days)
##Calculate VaR for X days
print(np.round(a*np.sqrt(30),2))

lookback_days =delta.days
alpha = 0.95

# Multiply asset returns by weights to get one weighted portfolio return
portfolio_returns = returns.fillna(0.0).iloc[-lookback_days:].dot(weights)

portfolio_VaR = value_at_risk(value_invested, returns, weights, alpha=0.95)
# Need to express it as a return rather than absolute loss
portfolio_VaR_return = portfolio_VaR / value_invested

plt.hist(portfolio_returns, bins=20)
plt.axvline(portfolio_VaR_return, color='red', linestyle='solid')
plt.legend(['VaR for Specified Alpha as a Return', 'Historical Returns Distribution'])
plt.title('Historical VaR')
plt.xlabel('Return')
plt.ylabel('Observation Frequency')
plt.show()
##lets check ke return of the portfolio that can be normal or not 
_, pvalue, _, _ = jarque_bera(portfolio_returns)

if pvalue > 0.05:
    print('The portfolio returns are likely normal.')
else:
    print('The portfolio returns are likely not normal.')
##You'll notice the VaR computation conspicuously uses a lookback window. This is a parameter to the otherwise 'non-parametric' historical VaR. Keep in mind that because lookback window affects VaR, it's important to pick a lookback window that's long enough for the VaR to converge. To check if our value has seemingly converged let's run an experiment
N = 2000
VaRs = np.zeros((N, 1))
for i in range(N):
    VaRs[i] = value_at_risk(value_invested, returns, weights, lookback_days=i)

plt.plot(VaRs)
plt.xlabel('Lookback Window')
plt.ylabel('VaR')
plt.show()
##stationarity of the portfolio returns over this time period.
from statsmodels.tsa.stattools import adfuller

results = adfuller(portfolio_returns)
pvalue = results[1]

if pvalue < 0.05:
    print('Process is likely stationary.')
else:
    print('Process is likely non-stationary.')

##calculate the Cvar/expectedShortfall with function that calculated 
b=cvar(value_invested, returns, weights, lookback_days=delta.days)
##Calculate Cvar/ExpectedShortfall for X days
print(np.round(b*np.sqrt(30),2))
##lets visualize 
lookback_days = delta.days
alpha = 0.95

# Multiply asset returns by weights to get one weighted portfolio return
portfolio_returns = returns.fillna(0.0).iloc[-lookback_days:].dot(weights)

portfolio_VaR = value_at_risk(value_invested, returns, weights, alpha=0.95)
# Need to express it as a return rather than absolute loss
portfolio_VaR_return = portfolio_VaR / value_invested

portfolio_CVaR = cvar(value_invested, returns, weights, alpha=0.95)
# Need to express it as a return rather than absolute loss
portfolio_CVaR_return = portfolio_CVaR / value_invested

# Plot only the observations > VaR on the main histogram so the plot comes out
# nicely and doesn't overlap.
plt.hist(portfolio_returns[portfolio_returns > portfolio_VaR_return], bins=20)
plt.hist(portfolio_returns[portfolio_returns < portfolio_VaR_return], bins=10)
plt.axvline(portfolio_VaR_return, color='red', linestyle='solid')
plt.axvline(portfolio_CVaR_return, color='red', linestyle='dashed')
plt.legend(['VaR for Specified Alpha as a Return',
            'CVaR for Specified Alpha as a Return',
            'Historical Returns Distribution', 
            'Returns < VaR'])
plt.title('Historical VaR and CVaR')
plt.xlabel('Return')
plt.ylabel('Observation Frequency')
plt.show()
##Checking for convergence again for Cvar/ExpectedShortfall 
N = 1000
CVaRs = np.zeros((N, 1))
for i in range(N):
    CVaRs[i] = cvar(value_invested, returns, weights, lookback_days=i)

plt.plot(CVaRs)
plt.xlabel('Lookback Window')
plt.ylabel('VaR')
plt.show()














