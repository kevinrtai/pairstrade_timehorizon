from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import DataReader
from pykalman import KalmanFilter


def draw_date_coloured_scatterplot(etfs, prices):
  """
  Create a scatterplot of the two ETF prices, which is
  coloured by the date of the price to indicate the 
  changing relationship between the sets of prices    
  """
  # Create a yellow-to-red colourmap where yellow indicates
  # early dates and red indicates later dates
  plen = len(prices)
  colour_map = plt.cm.get_cmap('YlOrRd')    
  colours = np.linspace(0.1, 1, plen)
  
  # Create the scatterplot object
  scatterplot = plt.scatter(
      prices[etfs[0]], prices[etfs[1]], 
      s=30, c=colours, cmap=colour_map, 
      edgecolor='k', alpha=0.8
  )
  
  # Add a colour bar for the date colouring and set the 
  # corresponding axis tick labels to equal string-formatted dates
  colourbar = plt.colorbar(scatterplot)
  colourbar.ax.set_yticklabels(
      [str(p.date()) for p in prices[::plen//9].index]
  )
  plt.xlabel(prices.columns[0])
  plt.ylabel(prices.columns[1])
  plt.show()


def calc_slope_intercept_kalman(etfs, prices):
  """
  Utilise the Kalman Filter from the pyKalman package
  to calculate the slope and intercept of the regressed
  ETF prices.
  """
  delta = 1e-5
  trans_cov = delta / (1 - delta) * np.eye(2)
  obs_mat = np.vstack(
      [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
  ).T[:, np.newaxis]
  
  kf = KalmanFilter(
      n_dim_obs=1, 
      n_dim_state=2,
      initial_state_mean=np.zeros(2),
      initial_state_covariance=np.ones((2, 2)),
      transition_matrices=np.eye(2),
      observation_matrices=obs_mat,
      observation_covariance=1.0,
      transition_covariance=trans_cov
  )
  
  state_means, state_covs = kf.filter(prices[etfs[1]].values)
  return state_means, state_covs    
    

def draw_slope_intercept_changes(prices, state_means):
  """
  Plot the slope and intercept changes from the 
  Kalman Filte calculated values.
  """
  pd.DataFrame(
      dict(
          slope=state_means[:, 0], 
          intercept=state_means[:, 1]
      ), index=prices.index
  ).plot(subplots=True)
  plt.show()


def generate_prediction(prices, state_means, names):
  slope = state_means[:, 0]
  intercept = state_means[:, 1]
  predicted = slope * prices[names[0]] + intercept
  return predicted


def draw_dynamic_prediction(prices, state_means, names):
  slope = state_means[:, 0]
  intercept = state_means[:, 1]
  predicted = generate_prediction(prices, state_means, names)
  pd.DataFrame(
    {
      names[0]: prices[names[0]],
      names[1]: prices[names[1]],
      'predicted': predicted
    }, index=prices.index
  ).plot()
  plt.show()


if __name__ == "__main__":
  # Choose the ETF symbols to work with along with 
  # start and end dates for the price histories
  # names = ['TLT', 'IEI']
  # start_date = "2012-8-01"
  # end_date = "2016-08-01"    
  
  # Obtain the adjusted closing prices from Yahoo finance
  # prices = DataReader(
  #     names, 'yahoo', start_date, end_date
  # )['Adj Close']

  # Load csvs
  csvs = ['10Yr30Yr.csv', 'es_djia.csv', 'FiveYrTenYr.csv', 'wti_brent.csv']
  titles = [('10Yr', '30Yr'), ('Dow', 'SPX'), ('FiveYr', 'TenYr'), ('WTI', 'Brent')]
  i = 3
  csv = csvs[i]
  names = titles[i]
  prices = pd.read_csv(csv, index_col=0, parse_dates=True) 

  # draw_date_coloured_scatterplot(names, prices)
  state_means, state_covs = calc_slope_intercept_kalman(names, prices)
  # draw_slope_intercept_changes(prices, state_means)
  # draw_dynamic_prediction(prices, state_means, names)

  # Plot differences between predicted and actual w/ 2std lines
  predicted = generate_prediction(prices, state_means, names)
  difference = prices[names[1]] - predicted
  stddev = np.std(difference)
  pd.DataFrame(
    {'difference': difference, 
     'sigma': np.repeat(stddev, difference.size), 
     '-sigma': np.repeat(-stddev, difference.size)
    }, index=prices.index).plot()
  plt.show()

