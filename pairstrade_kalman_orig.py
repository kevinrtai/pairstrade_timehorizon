from __future__ import print_function
from zipline.api import order, record, symbol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import DataReader
from pykalman import KalmanFilter

def initialize(context):
  context.trading_day_counter = 0
  context.pair = [symbol('EWA'), symbol('EWC')]
  context.size = 10000

  # Kalman filter parameters
  context.delta = 1e-5
  context.trans_cov = context.delta / (1 - context.delta) * np.eye(2)
 
 
def handle_data(context, data):
  # Keep track of how many days the algorithm has been running
  context.trading_day_counter += 1

  # Construct Kalman filter since the start of trading 
  all_history = data.history(context.pair, 'close', 10 + context.trading_day_counter, '1d')
  pair_0_history = all_history[context.pair[0]]
  obs_mat = np.vstack([all_history[context.pair[0]], np.ones(all_history[context.pair[0]].shape)]).T[:, np.newaxis]
  context.kf = KalmanFilter(n_dim_obs=1, 
                            n_dim_state=2, 
                            initial_state_mean=np.zeros(2),
                            initial_state_covariance=np.ones((2, 2)),
                            transition_matrices=np.eye(2),
                            observation_matrices=obs_mat,
                            observation_covariance=1.0,
                            transition_covariance=context.trans_cov)
  means, covariances = context.kf.smooth(all_history[context.pair[1]])
  
  # Compute what the current spread is
  spread = all_history[context.pair[1]][-1] - (means[-1][0] * all_history[context.pair[0]][-1] + means[-1][1])
  if spread > 1:
    order(context.pair[0], context.size)
    order(context.pair[1], -int(means[-1][0] * context.size))
  elif spread < -1:
    order(context.pair[0], -context.size) 
    order(context.pair[1], int(means[-1][0] * context.size))
  else:
    order(context.pair[0], 0)
    order(context.pair[1], 0)

  record(pair_0=data.current(context.pair[0], 'price'))
  record(pair_1=data.current(context.pair[1], 'price'))
  record(spread=spread)
