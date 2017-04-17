from __future__ import print_function
from zipline.api import order_target, record, symbol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import DataReader
from pykalman import KalmanFilter

def initialize(context):
  context.trading_day_counter = -1
  context.pair = [symbol('IEI'), symbol('TLT')]
  context.size = 1000

  # Kalman filter parameters
  context.delta = 1e-5
  context.trans_cov = context.delta / (1 - context.delta) * np.eye(2)

  # Trading parameters
  context.in_trade = False
  context.trading = True
  context.days_held = 0
  context.max_days = 5
  context.spread = 0.5
  context.rebalancing = 5
 
def handle_data(context, data):
  # Keep track of how many days the algorithm has been running
  context.trading_day_counter += 1

  # Construct Kalman filter since the start of trading 
  all_history = data.history(context.pair, 'close', 10 + context.trading_day_counter, '1d')
  pair_0_history = all_history[context.pair[0]]
  if context.trading_day_counter % context.rebalancing == 0:
    obs_mat = np.vstack([all_history[context.pair[0]], np.ones(all_history[context.pair[0]].shape)]).T[:, np.newaxis]
    context.kf = KalmanFilter(n_dim_obs=1, 
                              n_dim_state=2, 
                              initial_state_mean=np.zeros(2),
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              observation_matrices=obs_mat,
                              observation_covariance=1.0,
                              transition_covariance=context.trans_cov)
    context.means, covariances = context.kf.smooth(all_history[context.pair[1]])
  
  # Compute what the current spread is
  spread = all_history[context.pair[1]][-1] - (context.means[-1][0] * all_history[context.pair[0]][-1] + context.means[-1][1])

  if context.trading:
    # Calculate size. Put 60% in long, 60% in short
    context.size = 0.6 * context.portfolio.cash / all_history[context.pair[0]][-1]

    if spread > context.spread:
      order_target(context.pair[0], -context.size)
      order_target(context.pair[1], int(context.means[-1][0] * context.size))

      # Exit if it hasn't mean reverted
      if context.in_trade and context.days_held > context.max_days:
        print('bailing long')
        order_target(context.pair[0], 0)
        order_target(context.pair[1], 0)
        context.trading = False

      context.in_trade = True
      context.days_held += 1
      
    elif spread < -context.spread:
      order_target(context.pair[0], context.size) 
      order_target(context.pair[1], -int(context.means[-1][0] * context.size))

      # Exit if it hasn't mean reverted
      if context.in_trade and context.days_held > context.max_days:
        print('bailing short')
        order_target(context.pair[0], 0)
        order_target(context.pair[1], 0)
        context.trading = False

      context.in_trade = True
      context.days_held += 1

    else:
      order_target(context.pair[0], 0)
      order_target(context.pair[1], 0)
      context.in_trade = False
      context.days_held = 0
  elif spread > -context.spread and spread < context.spread:
    print('resetting')
    context.trading = True
    order_target(context.pair[0], 0)
    order_target(context.pair[1], 0)
  else:
    order_target(context.pair[0], 0)
    order_target(context.pair[1], 0)

  record(pair_0=data.current(context.pair[0], 'price'))
  record(pair_1=data.current(context.pair[1], 'price'))
  record(spread=spread)
