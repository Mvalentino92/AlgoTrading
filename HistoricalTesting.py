import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import torch
from ENV_Variables import *
from CreateFunctions import *

# The Environment class. Contains the standard openAIGym functions
class Environment:

    # Holds the api and av object, intraday data, starting moving averages,
    # and current index for data. Also what symbol of course!
    def __init__(self,key_id,secret_key,base_url,symbol,ks=[10,50,100,200]):

        # The api and av object, will be used by everything else in init
        self.api = tradeapi.REST(key_id,secret_key,base_url,'v2')
        self.av = self.api.alpha_vantage

        # Max equity started with
        self.starting_equity = self.api.get_account().equity

        # Set the ks (for moving averages), (sorted)
        self.ks = np.sort(ks)
        self.idx = np.max(self.ks)

        # All the intraday data available for this symbol as pd (every minute)
        # Flips data (earlier -> latest), and cleans it.
        self.data = clean_data(self.av.intraday_quotes(symbol, '1min', 'full', 'pandas'))

        # Get all the moving average data for every field.
        self.moving_averages = moving_averages(self.data, self.ks)

        # Get last values used for moving averages
        self.last_values = np.array(self.data.iloc[self.idx-1])

    # Returns the current state as an np.array
    # Updates index to next
    def next_state(self):

        # The account object so get equity value
        account = self.api.get_account()
        equity = np.array([account.equity])

        # Next datum of data
        datum = self.data.iloc[self.idx]

        # Timestamp for time and grab the time normalized for max time
        timestamp = datum.name
        t = np.array([((timestamp.time().hour*60 + timestamp.time().minute) - OPEN_TIME)/(CLOSE_TIME - OPEN_TIME)])

        # Convert datum to np.array (since we're done with it)
        datum = np.array(datum)

        # Update moving averages before grabbing values for moving averages
        for key in self.moving_averages.keys():
            self.moving_averages[key] += (datum - self.last_values)/key

        # Update last_values to be datum
        self.last_values = datum

        # Set moving averages
        moving_average_values = np.array(list(self.moving_averages.values())).flatten()

        # Join everything but time together, then normalize
        all_but_time = np.concatenate((equity, datum, moving_average_values)).astype(np.float64)
        all_but_time = all_but_time / np.linalg.norm(all_but_time)

        # Finally join that with time (which was normalized seperately)
        retval = np.concatenate((t,all_but_time))

        # update idx (since were done, and return)
        self.idx += 1
        return retval





# Create a test environement
env = Environment(APCA_API_KEY_ID,APCA_API_SECRET_KEY,APCA_API_BASE_URL,'GOOGL')
