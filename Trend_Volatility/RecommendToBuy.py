from ENV_VAR import *
from IMPORTS import *
from utils import *
from FiltersAndTests import *

# Get APIS
api = tradeapi.REST(APCA_API_KEY_ID,APCA_API_SECRET_KEY,APCA_API_BASE_URL,'v2')
av = api.alpha_vantage

# Get tradable symbols
symbols = filter_stocks(api,av,n=300,t=15,exchange='NASDAQ',low=10,high=20)

# (trend/volatility,mean_reversion)
Ns = (30,100)

# Get two weight vectors
w1 = get_weights(lambda x: x**2,100,k=0.2)
w2 = np.array([0.25,0.50,0.25])
w3 = get_weights(lambda x: x,Ns[0]-1,k=0.8)
w4 = np.array([0.50,0.25,0.25])

# Get recommended to buy and put in plot
recos = recommend_to_buy(api,av,symbols,w1,w2,w3,w4,Ns,M=15)
