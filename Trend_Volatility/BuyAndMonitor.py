from ENV_VAR import *
from IMPORTS import *
from utils import *

# Get the current date and create the directory for this date
current_date = datetime.date.today()
date_string = current_date.strftime('%Y-%m-%d')
dirname = date_string + '_ClosingPrices'
os.mkdir(dirname)

# Set calls
calls = 0

# Load log file for errors
logfile = open("log.txt","w")

# Set pscale and lscale
pscale = 3.85
lscale = 5

# Get APIS
api = tradeapi.REST(APCA_API_KEY_ID,APCA_API_SECRET_KEY,APCA_API_BASE_URL,'v2')
av = api.alpha_vantage

# Read in all stocks to buy from Buy file
file = open("BuyStocks.txt","r")
contents = file.read()
to_buy = ast.literal_eval(contents)
file.close()

# Read in the dictionary selling tolerances
file = open("SellStocks.txt","r")
contents = file.read()
to_sell = ast.literal_eval(contents)
file.close()

# CONFIRM WE HAVE ALL THE SAME STOCKS HERE
if to_buy.keys() != to_sell.keys():
    print('BUY AND SELL SYMBOL LIST DONT MATCH: EXITING')
    exit()

# Read and extend this dictionary by the stocks we already own
file = open("OwnedStocks.txt","r")
contents = file.read()
to_sell.update(ast.literal_eval(contents))
file.close()

# Get original cash
while True:
    try:
        calls = (calls + 1) % APCA_MAX_CALLS
        starting_cash = float(api.get_account().cash)
        break
    except Exception:
        logfile.write('Failed to get cash from starting account at {}\n'.format(get_time()))
        time.sleep(sleeptime(calls))
time.sleep(sleeptime(calls,t=0))

# Wait for the market to open
isopen = False
while not isopen:
    try:
        calls = (calls + 1) % APCA_MAX_CALLS
        isopen = api.get_clock().is_open
        time.sleep(sleeptime(calls,t=(0 if isopen else 60)))
    except Exception:
        logfile.write('Failed to wait for market open at {}\n'.format(get_time()))
        time.sleep(sleeptime(calls))

# Now that market has opened, buy the stocks we have
for sym in to_buy:

    # Calculate how many shares we can do and attempt to buy, always use try incase error
    try:
        calls = (calls + 1) % APCA_MAX_CALLS
        price = api.get_barset(sym,'1Min')[sym][0].c
        investment = to_buy.get(sym)
        shares = divmod(investment,price)[0]
        api.submit_order(
                symbol=sym,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day'
            )
    except Exception:

        # If something went wrong, we need to take this stock off of to_sell
        logfile.write('Failed to buy stock {} at time {}\n'.format(sym,get_time()))
        to_sell.pop(sym,None)  
    time.sleep(sleeptime(calls,t=0)) # Sleep

# Get original investment
#investment = starting_cash - float(api.get_account().cash)

# Wait for orders to go through, grab entry prices and create arrays for plotting price over time
# Note, keep plotting for any stocks we have, regardless of selling.
time.sleep(5)
while True:
    try:
        calls = (calls + 1) % APCA_MAX_CALLS
        positions = api.list_positions()
        break
    except Exception:
        logfile.write('Failed to get initial positions at {}\n'.format(get_time()))
        time.sleep(sleeptime(calls))
time.sleep(sleeptime(calls,t=0)) # Sleep
starting_positions = [pos.symbol for pos in positions]
entry_prices = [float(pos.avg_entry_price) for pos in positions]
graph_data = [[0] for i in range(len(starting_positions))]
sold_at = [None]*len(starting_positions)
taxis = []

# Now that the stocks have been bought, time to monitor them being sold til the end of the market day
# And obviously while we still have stocks to sell
while isopen and len(positions) > 0:

    print('Checking at time {}'.format(get_time()))

    # Iterate every position and check if we need to sell
    for pos in positions:

        # Get info about stock
        sym = pos.symbol
        qty = int(pos.qty)
        plpc = float(pos.unrealized_plpc)

        # Get info about our tolerances
        # If None, use default tolerances of 1 percent
        tols = to_sell.get(sym)
        if tols is None:
            tols = DEFAULT_TOLS

        # Grab (ptol,ltol)
        ptol = tols[0]*pscale
        ltol = tols[1]*lscale

        # Check if need to sell
        if plpc >= ptol or plpc <= ltol:


            # Sell and take this off the to_sell dict (always use try catch)
            try:

                calls = (calls + 1) % APCA_MAX_CALLS
                api.submit_order(
                        symbol=sym,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                to_sell.pop(sym,None)
                
                # Record that we sold this symbol
                index_of_sold = starting_positions.index(sym)
                sold_at[index_of_sold] = (get_time(),pos.current_price)

            except Exception:

                # Print error to log
                logfile.write('Failed to SELL symbol {} at {}\n'.format(sym,get_time()))

            time.sleep(sleeptime(calls,t=0)) # Sleep

    # Add data points to graph real quick
    for i in range(len(starting_positions)):
        
        # Grab this position to look up barset data
        pos = starting_positions[i]

        # Wrap in try of course
        try:
            calls = (calls + 1) % APCA_MAX_CALLS
            current_price = api.get_barset(pos,'1Min',1)[pos][0].c
        except Exception:
            logfile.write('Failed to get current price for graphing at {}\n'.format(get_time()))
            curent_price = graph_data[i][-1] # Just use last value if problem
        
        # Get append price and time then sleep
        taxis.append(get_time())
        graph_data[i].append(curent_price) 
        time.sleep(sleeptime(calls,t=0))

    # Get new positions
    while True:
        try:
            calls = (calls + 1) % APCA_MAX_CALLS
            positions = api.list_positions()
            break
        except Exception:
            logfile.write('Failed to get new postions at {}\n'.format(get_time()))
            time.sleep(sleeptime(calls))
    time.sleep(sleeptime(calls,t=0))

    # Sleep 60 seconds and wait for the next minute, reset calls to 0! (willing to wait extra 60 seconds worst case)
    time.sleep(60)
    calls = 0

    # Get if open
    while True:
        try:
            calls = (calls + 1) % APCA_MAX_CALLS
            isopen = api.get_clock().is_open
            break
        except Exception:
            logfile.write('Failed to check if market still open at {}'.format(get_time()))
            time.sleep(sleeptime(calls))
    time.sleep(sleeptime(calls,t=0))

# Add stocks we still have to sell ownedstocks file
file = open("OwnedStocks.txt","w")
file.write(str(to_sell))
file.close()

# Close logfile
logfile.close()

# Convert plots and plot prices
graph_data = np.array(graph_data)
for i in range(len(starting_positions)):
    p = graph_data[i][1:] # Cause we started with 0 as padding in beginning
    plt.plot(taxis,p)
    plt.plot(taxis,np.full(len(taxis),entry_prices[i]))
    if sold_at[i] is not None: # Only if sold, add this scatter to see where sold
        plt.scatter(*sold_at[i])
    plt.title(starting_positions[i])
    plt.legend(('ClosingPrices','BuyIn','SoldAt'))
    plt.savefig(dirname+'/'+starting_positions[i])
