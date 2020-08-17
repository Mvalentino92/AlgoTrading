from ENV_VAR import *
from IMPORTS import *
from utils import *

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
starting_cash = float(api.get_account().cash)

# Wait for the market to open
while not api.get_clock().is_open:
    time.sleep(60)

# Now that market has opened, buy the stocks we have
for sym in to_buy:

    # Calculate how many shares we can do and attempt to buy, always use try incase error
    try:
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
        to_sell.pop(sym,None)  
        continue


# Now that the stocks have been bought, time to monitor them being sold til the end of the market day
# And obviously while we still have stocks to sell
time.sleep(60) # Wait a minute for orders to go through
positions = api.list_positions()
while api.get_clock().is_open and len(positions) > 0:

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

            # Print we sold
            print('Sold percent:',plpc)

            # New Cash
            print('Total cash:',api.get_account().cash)

            # Sell and take this off the to_sell dict
            api.submit_order(
                    symbol=sym,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
            to_sell.pop(sym,None)

        # Get new positions
        positions = api.list_positions()

        # Sleep 60 seconds and wait for the next minute
        time.sleep(60)

# Total loss or gain
print('Total profit or loss:',float(api.get_account().cash) - starting_cash)

# Add stocks we still have to sell ownedstocks file
file = open("OwnedStocks.txt","w")
file.write(str(to_sell))
file.close()
