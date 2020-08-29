from IMPORTS import *
from ENV_VAR import *
from utils import *

# Returns a list of the stocks meeting filtering criteria
# Pass how many you want to get total, and time willing to wait (in minutes)
def filter_stocks(api, av, n=100, t=10, low=1, high=10, exchange=None):

    # Stocks we already own
    currently_own = [p.symbol for p in api.list_positions()]

    # Initial filter for active, tradable and exchange
    tradable_symbols = [stock for stock in api.list_assets(status='active') if stock.tradable]
    if exchange is not None:
        tradable_symbols = [stock.symbol for stock in tradable_symbols if stock.exchange == exchange]
    else:
        tradable_symbols = [stock.symbol for stock in tradable_symbols]

    # Shuffle and begin to filter based on price range
    random.shuffle(tradable_symbols)
    symbols = []
    calls = 0
    for symbol in tradable_symbols:

        # Check if gotten the desired stocks or went above time
        if not n or not t:
            break

        # Check if we already have shares for this stock, if so, continue
        if symbol in currently_own:
            continue

        # Get current price and add one to calls (for getting price)
        try:
            calls = (calls + 1) % (APCA_MAX_CALLS - 1)  # Minus 1 because of initial call up top for safety
            close = api.get_barset(symbol, '1Min', 1)[symbol][0].c
        except Exception:
            if not calls: # Duplicate in case api fail
                time.sleep(60)
                t -= 1
            continue
        
        # Get close (set to Inf if had no price to ensure it's not added)
        if low <= close <= high:
            symbols.append(symbol)
            n -= 1

        # Check if have to sleep to not exceed max calls
        if not calls:
            time.sleep(60)
            t -= 1

    # Remember to sleep for a minute before returning
    time.sleep(60)
    return symbols

# Function that returns the stocks recommended to buy
# w1: Weight vector for the mean gradients
# w2: Weight vector for the curve fits (should match length of polys)
# w3: Weight vector for daily volatility calculations
# w4: Weight vector for final scores
def recommend_to_buy(api, av, symbols, w1, w2, w3, w4, Ns, M = 5, day_spacing=2,
                     polys=[linear, quadratic, cubic]):

    # Get the current date and create the directory for this date
    current_date = datetime.date.today()
    date_string = current_date.strftime('%Y-%m-%d')
    dirname = date_string + '_Recommend'
    if os.path.exists(dirname): # Delete and remake to clear
        shutil.rmtree(dirname)
    os.mkdir(dirname)

    # Get a date we want prices from
    # To ensure tight spacing between recent pricing
    N = np.max(Ns) # Get max
    N_tv, N_mr = Ns # unpack
    days_go_back = int(N*1.5*day_spacing)
    delta_date = datetime.timedelta(days=days_go_back)
    lower_date = (current_date - delta_date).strftime('%Y-%m-%d')

    # Lists all things to track
    syms = []
    trend_strengths = []
    daily_avg_volatilities = []
    diff_from_means = []
    intraday_avg_volatilities = []
    all_closes = []
    all_intraday_closes = []

    # Begin to look up historical data for each up to date specified
    calls = 0
    tracking_number = 0
    track_after = 5
    for symbol in symbols:

        # Track
        tracking_number += 1
        if tracking_number % track_after == 0:
            print('Working on stock #{}'.format(tracking_number))

        # Grab raw values, filter by date, turn to np and reverse for proper order
        try:
            calls = (calls + 1) % AV_MAX_CALLS # Put before so it still increments if we exit try clause
            hist = av.historic_quotes(symbol)
            hist = np.array([list(hist.get(i).values()) for i in hist if i >= lower_date],
                   dtype=np.float)[::-1]
            hist[N] # See if we at least have (N+1) of history, so not dead stock
            closes = hist[-N:, CLOSE_INDEX] # If so, certainly we can get last N
        except Exception: # Duplicate sleep api call check in except, since we continue
            if not calls:
                time.sleep(60)
            continue

        # See if we can get intraday volatilities, if not then skip this stock
        if not calls: # Sleep if need to before another call
            time.sleep(60)
        try:
            calls = (calls + 1) % AV_MAX_CALLS
            intradays = av.intraday_quotes(symbol,'1min')
            intraday_closes = np.array([list(sample.values())
                              for sample in intradays.values()],dtype=np.float)[::-1,CLOSE_INDEX]
            intraday_closes[390*5] # See if we can get at least 3 days worth (missing price tol = 10/5)
        except Exception:
            if not calls: # Check for sleep again because of continue
                time.sleep(60)
            continue

        # ** PASSED INITIAL FILTERING, onto tests **

        # Split data, normalize some
        closes_tv = closes[-N_tv:] / np.sum(closes[-N_tv:]) # Normalized
        closes_mr = closes[-N_mr:] 

        # ---------------------------------------------------------------------------------------------------------------
        # TEST 1) Curve fitting for trend strength
        #         PASS: Positive value
        #         SCORE: As is
        pass1 = False
        strengths = []
        xs = np.linspace(0, 1, N_tv)
        for poly in polys:

            # Get coefs for this curve fit
            coefs, _ = opt.curve_fit(poly, xs, closes_tv)

            # Grab all values given from the fit (using more values but same bounds)
            xs_dense = np.linspace(0, 1, len(w1))
            h = xs_dense[1] - xs_dense[0]
            y = poly(xs_dense, *coefs)

            # Add mean gradient to trend_strength
            strengths.append(np.sum(np.gradient(y, h) * w1))

        # Get the mean of the all the trend strengths
        trend_strength = np.sum(np.array(strengths) * w2)
        pass1 = trend_strength > 0
        # -------------------------------------------------------------------------------------------------------------

        # Check if passed test 1, if not continue
        if not pass1:
            continue

        # -------------------------------------------------------------------------------------------------------------
        # TEST 2): Checking that weighted average volatility
        #          PASS: Moves enough, but not too much (see below)
        #          SCORE: 1 - value
        pass2 = False
        daily_diffs = np.abs(closes_tv[1:] - closes_tv[:-1])/closes_tv[:-1]
        daily_avg_volatility = np.sum(daily_diffs*w3)
        pass2 = 0.01 < daily_avg_volatility < 0.05
        # -----------------------------------------------------------------------------------------------------------

        # Check if passed test 2, if not continue
        if not pass2:
            continue

        # ----------------------------------------------------------------------------------------------------------
        # TEST 3): Mean reversion
        #          PASS: negative (little more negative)
        #          SCORE: Absolute value
        diff_mean_tol = 1e-2
        pass3 = False
        close_mean = np.mean(closes_mr)
        diff_from_mean = (closes_mr[-1] - close_mean)/close_mean
        pass3 = diff_from_mean < -diff_mean_tol
        # --------------------------------------------------------------------------------------------------------

        # No one fails this test, just get's penalized for not passing
        if not pass3:
            0 # nothing
        else:
            print('Passed mr with {}'.format(diff_from_mean))

        # ----------------------------------------------------------------------------------------------------------
        # TEST 4): ??
        #          PASS: ??
        #          SCORE: ??

        # ** ALL TESTS ARE DONE, store all values and calculate suggested tolerance with intraday **

        # Appending values
        syms.append(symbol)
        trend_strengths.append(trend_strength) # Test 1
        daily_avg_volatilities.append(daily_avg_volatility) # Test 2
        diff_from_means.append(diff_from_mean) # Test 3

        # Get intraday calculations and append
        #intraday_diffs = (intraday_closes[1:] - intraday_closes[:-1])/intraday_closes[:-1]
        #intraday_diffs_ups = intraday_diffs[intraday_diffs > 0]
        #intraday_diffs_downs = intraday_diffs[intraday_diffs < 0]
        #intraday_avg_volatility = (np.mean(np.abs(intraday_diffs)), # pos + neg
        #                       np.mean(intraday_diffs_ups), # pos
         #                      np.mean(intraday_diffs_downs)) # neg
        #intraday_avg_volatilities.append(intraday_avg_volatility)

        # Add to closes and intraday_closes
        all_closes.append(closes)
        all_intraday_closes.append(intraday_closes)

        # Sleep if hit max calls
        if not calls:
            time.sleep(60)

    # If no stocks made it, print so and exit
    if len(trend_strengths) == 0:
        print('No stocks passed tests')
        exit()
    else:
        print('Stocks passed: {}'.format(len(trend_strengths)))

    # Turn all to numpy array
    trend_strengths = np.array(trend_strengths)
    daily_avg_volatilities = np.array(daily_avg_volatilities)
    diff_from_means = np.array(diff_from_means)

    # ** FIX ANY VALUES BEFORE DIVMAXING **

    # Set losers for mean reversion as tolerance / 5. Then take absolute value for scoring
    diff_winners_max = np.max(diff_from_means[diff_from_means < diff_mean_tol]) # Really worst value that passed
    diff_loser_indices = list(itertools.compress(range(len(diff_from_means)),diff_from_means >= diff_mean_tol)) # indices
    diff_from_means[diff_loser_indices] = diff_winners_max / 5
    diff_from_means = np.abs(diff_from_means)
    
    # Rescale all test values with respect to each other for scoring
    # Originally used minmax, trying divmax because if values are really close (which they probably are)
    # minmax punishes and increases distances for values very close
    # Note all > 0, so it will put them in same range
    trend_strengths = divmax(trend_strengths)
    daily_avg_volatilities = divmax(daily_avg_volatilities)
    diff_from_means = divmax(diff_from_means)

    # ** FIX ANY VALUES BEFORE SCORING **

    # Fix volatility so high scores are bad, low scores are good
    daily_avg_volatilities = 1 + np.min(daily_avg_volatilities) - daily_avg_volatilities

    # Take the weighted average to figure out scores
    # Test 1) As is
    # Test 2) 1 + min(daily_vol) - daily_vol, because lower volatilities are more stable, and want best value to receive 1
    # Test 3) As is
    scores = np.sum(np.array([trend_strengths,
                   daily_avg_volatilities,
                   diff_from_means]).T*w4,axis=1)

    # Argsort on scores
    sorted_order = np.argsort(scores)[::-1]

    # Reshift all values by this argsort and grab the top M
    scores = scores[sorted_order][:M]
    syms = np.array(syms)[sorted_order][:M]
    trend_strengths = trend_strengths[sorted_order][:M]
    daily_avg_volatilities = daily_avg_volatilities[sorted_order][:M]
    diff_from_means = diff_from_means[sorted_order][:M]
    #intraday_avg_volatilities = np.array(intraday_avg_volatilities)[sorted_order][:M] # np
    all_closes = np.array(all_closes)[sorted_order][:M] #np
    all_intraday_closes = np.array(all_intraday_closes)[sorted_order][:M] #np

    # Start plotting these values with the title as values 
    # Find mn and mx values for ylims
    ylow = np.min(all_closes)
    yhigh = np.max(all_closes)
    for i in range(len(syms)):

        # TODO: Calculate suggested tolerances, running simulation with intraday data
        start_time = time.time()
        ptol,ltol = get_tolerances(all_intraday_closes[i])
        stop_time = time.time()
        print('Time {}'.format((stop_time - start_time)/60))

        # Create sub plots, one for daily one for intraday
        fig,axs = plt.subplots(3,figsize=(16,22))
        fig.suptitle(syms[i],fontsize=35)
        axs[0].plot(np.arange(len(all_closes[i])),all_closes[i])
        axs[0].set_title('DAILY -> SCORE: {} | TREND: {} | VOL: {} | MR: {}'.format(np.around(scores[i],decimals=5),
                                                 np.around(trend_strengths[i],decimals=5),
                                                 np.around(daily_avg_volatilities[i],decimals=5),
                                                 np.around(diff_from_means[i],decimals=5)),fontsize=25)
        axs[0].set_ylim([ylow,yhigh])
        axs[1].set_title('DAILY: Zoom in look',fontsize=25)
        axs[1].plot(np.arange(len(all_closes[i])),all_closes[i])
        axs[2].plot(np.arange(len(all_intraday_closes[i])),all_intraday_closes[i])
        #axs[2].set_title('INTRADAY -> VOL: {} UPS: {} DOWNS: {}'.format(np.around(intraday_avg_volatilities[i,0],decimals=5),
        #                                                         np.around(intraday_avg_volatilities[i,1],decimals=5),
        #                                                         np.around(intraday_avg_volatilities[i,2],decimals=5)))
        axs[2].set_title('INTRADAY -> PTOL {} | LTOL: {}'.format(np.around(ptol,decimals=5),np.around(ltol,decimals=5)),fontsize=25)

        # Save the plot for each top stock (and close)
        plt.savefig(dirname+'/'+'Stock_'+str(i+1))
        plt.close()

    # Returns nothing, plots are enough
    time.sleep(60)
