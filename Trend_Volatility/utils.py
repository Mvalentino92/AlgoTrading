from IMPORTS import *
from ENV_VAR import *

# *** Any function that calls the API should wait 1 minute after finished ***

# Gets current time hh:mm
def get_time(asint=False):
    if asint:
        dtime = datetime.datetime.now()
        return dtime.hour*60 + dtime.minute
    return time.strftime('%H:%M')

# Sleep for either specified time, or 60 seconds because of API calls
def sleeptime(calls,t=1):
    return 60 if not calls else t

# Append a message to the log file and save it
def update_log(message,filename):
    file = open(filename,"a")
    file.write(message+'\n')
    file.close()

# Returns n weights for mean gradients following some function
# Provide the power of the polynomial you want to fit for
def get_weights(f,n,k=0):
    x = np.linspace(k,1,n)
    y = f(x)
    y /= np.sum(y)
    return y

# Divides by max value 
# Used for normalization when there are few entries. Don't want (4,3.999) -> (1,0). Lose closeness
def divmax(x):
    return x / np.max(x)

# Simple MinMax Scaler for values between 0 and 1
# Defaults to divide by max if min and max too close
def minmax(x):
    mn = np.min(x)
    mx = np.max(x)
    if mx - mn < 1e-9:
        return divmax(x)
    return (x - mn)/(mx-mn)

# Mean absolute percent error
def mape(y,y_hat):
    return np.mean(np.abs((y_hat - y)/y))

# Curve fitting
def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

# Run simulations to get tolerances for selling
def score_tolerances(x,*args):

    # Unpack args
    data = args[0]
    trials = args[1]

    # Set number profit, loss, and track how many trials ended sucessfully
    profit = 0
    loss = 0
    trials_sold = 0

    # Useful vars
    data_len = len(data)
    drop_bound = data_len - 390 - 100 # minus 1 day for day trade, minus 100 for time to actually sell next day

    # Begin running simulations using these tolerances
    for i in range(trials):

        # Get random starting point
        j = random.randint(0,drop_bound)
        start = j
        sold = False

        # Get price at this point
        price = data[j]

        # Begin to iterate forward, and sell if hit tolerances
        j += 390
        while j < data_len:

            # Get plpc and check if had to sell, tally accordingly
            plpc = (data[j] - price)/price
            if plpc >= x[0]:
                profit += 1
                trials_sold += 1
                sold = True
                break
            elif plpc <= x[1]:
                loss += 1
                trials_sold += 1
                sold = True
                break

            # Incremet j if didnt break
            j += 1

    #print(sold,' ',j - start + 390)
    # TODO: How much profit or loss needs to be reflected here,
    # otherwise, small profit and big loss will always be the best
    # TODO: Reflect frequency of trials ending in selling. by dividing by just trials
    # NOTE: Given current, if 0.01 tol triggers all, 0.1 needs to trigger 1/10 to be equal. I like it.
    return (loss*np.abs(x[1])-profit*np.abs(x[0]))/trials if trials_sold > 0 else np.Inf

# Get best tolerances using brute force grid search
def get_tolerances(data,trials=1000):

    # Create the ranges and number of ways to chop axis
    rranges = ((0.01,0.1),(-0.1,-0.01))
    Ns = 20

    # Create the args
    args = (data,trials)

    # Pass to brute
    ret = opt.brute(score_tolerances,rranges,args=args,Ns=Ns,finish=None)

    # Return the best tolerances
    return ret
