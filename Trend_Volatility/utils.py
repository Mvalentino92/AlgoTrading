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

