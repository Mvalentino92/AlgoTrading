from IMPORTS import *
from CLASSES import *
from UTILITY_FUNCS import *

# Just employ and graph a random strategy before we actually try RL
# Make sure the environment works
env = Environment('GOOGL')
time = []
price = []
buy_time = []
sell_time = []
buy_price = []
sell_price = []
env.reset()
done = False

while not done:
    is_selling = env.has_stock
    action = env.sample(is_selling)
    time.append(env.day.t)
    price.append(env.day.open)
    if action == BUY:
        buy_time.append(env.day.t)
        buy_price.append(env.day.open)
    elif action == SELL:
        sell_time.append(env.day.t)
        sell_price.append(env.day.open)
    _,_,done = env.step(action)

# Plot the price at every time
plt.plot(time,price)

# Plot scatters for buy and sell
plt.scatter(buy_time,buy_price)
plt.scatter(sell_time,sell_price)

print(time[0])
print(time[-1])
plt.show()
