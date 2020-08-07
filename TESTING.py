from IMPORTS import *
from CLASSES import *
from UTILITY_FUNCS import *

# Set layers for BUY NN
buy_input_ = 6
buy_hidden1 = 256
buy_hidden2 = 128
buy_output_ = 2

# Build the buy model
class Buy_Model(nn.Module):
    def __init__(self):
        super(Buy_Model,self).__init__()
        self.input_ = nn.Linear(buy_input_,buy_hidden1)
        self.hidden1 = nn.Linear(buy_hidden1,buy_hidden2)
        self.hidden2 = nn.Linear(buy_hidden2,buy_output_)

    # Standardize the input (not the normalized time)
    def forward(self,x):
        x = F.leaky_relu(self.input_(x))
        x = F.leaky_relu(self.hidden1(x))
        x = self.hidden2(x)
        return x


# Create model and set up optimizer
buyer = Buy_Model()
buyer.load_state_dict(torch.load('buyer.pt'))

# Now create the seller
sell_input_ = 6
sell_hidden1 = 256
sell_hidden2 = 128
sell_output_ = 2

# Build the sell model
class Sell_Model(nn.Module):
    def __init__(self):
        super(Sell_Model,self).__init__()
        self.input_ = nn.Linear(sell_input_,sell_hidden1)
        self.hidden1 = nn.Linear(sell_hidden1,sell_hidden2)
        self.hidden2 = nn.Linear(sell_hidden2,sell_output_)

    # Standardize the input (not the normalized time)
    def forward(self,x):
        x = F.leaky_relu(self.input_(x))
        x = F.leaky_relu(self.hidden1(x))
        x = self.hidden2(x)
        return x


# Create model and set up optimizer
seller = Sell_Model()
seller.load_state_dict(torch.load('seller.pt'))

# Just employ and graph a random strategy before we actually try RL
# Make sure the environment works
env = Environment('GOOGL')

for i in range(len(env.intradays)):
    state = env.reset(day_index=i)
    done = False
    profits = []
    time = []
    price = []
    buy_time = []
    sell_time = []
    buy_price = []
    sell_price = []
    while not done:
        time.append(env.day.t)
        price.append(env.day.open)

        has_stock = env.has_stock
        with torch.no_grad():
            if has_stock:
                action_values = seller(torch.from_numpy(state).float())
                action = torch.argmax(action_values) + has_stock
            else:
                action_values = buyer(torch.from_numpy(state).float())
                action = torch.argmax(action_values) + has_stock

        if action == BUY:
            buy_time.append(env.day.t)
            buy_price.append(env.day.open)
        elif action == SELL:
            sell_time.append(env.day.t)
            sell_price.append(env.day.open)
        state,_,done = env.step(action)

    held = False
    if env.has_stock:
        held = True
        env.equity += env.shares*env.day.open
    profit = env.equity - 20000

    # Plot the price at every time
    plt.plot(time,price)

    # Plot scatters for buy and sell
    plt.scatter(buy_time,buy_price)
    plt.scatter(sell_time,sell_price)
    plt.legend(('Price','Buy','Sell'))
    plt.title('Profit: {}, Held Stock: {}'.format(profit,held))
    plt.show()
