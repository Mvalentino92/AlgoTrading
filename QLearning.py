from IMPORTS import *
from CLASSES import *
from UTILITY_FUNCS import *

# Set layers for BUY NN
buy_input_ = 13
buy_hidden1 = 150
buy_hidden2 = 50
buy_output_ = 2

# Build the buy model
class Buy_Model(nn.Module):
    def __init__(self):
        super(Buy_Model,self).__init__()
        self.input_ = nn.Linear(buy_input_,buy_hidden1)
        self.hidden1 = nn.Linear(buy_hidden1,buy_hidden2)
        self.hidden2 = nn.Linear(buy_hidden2,buy_output_)

    # Standardize the input
    def forward(self,x):
        x = standardize(x)
        x = F.leaky_relu(self.input_(x))
        x = F.leaky_relu(self.hidden1(x))
        x = self.hidden2(x)
        return x


# Create model and set up optimizer
buyer = Buy_Model()
buy_lr = 0.00065
buy_optimizer = torch.optim.Adam(buyer.parameters(),lr=buy_lr)

# The loss function for buyer
buy_loss_fn = nn.MSELoss()

# Now create the seller
sell_input_ = 13
sell_hidden1 = 150
sell_hidden2 = 50
sell_output_ = 2

# Build the sell model
class Sell_Model(nn.Module):
    def __init__(self):
        super(Sell_Model,self).__init__()
        self.input_ = nn.Linear(sell_input_,sell_hidden1)
        self.hidden1 = nn.Linear(sell_hidden1,sell_hidden2)
        self.hidden2 = nn.Linear(sell_hidden2,sell_output_)

    # Standardize the input
    def forward(self,x):
        x = standardize(x)
        x = F.leaky_relu(self.input_(x))
        x = F.leaky_relu(self.hidden1(x))
        x = self.hidden2(x)
        return x


# Create model and set up optimizer
seller = Sell_Model()
sell_lr = 0.00065
sell_optimizer = torch.optim.Adam(seller.parameters(),lr=sell_lr)

# The loss function for seller
sell_loss_fn = nn.MSELoss()

# Set up an epislon greedy stratedgy (start with very high epsilon)
epsilon = 1
epsilon_min = 0.1
epochs = 60
discount = 0.99
delta = (epsilon_min/epsilon)**(1/epochs)
clips = 20

# Set up experience replay
buy_deque_len = 500
buy_batch_size = 50
buy_transitions = deque(maxlen=buy_deque_len)

sell_deque_len = 500
sell_batch_size = 50
sell_transitions = deque(maxlen=sell_deque_len)

# Begin to start episodes (create env first)
env = Environment('GOOGL')
for epoch in range(epochs):

    # Set up
    state = env.reset(day_index=0)
    done = False

    # This is for tracking and plotting
    time = []
    price = []
    buy_time = []
    sell_time = []
    buy_price = []
    sell_price = []

    # While we haven't terminated yet
    while not done:

        # Print the time and epoch
        print('Epoch:',epoch,'\tTime:',env.day.t)

        # Grab if has stock (need original value after we take a step)
        has_stock = env.has_stock

        # With no grad, grab action values (check what network to use)
        with torch.no_grad():
            if has_stock:
                action_values = seller(torch.from_numpy(state).float())
            else:
                action_values = buyer(torch.from_numpy(state).float())

        # Use epsilon greedy strategy
        # TODO: Change it to NOT be able to select highest
        # TODO: Add has_stock to sell (sell has to be 2)
        if np.random.rand() < epsilon:
            action = env.sample(has_stock)
        else:
            action = torch.argmax(action_values) + has_stock

        # Add this time and price
        time.append(env.day.t)
        price.append(env.day.open)

        # Add these actions
        if action == BUY:
            buy_time.append(env.day.t)
            buy_price.append(env.day.open)
        elif action == SELL:
            sell_time.append(env.day.t)
            sell_price.append(env.day.open)

        # Take a step with this action and get observables
        state_prime,reward,done = env.step(action)

        # Add everything to transition
        # TODO: Add done, so we can not zero out the expected reward for terminal state
        # TODO: Possibly minus 1 from sell action
        # TODO: Add has stock
        if has_stock:
            sell_transitions.append((state,action-1,reward,state_prime,has_stock,done))
        else:
            buy_transitions.append((state,action,reward,state_prime,has_stock,done))

        # If we can train, do so
        if len(buy_transitions) > buy_batch_size:
            train_model(buyer,buy_transitions,buy_batch_size,buy_optimizer,
                        buy_loss_fn,seller,discount)
        if len(sell_transitions) > sell_batch_size:
            train_model(seller,sell_transitions,sell_batch_size,sell_optimizer,
                        sell_loss_fn,buyer,discount)

        # Set state as state_prime
        state = state_prime

    # Show a plot if applicable
    if (epoch + 1) % clips == 0:

        # Plot the prices as line graph
        plt.plot(time,price)

        # Plot scatterplots for buying and selling
        plt.scatter(buy_time,buy_price)
        plt.scatter(sell_time,sell_price)

        plt.show()
