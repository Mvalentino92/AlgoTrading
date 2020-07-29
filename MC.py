# Make all necessary imports (importing Historical handles all np and pd ect)
from HistoricalTesting import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

total_pl = []
eps = 1
diff = 0.025
while eps >= diff:
    # Create the environment first, so we have info to build NN
    env = Environment(APCA_API_KEY_ID,APCA_API_SECRET_KEY,APCA_API_BASE_URL,'GOOGL',
                      filename='intraday.pkl',equity=20000)

    # Create layer lengths for NN
    input_ = 28
    hidden1 = 150
    hidden2 = 50
    output_ = env.num_actions

    # Build the model
    class Model(nn.Module):
        def __init__(self):
            super(Model,self).__init__()
            self.input_ = nn.Linear(input_,hidden1)
            self.hidden1 = nn.Linear(hidden1,hidden2)
            self.hidden2 = nn.Linear(hidden2,output_)

        def forward(self,x):
            x = F.leaky_relu(self.input_(x))
            x = F.leaky_relu(self.hidden1(x))
            x = self.hidden2(x)
            return x

    # Build model and set up optimizer
    model = Model()
    if eps < 1:
        model.load_state_dict(torch.load('themodel.pt'))
        model.eval()
    lr = 0.0009
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # Set the loss function
    loss_fn = nn.MSELoss()

    # Set up variables for epsilon greedy strategy
    epsilon = eps
    epsilon_min = epsilon - diff + diff/100
    intervals = env.data_len - env.idx
    delta = (epsilon_min/epsilon)**(1/intervals)

    # Set variables for experience replay
    batch_size = 32
    deque_len = 370 # The entire day
    transitions = deque(maxlen=deque_len)

    # Set variable for stopping if data is used up
    end_of_data = False

    # Track the number of updates
    times_trained = 0

    # Daily profit loss
    daily_pl = []

    # While we still have data left (contiue to run episodes)
    while not end_of_data:

        # Set up the episode
        state = env.reset()
        sold = False
        episode_transitions = []

        # Grab end of data fresh for safety
        end_of_data = env.idx == env.data_len

        # While we have not sold or run out of data (marking end of episode)
        # NEED TO ADD AND TRACK END OF DAY HERE
        # If end of day (set as 15 minutes before market closes)
        # We still let the ELSE take care of it for not selling
        while not sold and not end_of_data and env.current_time < 945:

            # Grab the action values for this state (no gradient)
            with torch.no_grad():
                action_values = model(torch.from_numpy(state).float())

            # Enforce epsilon greedy strategy
            if np.random.rand() < epsilon:
                action = env.sample(action_values)
            else:
                # Grab the available actions we have, then find max
                bools = env.available_actions_bools()
                action = env.action_space[bools][action_values[bools].argmax()]

            # Take a step with this action
            # if action < env.num_buying_actions:
            #     print('BUY AT:',env.bp[action])
            # elif action == env.num_buying_actions:
            #     print('SELL')
            # else:
            #     print('HOLD')
            state_prime,reward,sold,end_of_data = env.step(action)

            # Update episode_transitions
            episode_transitions.append((state,action,reward))

            # If we have enough transitions, update
            if len(transitions) >= batch_size:
                times_trained += 1
                train_model(transitions,batch_size,model,optimizer,
                            loss_fn,env.num_actions)

            # Decrease epsilon
            epsilon *= delta

            # Printing for debuggin
            #print(env.equity)

        # The episode has ended
        # If it has ended because of a sell, calculate returns and add to transitions
        if sold:

            # Set the return to the reward of the sell, keep all other data
            pl = episode_transitions[-1][-1]
            episode_transitions = [(s,a,pl) for (s,a,_) in episode_transitions]

            # Train specifically for this sell now
            train_model(episode_transitions, len(episode_transitions), model, optimizer,
                        loss_fn, env.num_actions)

            # Then append to transitions
            transitions.extend([(s,a,pl) for (s,a,_) in episode_transitions])


        # Otherwise, it ended cause of the last data so see if we can sell
        # And then run on update just on this episode (potentially with special penalty for
        # not selling)
        else:

            # Check if we can sell
            if env.buying_price.size > 0 and env.shares_owned.size > 0:

                # Force sell (observe reward)
                reward = env.sell_stocks()

                # Redo episode transition (set return to what sold for)
                episode_transitions = [(s,a,reward) for (s,a,_) in episode_transitions]

                # Train on this entire batch
                times_trained += 1
                train_model(episode_transitions,len(episode_transitions),model,optimizer,
                            loss_fn,env.num_actions)

            # If can't sell, just do nothing.

        # If ended cause of current time or end_of_data then store the daily profit loss
        # And reset equity
        if env.current_time >= 945 or end_of_data:
            daily_pl.append(env.equity - 20000)
            env.equity = 20000

    # After everything is done, print total equity
    total_pl.append(np.sum(daily_pl))

    # Save the model
    torch.save(model.state_dict(),'themodel.pt')

    # Reduce eps
    eps -= diff

x = np.arange(len(total_pl))
y = np.array(total_pl)
plt.scatter(x,y)
plt.show()


