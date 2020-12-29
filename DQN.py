"""
    Minh T Nguyen
    Dec 28, 2020
    Project: Cart & Pole Problem.

    *** NOTE ***
    DQN:
        Experience Replay and Replay memory:
            e(t) = (s(t), a(t), r(t), s(t+1)) # Experience

    Step:
        1/ Initialize replay memory capacity
        2/ Initialize the network with random weights
        3/ For each episode:
            1/ Initialize the starting state
            2/ For each time step:
                1/ Select an action: Exploration or exploitation
                2/ Execute selected action in the emulator
                3/ Observe reward and the next state
                4/ Store experience in replay memory **** Finish Q-Learning
                5/ Sample random batch from replay memory
                6/ Preprocess states from batch
                7/ Pass batch of preprocessed state to policy network
                8/ Calculate loss between output Q-values and target Q-Values
                    * Requires a second pass to the nework for the next state
                9/ Gradient descent updates weights in the policy network to minimize loss
                    * After x timestep, weights in the target network are updated to the weights in the policy network

    The Policy network: Optimize the policy by finding the optimal q functions

    The Target Network: Instead of using the same network to calculate both prediction and target, we use a separate network
        => Enhance stability!
"""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

# Check environment wise, no training yet ########################
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
# env.close()

# Build network: #################################################
"""
    DEEP Q NETWORK
    Class DQN enheritance from the network Module class allow for keeping track of weights
        3 Linear Layer with relu activation function
    @param: img_height = number of pixel of image's height
            img_width = number of pixel of image's width
"""
class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        self.fc1 = nn.Linear(in_features=img_height * img_width * 3,
                             out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t

# EXPERIENCE
# Experience tuple object named Experience storing tuple of state, action, next_state, rewards
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

"""
    REPLAY MEMORY
    ReplayMemory class: store the experience in memory arrays
    @param: capacity = capacity of the ReplayMemory object
"""
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []  #
        self.push_count = 0  # Keep track how many experience store in memory

    # push store experience into memory array
    def push(self, experience):

        # if the memory array less than the capacity of the ReplayMemory object
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # If the memory is full, then push new experience to the front of memory overriding the older experience
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    # Sample of random experience used in training DQN with the size of batch size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Can we provide the sample experience or not
    # Ex: training memory = 20, batch_size = 50 => Can't provide batch size sample
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

"""
    EPSILON GREEDY STRATEGY
    Class EpsilonGreedyStrategy: Determine whether to explore or exploit or not
    @param: start = start of the exploration 
            end = end of the exploration
            decay = decaying rate of the exploration_rate which is set initially to 1
"""
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    # math equation in research paper
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

"""
    RL AGENT
    Class Agent: 
    @param: strategy = exploration or exploitation
            num_actions = number of action in an episode
            device = cpu or gpu
    Note: since we want the result of the forward pass, no need to track the gradient
"""
class Agent():
    def __init__(self, strategy, numn_actions, device):
        self.current_step = 0  # Current step number in the environment
        self.strategy = strategy
        self.num_actions = numn_actions  # how many possible actions can the agent take from a given state
        self.device = device

    # function determine the exploration rate => research paper
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)  # exploration rate
        self.current_step += 1

        if rate > random.random():  # Check if larger than number (0,1)
            action = random.randrange(self.num_actions)  # explore
            return torch.tensor([action]).to(
                self.device)  # turn action variable into a tensor, then calculate it in cpu or gpu
        else:
            with torch.no_grad():  # Don't track gradient forward calculation since we use model for inference, not training
                return policy_net(state).argmax(dim=1).to(
                    self.device)  # exploit

"""
    ENVIRONMENT MANAGER
    Class CartPole Environment Manager
        Function set up environment for GYM CartPole
    @param: device: cpu or gpu
"""
class CartPoleEnvManager():
    def __init__(self, device):  # Function create environment class
        self.device = device

        # unwrapped give us access to behind-the-scene dynamics of the environment
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()

        # Current screen track the current screen at any given time, None = we at the start of the episode and have not yet render the screen of intial observation
        self.current_screen = None
        self.done = False

    def reset(self):  # reset at the end of the episode so the current screen can be none
        self.env.reset() # reset the environment, function from GYM lib
        self.current_screen = None # None = we are at the start of the episode and have not yet rendered the screen of the initial observation

    def close(self): # Function call to close the env when we finish it.
        self.env.close()

    def render(self, mode='human'): # Function is called to render the current state to the screen
        return self.env.render(mode) # return numpy array version of the rendered screen

    # NUMBER OF ACTION AVAILABLE TO AGENT
    def num_actions_available(self):  # agent can have 2 action left or right
        return self.env.action_space.n # return number of actions availabel to the agent in the environemnt

    # TAKE ACTION IN THE ENVIRONMENT
    # Note OUTPUT: that we have item since we want the return of tensor since we want the input and output to be consistent as tensor
    def take_action(self, action):  # execute an action.
        #  Function return tuple (env observation, reward, episode end or not, dianogstic info)
        _, reward, self.done, _ = self.env.step(
            action.item())  # we only care about the reward and the episode ended or not
        return torch.tensor([reward], device=self.device)

    # START AN EPISODE
    # Note: the current screen is None in the coinstructor and none in the environment is reset
    def just_starting(self):  # return true if current screen is none
        return self.current_screen is None

    # GET THE STATE OF THE ENVIRONMENT:
    # A single state in env = different between current screen and previous screen => Take velocigy of the pole into account from 1 single images
    def get_state(
            self):  # return the current state of the environment in the form of the process of the screen
        if self.just_starting() or self.done: # Check if we start or done with the episode
            # Note: if we just start, the initial screen has not been rendered from the initial observation in the env
            #       if done, the last action taken by the agent ended the episode
            self.current_screen = self.get_processed_screen() # if just start, so previous screen is black; if end, then next screen also black
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:  # in the middle of the episode
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    # PROCESS SCREEN IMAGE
    def get_screen_height(self):  # get height of screen
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):  # get width of screen
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0,
                                                     1))  # render environment as rbg array then render (channel, height, width)
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    # CROP SCREEN IMAGE
    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # strip off top and bottom
        top = int(screen_height * 0.4)  # 40% of the screen height
        bottom = int(screen_height * 0.8)  # 80% of the screen height
        screen = screen[:, top:bottom, :]  # strip top 40% and bottom 20%
        return screen

    # CONVERT AND RESCALE SCREEN IMAGE DATA
    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen,
                                      dtype=np.float32) / 255  # All value of the array store sequentially in memory
        screen = torch.from_numpy(
            screen)  # convert numpy array to pytorch tensor

        # Use torchvision to compose image transforms
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40, 90)),
            T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device)


def extract_tensors(experiences):  # Take in batch of experience
    batch = Experience(
        *zip(*experiences))  # tranfer it to experience of batches

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)


# Q-Value Calculator: for the current state using the policy net, the next state using the target net
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod  # call function without create an instance of class
    def get_current(policy_net, states,
                    actions):  # call in main: the state action pairs that sampled from memory
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    # Find out if any final state in our next state tensor (since the agent is unable to recieve any reward once an episode is ended), if we do, need to find where they are so we don't pass them to the target net
    def get_next(target_net,
                 next_states):  # return the max Qvalue predicted by the target net among all possible next actions
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[
            0].eq(0).type(
            torch.bool)  # look at next state tensor and find location of all final states
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = \
        target_net(non_final_states).max(dim=1)[0].detach()
        return values


def plot(values,
         moving_avg_period):  # plot the duration of each episode and 100 episode moving averaga
    plt.figure(2)
    plt.clf()
    plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)
    print("Episode ", len(values), "\n", moving_avg_period,
          "episode moving avg: ", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)

# Transform the value to tensor, then check if len of value >= period, we want to have enought episodes to train
def get_moving_average(period, values):  # Plot 100 episode moving average
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(
            dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


# Testing plot() Function
# plot(np.random.rand(300), 100)

def main():
    print("Running")
    batch_size = 256
    gamma = 0.999 # discount factor used the Bellman equation

    eps_start = 1  # epsilon: exploration rate
    eps_end = 0.01
    eps_decay = 0.001

    target_update = 10  # update the target network every 10 episode
    memory_size = 100000
    lr = 0.001
    num_episodes = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = CartPoleEnvManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    # Initializer policy network with random state
    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    # Training Process:
    episode_durations = []  # Empty list to store events
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()  # get initial state

        for timestep in count():
            # Set action: explore or exploit, receive reward, then get to the next state
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            next_state = em.get_state()

            # Create an experience and push onto replay memeory
            memory.push(Experience(state, action, next_state,
                                   reward))  # create experience and push to memory
            state = next_state

            # Agent got experience, check if can get a sample from replay memory to train the policy net
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)

                states, actions, rewards, next_states = extract_tensors(
                    experiences)

                # get current = return the q value for any given state-action pair as prediced from the policy network
                current_q_values = QValues.get_current(policy_net, states,
                                                       actions)

                # get the qvalues for the next states in the batch
                next_q_values = QValues.get_next(target_net, next_states)

                # Calculate the target q value => Research paper equation
                target_q_values = (next_q_values * gamma) + rewards # gamma = discount rate

                loss = F.mse_loss(current_q_values,
                                  target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if em.done:  # Break out to start a new episode
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break

        if episode % target_update == 0:  # Check if we should do update in the target net
            target_net.load_state_dict(policy_net.state_dict())

    em.close()


if __name__ == "__main__":
    main()

##############################################################################
# # Example of non-processed screen
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# em = CartPoleEnvManager(device)
# em.reset()
# screen = em.render("rgb_array")
#
# plt.figure()
# plt.imshow(screen)
# plt.title("Non-processed screen example")
# plt.show()
#
# # Example of processed screen
# screen = em.get_processed_screen()
# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
# plt.title("Processed screen example")
# plt.show()
#
# # Example of starting state - black screen
# screen = em.get_state()
# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
# plt.title('Processed started screen example')
# plt.show()
#
# # Example State that not in the starting state - take action then get state: tell us where screen previously and now
# for i in range(6):
#     em.take_action(torch.tensor([1]))
# screen = em.get_state()
#
# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
# plt.title("Processed non-start screen example")
# plt.show()
#
# # Example of end state: same as start state
# em.done = True
# screen = em.get_state()
#
# plt.figure()
# plt.imshow(screen.squeeze(0).permute(1,2,0), interpolation='none')
# plt.title("Processed image end state")
# plt.show()
# em.close()