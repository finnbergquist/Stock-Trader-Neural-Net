import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''
We will have 5 inputs: Holdings, Price of Stock, RSI, MACD signal, and BBP
We will have 3 ouputs: Desired positions of FLAT, LONG, or SHORT
How many nodes in our hidden layer(s)? Lets start with 8 or 9 nodes in 1 layer

Our loss function will be 
'''



class DQN(nn.Module):
    def __init__(self, state_size=4, action_size=3):
        super(DQN, self).__init__()
        self.main = nn.Sequential(
        nn.Linear(state_size, 9),
        nn.LeakyReLU(),
        nn.Linear(9, action_size),
        nn.Sigmoid(),
        )
        

    def forward(self, input):
        return self.main(input)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size=4, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, is_eval=False):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # sit, buy, sell
        self.memory = ReplayMemory(10000)
        self.inventory = []
        self.is_eval = is_eval
        self.batch_size = 15
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.losses = []

        self.tNet = DQN(state_size, self.action_size)
        self.pNet = DQN(state_size, self.action_size)
        self.optimizer = optim.SGD(self.pNet.parameters(), lr=0.005, momentum=0.9)


    def act(self, state):
        #if not self.is_eval and np.random.rand() <= self.epsilon:
        a = random.randrange(self.action_size)
        #print(a)
            #print('random action')
            #print(self.net(torch.FloatTensor(state).to(device)))
        return a
        
        # tensor = torch.FloatTensor(state).to(device)
        # options = self.tNet(tensor)
        # print(options)
        # a = int(torch.argmax(options))
        # print(int(a))
        # return a

    def testAct(self, state):
        
        tensor = torch.FloatTensor(state).to(device)
        options = self.tNet(tensor)
        print(options)
        a = int(torch.argmax(options))
        print(a)
        return a

    def train(self, state, nextState, reward):
        averageLoss = 0
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to net
    #     policy = self.pNet(torch.FloatTensor(state).to(device))
    #     qOld = torch.max(policy)
    #     #print('Old:')
    #    # print(qOld)
    #     target = self.tNet(torch.FloatTensor(nextState).to(device))
    #     qNew = (torch.max(target) * self.gamma) + reward
    #     #print('New:')
    #     #print(qNew)
    #     #loss = (reward + self.gamma(qNew) - qOld)**2
    #     mse = F.mse_loss
    #     loss = mse(qOld, qNew)
    #     #print(loss)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     for param in self.pNet.parameters():
    #         param.grad.data.clamp_(-1, 1)
    #     self.optimizer.step()
        #Optimize the model
        
        if len(self.memory) > 100:
            for i in range(10):

                T = self.memory.sample(1)[0]
                qOld = torch.max(self.pNet(torch.FloatTensor(T.state).to(device)))
                qNew = (torch.max(self.tNet(torch.FloatTensor(T.next_state).to(device))) * self.gamma) + T.reward
                loss = F.mse_loss(qOld, qNew)
                #print(int(loss))
                averageLoss += int(loss)

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.pNet.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
            
        return averageLoss
        #self.epsilon = self.epsilon * self.epsilon_decay
        