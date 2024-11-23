import numpy as np
import torch
from collections import namedtuple, deque
import random
import math

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

'''
The ReplayMemory class is used to store experiences that the borrower has had in the market.
The memory is an doubly ended queue, wherein the experiences are stored.  We randomly samples
experiences from the memory to train the DQN model. The push method is used to add an experience.
'''
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
'''
The DQN or Deep Q Network, is a neural network that is used to approximate the Q function in the
reinforcement learning algorithm. The DQN is used to predict the Q values for each state-action pair.
The DQN has 3 fully connected layers, with the first two layers having 64 neurons and the last layer
having the number of output neurons equal to the number of actions. The forward method is used to
compute the forward pass of the network.

Q-Value means the expected future rewards that the agent will receive if it takes a particular action
in a particular state. The Q-Value is calculated as the sum of the immediate reward and the discounted
future rewards.

So, in a nutshell the followign DNN, is trying to predict the Q-Value for each state-action pair and become 
the approximator of the Q-Function.
'''
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Borrower:
    def __init__(self, id, best_values=None):
        self.id = id
        if best_values:
            self.credit_score = int(best_values.get('highest_credit_score', np.random.randint(300, 850)))
            self.income = int(best_values.get('highest_income', np.random.randint(20000, 150000)))
            self.debt = int(best_values.get('lowest_debt', np.random.randint(0, 100000)))
        else:
            self.credit_score = np.random.randint(300, 850)
            self.income = np.random.randint(20000, 150000)
            self.debt = np.random.randint(0, 100000)
        
        self.loans = []
        self.risk_tolerance = np.random.uniform(0.1, 0.9)
        self.financial_literacy = np.random.uniform(0.1, 0.9)

        # RL setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 14  # 14 features in the state (Features of the borrower and market)
        self.output_size = 2 # 0: Reject, 1: Accept
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device) 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(5000) # Memory capacity
        self.batch_size = 32 # Number of experiences to sample from memory
        self.gamma = 0.99 #Discount factor
        self.eps_start = 0.9 
        self.eps_end = 0.05
        self.eps_decay = 200
        self.steps_done = 0

        self.employment_stability = np.random.uniform(0.5, 1.0)

    def can_borrow(self):
        # Allow up to 3 loans and a debt-to-income ratio of 0.6
        return len(self.loans) < 3 and self.debt_to_income_ratio() < 0.6

    def debt_to_income_ratio(self):
        return self.debt / self.income if self.income > 0 else 1

    def state_to_tensor(self, state, loan_offer):
        #Map the action space to a tensor
        return torch.tensor([
            self.credit_score / 850,
            self.income / 150000,
            self.debt / 100000,
            len(self.loans) / 3,
            state['avg_credit_score'] / 850,
            state['avg_income'] / 150000,
            state['avg_debt'] / 100000,
            state['num_loans'] / 1000,
            state['default_rate'],
            state['avg_interest_rate'],
            state['market_liquidity'],
            loan_offer[0],
            loan_offer[1] / 100000,
            loan_offer[2] / 60 
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

    def evaluate_loan(self, loan, market_state):
        # Evaluate whether to accept or reject a loan offer
        if loan is None:
            return False
        sample = random.random()
        # Epsilon-greedy policy (Exploration vs Exploitation) 
        # This is used to balance the exploration and exploitation in the model 
        # in simple terms, it is used to decide whether to take a random action or the action with the highest Q-Value
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # Convert the state and loan offer to a tensor
        loan_offer = (loan.interest_rate, loan.amount, loan.term)
        state_tensor = self.state_to_tensor(market_state, loan_offer)

        # Choose the action with the highest Q-Value
        if sample > eps_threshold:
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
                action = action_values.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.output_size)]], device=self.device, dtype=torch.long)

        # Incorporate risk tolerance
        decision = action.item() == 1
        if decision:
            affordability = self.calculate_affordability(loan) # Affordability of the borrower
            risk_factor = np.random.random() * self.risk_tolerance # Risk tolerance of the borrower
            literacy_factor = np.random.random() * self.financial_literacy # Financial literacy of the borrower 
            decision = decision and (affordability > 0.7 or (affordability > 0.5 and risk_factor > 0.5 and literacy_factor > 0.5)) #Could be simplified

        stability_factor = self.employment_stability * 0.2
        decision = decision and (stability_factor > 0.7 or (affordability > 0.5 and risk_factor > 0.5))
        
        return decision

    def calculate_affordability(self, loan):
        # If they can't afford the monthly payment, they can't afford the loan
        monthly_payment = loan.monthly_payment()
        disposable_income = self.income - sum(l.monthly_payment() for l in self.loans) - self.debt / 12
        return disposable_income / monthly_payment if monthly_payment > 0 else 0

    def apply_for_loan(self, loan):
        # Apply for a loan and add it to the list of loans
        if self.can_borrow():
            self.loans.append(loan)
            self.debt += loan.amount
            return True
        return False

    def make_payment(self, loan):
        # Make monthly payment on a loan
        payment = loan.monthly_payment()
        if self.income >= payment:
            self.income -= payment
            loan.amount -= payment
            if loan.amount <= 0:
                self.loans.remove(loan)
                self.improve_credit_score(10)
            return True
        return False

    def improve_credit_score(self, points):
        # Improve the credit score of the borrower
        self.credit_score = max(300, min(850, self.credit_score + points))

    def update_state(self, market_state, action, reward, next_state, loan_offer):
        # Update the state of the borrower and train the DQN
        # Store the experience in the memory and sample a batch to train the model
        self.memory.push(self.state_to_tensor(market_state, loan_offer),
                         torch.tensor([[int(action)]], device=self.device, dtype=torch.long),
                         torch.tensor([reward], device=self.device),
                         self.state_to_tensor(next_state, loan_offer))

        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward) # Immediate reward
        next_state_batch = torch.cat(batch.next_state) # Next state

        # Compute the Q-Value for the current state-action pair
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute the Q-Value for the next state t + 1
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Commbies the immediate reward and the discounted future rewards
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute the loss and backpropagate
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad() # Zero the gradients
        loss.backward() # Backpropagate
        for param in self.policy_net.parameters(): # Clip the gradients
            param.grad.data.clamp_(-1, 1) #When the graidents are too large, they are clipped to prevent exploding gradients
        self.optimizer.step()

    def reset(self):
        self.loans = []
        self.debt = np.random.randint(0, 100000)
        self.credit_score = np.random.randint(300, 850)
        self.income = np.random.randint(20000, 150000)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())