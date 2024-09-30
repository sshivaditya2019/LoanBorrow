import numpy as np
import torch
from collections import namedtuple, deque
import random
import math

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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
        self.input_size = 14 
        self.output_size = 2
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(5000)
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.steps_done = 0

    def can_borrow(self):
        return len(self.loans) < 3 and self.debt_to_income_ratio() < 0.6  # Increased from 0.5

    def debt_to_income_ratio(self):
        return self.debt / self.income if self.income > 0 else 1

    def state_to_tensor(self, state, loan_offer):
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
        if loan is None:
            return False
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        loan_offer = (loan.interest_rate, loan.amount, loan.term)
        state_tensor = self.state_to_tensor(market_state, loan_offer)

        if sample > eps_threshold:
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
                action = action_values.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.output_size)]], device=self.device, dtype=torch.long)

        # Incorporate risk tolerance and financial literacy
        decision = action.item() == 1
        if decision:
            affordability = self.calculate_affordability(loan)
            risk_factor = np.random.random() * self.risk_tolerance
            literacy_factor = np.random.random() * self.financial_literacy
            decision = decision and (affordability > 0.7 or (affordability > 0.5 and risk_factor > 0.5 and literacy_factor > 0.5))

        return decision

    def calculate_affordability(self, loan):
        monthly_payment = loan.monthly_payment()
        disposable_income = self.income - sum(l.monthly_payment() for l in self.loans) - self.debt / 12
        return disposable_income / monthly_payment if monthly_payment > 0 else 0

    def apply_for_loan(self, loan):
        if self.can_borrow():
            self.loans.append(loan)
            self.debt += loan.amount
            return True
        return False

    def make_payment(self, loan):
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
        self.credit_score = max(300, min(850, self.credit_score + points))

    def update_state(self, market_state, action, reward, next_state, loan_offer):
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
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def reset(self):
        self.loans = []
        self.debt = np.random.randint(0, 100000)
        self.credit_score = np.random.randint(300, 850)
        self.income = np.random.randint(20000, 150000)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())