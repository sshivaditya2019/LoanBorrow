import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math

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
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class Lender:
    def __init__(self, id, initial_capital=1000000, risk_tolerance=0.5, best_values=None):
        self.id = id
        self.initial_capital = initial_capital if best_values is None else best_values.get('highest_lender_capital', initial_capital)
        self.capital = self.initial_capital
        self.risk_tolerance = risk_tolerance if best_values is None else np.random.uniform(0.1, 0.9)
        self.loans = []
        self.min_loan_amount = 1000
        self.max_loan_amount = 100000
        self.min_interest_rate = 0.03
        self.max_interest_rate = 0.09

        if best_values and 'optimal_interest_rate' in best_values:
            self.optimal_interest_rate = best_values['optimal_interest_rate']
        else:
            self.optimal_interest_rate = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 10
        self.output_size = 20
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.steps_done = 0

    def state_to_tensor(self, state):
        return torch.tensor([
            state['avg_credit_score'] / 850,
            state['avg_income'] / 150000,
            state['avg_debt'] / 100000,
            state['num_loans'] / 1000,
            state['default_rate'],
            state['avg_interest_rate'],
            state['market_liquidity'],
            self.capital / self.initial_capital,
            self.risk_tolerance,
            len(self.loans) / 100
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                action_values = self.policy_net(state_tensor)
                action = action_values.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.output_size)]], device=self.device, dtype=torch.long)

        action_item = action.item()
        
        base_rate = (action_item % 10) / 9 * (self.max_interest_rate - self.min_interest_rate) + self.min_interest_rate
        market_adjustment = (state['default_rate'] - 0.05) * 0.5
        liquidity_adjustment = (1 - state['market_liquidity']) * 0.05
        
        random_factor = np.random.normal(0, 0.01)
        
        if self.optimal_interest_rate is not None:
            interest_rate = max(self.min_interest_rate, min(self.max_interest_rate, 
                                self.optimal_interest_rate + market_adjustment + liquidity_adjustment + random_factor))
        else:
            interest_rate = max(self.min_interest_rate, min(self.max_interest_rate, 
                                base_rate + market_adjustment + liquidity_adjustment + random_factor))

        loan_amount = ((action_item // 10) % 5) / 4 * (self.max_loan_amount - self.min_loan_amount) + self.min_loan_amount
        term = ((action_item // 50) % 3) * 12 + 12

        #Using stock indices to influence loan interest rates
        stock_influence = (state['stock_index'] - 1000) * 0.0001
        interest_rate += stock_influence
        
        return interest_rate, loan_amount, term

    def assess_loan(self, loan, borrower):
        credit_score_factor = (borrower.credit_score - 300) / 550
        dti_factor = 1 - borrower.debt_to_income_ratio()
        loan_amount_factor = 1 - (loan.amount / self.capital)

        loan_score = (credit_score_factor * 0.3 + dti_factor * 0.3 + loan_amount_factor * 0.4) * (self.risk_tolerance + 0.2)

        return np.random.random() < loan_score

    def grant_loan(self, loan):
        if self.capital >= loan.amount:
            self.capital -= loan.amount
            self.loans.append(loan)
            return True
        return False

    def recover_loan(self, loan):
        self.loans.remove(loan)
        recovery = loan.balance * 0.5
        self.capital += recovery

    def update_state(self, state, action, reward, next_state):
        action_index = self.action_to_index(action)
        self.memory.push(self.state_to_tensor(state),
                         torch.tensor([[action_index]], device=self.device, dtype=torch.long),
                         torch.tensor([reward], device=self.device),
                         self.state_to_tensor(next_state))

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

    def action_to_index(self, action):
        interest_rate, loan_amount, term = action
        interest_rate_index = int((interest_rate - self.min_interest_rate) / (self.max_interest_rate - self.min_interest_rate) * 9)
        loan_amount_index = int((loan_amount - self.min_loan_amount) / (self.max_loan_amount - self.min_loan_amount) * 4)
        term_index = (term - 12) // 12
        return interest_rate_index + loan_amount_index * 10 + term_index * 50

    def reset(self):
        self.capital = self.initial_capital
        self.loans = []
        self.risk_tolerance = np.random.uniform(0.1, 0.9)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
