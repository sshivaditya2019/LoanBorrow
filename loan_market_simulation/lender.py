import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math

# The experience tuple is used to store the experience of the lender in the market.
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

# The ReplayMemory class is used to store experiences that the lender has had in the market.(Experience Replay)
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# A Deep Q Approximator is used to approximate the Q function in the reinforcement learning algorithm.
# The DQN is used to predict the Q values for each state-action pair.
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Enhanced network architecture with layer normalization
        self.fc1 = nn.Linear(input_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, output_size)
        
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class Lender:
    def __init__(self, id, initial_capital=1000000, risk_tolerance=0.5, best_values=None, use_credit_history=False):
        self.id = id
        self.initial_capital = initial_capital if best_values is None else best_values.get('highest_lender_capital', initial_capital)
        self.capital = self.initial_capital
        self.risk_tolerance = risk_tolerance if best_values is None else np.random.uniform(0.1, 0.9)
        self.loans = []
        self.min_loan_amount = 10000
        self.max_loan_amount = 100000
        self.min_interest_rate = 0.03
        self.max_interest_rate = 0.12
        # Track performance metrics
        self.default_history = []
        self.profit_history = []
        self.use_credit_history = use_credit_history

        if best_values and 'optimal_interest_rate' in best_values:
            self.optimal_interest_rate = best_values['optimal_interest_rate']
        else:
            self.optimal_interest_rate = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reduced state space to most relevant features
        self.input_size = 7 if not self.use_credit_history else 8 # Simplified state representation
        self.output_size = 100 # Action space with 100 possible actions
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device) # DQN model for the lender
        self.target_net = DQN(self.input_size, self.output_size).to(self.device) # Target DQN model for the lender
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimized hyperparameters for better learning
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory(50000)
        self.batch_size = 128
        self.gamma = 0.99  # Discount factor for future rewards
        self.eps_start = 1.0  # Starting value of epsilon for epsilon-greedy policy
        self.eps_end = 0.01  # Minimum value of epsilon
        self.eps_decay = 2000  # Decay rate for epsilon
        self.steps_done = 0
        
        # Track metrics for adaptive learning
        self.avg_reward = 0
        self.reward_history = deque(maxlen=100)

    def get_default_rate(self):
        # Calculate rolling default rate from history
        if not self.default_history:
            return 0
        return sum(self.default_history[-100:]) / len(self.default_history[-100:])

    def get_profit_rate(self):
        # Calculate rolling profit rate from history
        if not self.profit_history:
            return 0
        return sum(self.profit_history[-100:]) / len(self.profit_history[-100:])

    def state_to_tensor(self, state):
        # Simplified and normalized state representation for better learning
        # Each feature is carefully selected and normalized to [0,1] range
        if self.use_credit_history:
            return torch.tensor([
                self.capital / self.initial_capital,  # Normalized capital position
                state['default_rate'],               # Market default rate
                state['market_liquidity'],           # Market liquidity
                self.get_default_rate(),             # Lender's historical default rate
                self.get_profit_rate(),              # Lender's historical profit rate
                len(self.loans) / 20,                # Normalized loan portfolio size
                state['economic_cycle'],             # Economic cycle indicator
                state['avg_credit_history']          # Average credit history
            ], dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            return torch.tensor([
                self.capital / self.initial_capital,  # Normalized capital position
                state['default_rate'],               # Market default rate
                state['market_liquidity'],           # Market liquidity
                self.get_default_rate(),             # Lender's historical default rate
                self.get_profit_rate(),              # Lender's historical profit rate
                len(self.loans) / 20,                # Normalized loan portfolio size
                state['economic_cycle']              # Economic cycle indicator
            ], dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_action(self, state):
        sample = random.random()
        # Epsilon greedy policy (Exploration vs Exploitation) 
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
        
        # Enhanced action mapping with economic considerations
        base_rate = (action_item % 10) / 9 * (self.max_interest_rate - self.min_interest_rate) + self.min_interest_rate
        market_adjustment = (state['default_rate'] - 0.05) * 0.5
        liquidity_adjustment = (1 - state['market_liquidity']) * 0.05
        economic_adjustment = state['economic_cycle'] * 0.01
        
        random_factor = np.random.normal(0, 0.005)  # Reduced noise for more stable learning
        
        if self.optimal_interest_rate is not None:
            interest_rate = max(self.min_interest_rate, min(self.max_interest_rate, 
                                self.optimal_interest_rate + market_adjustment + liquidity_adjustment + economic_adjustment + random_factor))
        else:
            interest_rate = max(self.min_interest_rate, min(self.max_interest_rate, 
                                base_rate + market_adjustment + liquidity_adjustment + economic_adjustment + random_factor))

        loan_amount = ((action_item // 10) % 5) / 4 * (self.max_loan_amount - self.min_loan_amount) + self.min_loan_amount
        term = ((action_item // 50) % 3) * 12 + 12

        #Using stock indices to influence loan interest rates
        economic_amplifier = 1 + 0.5 * state['economic_cycle']
        stock_influence = (state['stock_index'] - 1000) * 0.0001 * economic_amplifier
        interest_rate += stock_influence
        
        return interest_rate, loan_amount, term

    def assess_loan(self, loan, borrower):
        # Enhanced risk assessment with historical performance
        credit_score_factor = (borrower.credit_score - 300) / 550
        dti_factor = 1 - borrower.debt_to_income_ratio()
        loan_amount_factor = 1 - (loan.amount / self.capital)
        default_history_factor = 1 - self.get_default_rate()
        
        if self.use_credit_history:
            credit_history = borrower.credit_history
            
            # Buckets of credit history
            if 6 <= credit_history <= 180:
                credit_history_factor = 1/3
            if 181 <= credit_history <= 720:
                credit_history_factor = 5/6
            elif credit_history > 720:
                credit_history_factor = 1
            
            
            loan_score = (
                credit_score_factor * 0.2 + 
                credit_history_factor * 0.1 + 
                dti_factor * 0.3 + 
                loan_amount_factor * 0.2 + 
                default_history_factor * 0.2
            ) * (self.risk_tolerance + 0.2)

        else:    
            loan_score = (
                credit_score_factor * 0.3 + 
                dti_factor * 0.3 + 
                loan_amount_factor * 0.2 + 
                default_history_factor * 0.2
            ) * (self.risk_tolerance + 0.2)
        
        return np.random.random() < loan_score

    def grant_loan(self, loan):
        if self.capital >= loan.amount:
            self.capital -= loan.amount
            self.loans.append(loan)
            return True
        return False

    def recover_loan(self, loan):
        try:
            self.loans.remove(loan)
        except ValueError:
            pass
        recovery = loan.balance * 0.5
        self.capital += recovery
        # Update performance metrics
        self.default_history.append(1)
        self.profit_history.append(-loan.balance * 0.5)

    def update_state(self, state, action, reward, next_state):
        # Update performance metrics
        if reward > 0:
            self.profit_history.append(reward)
            self.default_history.append(0)
        
        # Update reward history for adaptive learning
        self.reward_history.append(reward)
        self.avg_reward = sum(self.reward_history) / len(self.reward_history)

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

        # Q-Learning algorithm 
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Huber loss for more stable learning
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
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
        # Reset performance metrics
        self.default_history = []
        self.profit_history = []
        self.reward_history.clear()
        self.avg_reward = 0
        self.use_credit_history = self.use_credit_history

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
