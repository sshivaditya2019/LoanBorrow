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
The DQN has 3 fully connected layers, with layer normalization for better training stability.
Q-Value means the expected future rewards that the agent will receive if it takes a particular action
in a particular state. The Q-Value is calculated as the sum of the immediate reward and the discounted
future rewards.
'''
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Enhanced architecture with layer normalization
        self.fc1 = nn.Linear(input_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, output_size)
        
        # Xavier initialization for better gradient flow (Graident Exploding/Vanishing)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class Borrower:
    def __init__(self, id, best_values=None, use_credit_history=False):
        self.id = id
        if best_values:
            self.credit_score = int(best_values.get('optimal credit score', np.random.randint(300, 850)))
            self.income = int(best_values.get('optimal income', np.random.randint(20000, 150000)))
            self.debt = int(best_values.get('optimal debt', np.random.randint(0, 50000)))
        else:
            self.credit_score = 600
            self.income = 55000
            self.debt = 0
        
        self.use_credit_history = use_credit_history
        if self.use_credit_history:
            self.credit_history = 6
        self.loans = []
        self.risk_tolerance = np.random.uniform(0.1, 0.7)
        self.financial_literacy = np.random.uniform(0.5, 0.9)
        self.annual_income = self.income
        
        # Track loan history for better decision making
        self.loan_history = []  # Track past loan performance
        self.payment_history = []  # Track payment success/failure

        # RL setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 8 if not self.use_credit_history else 9
        self.output_size = 2  # 0: Reject, 1: Accept
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimized learning parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory(10000)
        self.batch_size = 256
        self.gamma = 0.99  # Discount factor for future rewards
        self.eps_start = 1.0  # Starting value of epsilon for epsilon-greedy policy
        self.eps_end = 0.01  # Minimum value of epsilon
        self.eps_decay = 2000  # Decay rate for epsilon
        self.steps_done = 0
        
        # Track performance metrics
        self.avg_reward = 0
        self.reward_history = deque(maxlen=100)

    def get_payment_success_rate(self):
        if not self.payment_history:
            return 1.0
        return sum(self.payment_history[-100:]) / len(self.payment_history[-100:])

    def recover_loan(self, loan):
        # In case of default
        try:
            self.loans.remove(loan)
        except ValueError:
            pass
        self.debt = max(0, self.debt - (loan.balance * 0.5))
        self.loan_history.append(0)  # Record default

    def can_borrow(self):
        print("Checking if the borrower can borrow")
        # Only check debt-to-income ratio
        return self.debt_to_income_ratio() < 0.5  # Balanced threshold

    def debt_to_income_ratio(self):
        # The monthly payment ratio to the income
        return sum(l.monthly_payment() for l in self.loans) / (self.income / 12) if self.income > 0 else 0

    def encode_economic_cycle(self, economic_cycle):
        # Return normalized economic cycle value
        return economic_cycle  # Already in [-1, 0, 1] range

    def not_defaulted(self):
        # A borrower is defaulted if the debt to income ratio is greater than 0.5
        return self.debt_to_income_ratio() < 0.5  # Balanced threshold

    def state_to_tensor(self, state, loan_offer):
        # Simplified and normalized state representation
        if self.use_credit_history:
            return torch.tensor([
                self.credit_score / 850,                    # Normalized credit score
                self.debt_to_income_ratio(),                # Current DTI ratio
                self.get_payment_success_rate(),            # Historical payment performance
                loan_offer[0],                              # Interest rate (already normalized)
                loan_offer[1] / 100000,                     # Normalized loan amount
                loan_offer[2] / 36,                         # Normalized loan term
                state['economic_cycle'],                    # Economic cycle
                state['market_liquidity'],                  # Market liquidity
                self.credit_history                         # Credit history    
            ], dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            return torch.tensor([
                self.credit_score / 850,                    # Normalized credit score
                self.debt_to_income_ratio(),                # Current DTI ratio
                self.get_payment_success_rate(),            # Historical payment performance
                loan_offer[0],                              # Interest rate (already normalized)
                loan_offer[1] / 100000,                     # Normalized loan amount
                loan_offer[2] / 36,                         # Normalized loan term
                state['economic_cycle'],                    # Economic cycle
                state['market_liquidity']                   # Market liquidity
            ], dtype=torch.float32, device=self.device).unsqueeze(0)



    def evaluate_loan(self, loan, market_state):
        # Evaluate whether to accept or reject a loan offer
        if loan is None:
            return False

        # Initial rejection criteria
        if self.debt_to_income_ratio() >= 0.5:  # Balanced DTI threshold
            return False
            
        if self.credit_score < 550:  # More lenient credit score requirement
            return False

        # Calculate total monthly obligations including the new loan
        potential_monthly_payment = loan.monthly_payment()
        current_monthly_payments = sum(l.monthly_payment() for l in self.loans)
        total_monthly_obligations = potential_monthly_payment + current_monthly_payments
        monthly_income = self.income / 12

        # Reject if total monthly payments would exceed 50% of monthly income
        if total_monthly_obligations / monthly_income > 0.5:
            return False

        sample = random.random()
        # Epsilon-greedy policy (Exploration vs Exploitation)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # Convert the state and loan offer to a tensor
        loan_offer = (loan.interest_rate, loan.amount, loan.term)
        state_tensor = self.state_to_tensor(market_state, loan_offer)

        # Choose action based on epsilon-greedy policy
        if sample > eps_threshold:
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
                action = action_values.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.output_size)]], device=self.device, dtype=torch.long)

        # Enhanced decision making with risk and financial factors
        decision = action.item() == 1
        if decision:
            affordability = self.calculate_affordability(loan)
            risk_factor = np.random.random() * self.risk_tolerance
            literacy_factor = np.random.random() * self.financial_literacy
            payment_history_factor = self.get_payment_success_rate()
            
            # More balanced acceptance criteria
            decision = (
                affordability > 0.6 and    # More balanced threshold
                risk_factor > 0.4 and      # More balanced threshold
                literacy_factor > 0.4 and   # More balanced threshold
                payment_history_factor > 0.6 and  # More balanced threshold
                market_state['economic_cycle'] >= -0.5  # Accept loans in moderately negative to positive conditions
            )

        return decision

    def calculate_affordability(self, loan):
        # Enhanced affordability calculation
        monthly_payment = loan.monthly_payment()
        current_payments = sum(l.monthly_payment() for l in self.loans)
        monthly_income = self.income / 12
        disposable_income = monthly_income - current_payments - self.debt / 12
        
        if monthly_payment <= 0:
            return 0
        
        affordability_ratio = disposable_income / monthly_payment
        return min(1.0, max(0.0, affordability_ratio))

    def apply_for_loan(self, loan):
        print(f"Borrower {self.id} applied for a loan of ${loan.balance} at {loan.interest_rate}% interest")
        if self.can_borrow():
            self.loans.append(loan)
            self.debt += loan.balance
            self.loan_history.append(1)  # Record successful loan
            return True
        return False

    def can_pay(self, loan):
        # Check if the borrower can make the monthly payment
        return (self.income//12) >= loan.monthly_payment()

    def improve_credit_score(self, points):
        # Improve the credit score of the borrower
        self.credit_score = max(300, min(850, self.credit_score + points))

    def update_state(self, market_state, action, reward, next_state, loan_offer):
        # Update performance metrics
        self.reward_history.append(reward)
        self.avg_reward = sum(self.reward_history) / len(self.reward_history)

        # Store experience and train network
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

        # Compute Q-values and loss
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def reset(self):
        self.loans = []
        self.debt = 0
        self.loan_history = []
        self.payment_history = []
        self.reward_history.clear()
        self.avg_reward = 0
        if self.use_credit_history:
            self.credit_history = 6

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def update_credit_length(self, term):
        self.credit_length += term    # term is in month
