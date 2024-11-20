import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math

# The experience tuple is used to store the experience of the lender in the market.
Experience = namedtuple(
    'Experience', ('state', 'action', 'reward', 'next_state'))

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
        self.initial_capital = initial_capital if best_values is None else best_values.get(
            'highest_lender_capital', initial_capital)
        self.capital = self.initial_capital
        self.risk_tolerance = risk_tolerance if best_values is None else np.random.uniform(
            0.1, 0.9)
        self.loans = []
        self.min_loan_amount = 1000  # Smallest loan amount
        self.max_loan_amount = 100000  # Largest loan amount
        self.min_interest_rate = 0.03
        self.max_interest_rate = 0.09

        if best_values and 'optimal_interest_rate' in best_values:
            self.optimal_interest_rate = best_values['optimal_interest_rate']
        else:
            self.optimal_interest_rate = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # 10 features in the state (Features of the lender and market)
        self.input_size = 10
        self.output_size = 20  # 20 loan products
        self.policy_net = DQN(self.input_size, self.output_size).to(
            self.device)  # DQN model for the lender
        self.target_net = DQN(self.input_size, self.output_size).to(
            self.device)  # Target DQN model for the lender
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Adam optimizer for the DQN model this is used to update the weights of the model, why? Because the DQN model is a neural network and the weights of the neural network need to be updated in order to learn the Q-Values.
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        # Replay memory for the lender, why 10000? Becuase this the number of experiences that the lender can store in the replay memory.
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        # Discount factor why? Because the discount factor is used to discount future rewards in the reinforcement learning algorithm.
        self.gamma = 0.99
        # Exploration means that the agent is exploring the environment to learn more about it. Exploitation means that the agent is exploiting the knowledge it has already gained to maximize its rewards.
        self.eps_start = 0.9  # Episilon start why? Because the DQN model uses an epsilon greedy policy to explore the environment. The epsilon greedy policy is used to balance exploration and exploitation in the reinforcement learning algorithm.
        # Epsilon end why? Because the epsilon greedy policy is used to balance exploration and exploitation in the reinforcement learning algorithm.
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
        # Epsilon greedy policy
        # Math : eps_threshold = eps_end + (eps_start - eps_end) * exp(-1. * steps_done / eps_decay)
        # Explaination : Epsilon threshold is used to balance exploration and exploitation in the reinforcement learning algorithm.
        # eps_end is the minimum value of epsilon, eps_start is the maximum value of epsilon, eps_decay is the decay rate of epsilon, steps_done is the number of steps taken by the agent.
        # as the number of steps taken by the agent increases, the epsilon value decreases, this leads to more exploitation and less exploration.
        # the number of steps taken by the agent is stored in the steps_done variable. Math.exp(- 1 * steps_done / eps_decay) is used to calculate the epsilon value.
        # Why -1 * steps_done? Because the epsilon value should decrease as the number of steps taken by the agent increases.
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # If the sample is greater than the epsilon threshold, then the agent will exploit the environment by taking the action with the highest Q-Value.
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                action_values = self.policy_net(state_tensor)
                action = action_values.max(1)[1].view(1, 1)
        else:
            # If the sample is less than the epsilon threshold, then the agent will explore the environment by taking a random action.
            action = torch.tensor(
                [[random.randrange(self.output_size)]], device=self.device, dtype=torch.long)

        action_item = action.item()
        # The action space is divided into 3 parts: interest rate, loan amount, and term.
        # base_rate is the base interest rate for the loan product.
        # market_adjustment is the adjustment to the base interest rate based on the market conditions.
        # liquidity_adjustment is the adjustment to the base interest rate based on the market liquidity.
        base_rate = (action_item % 10) / 9 * (self.max_interest_rate -
                                              self.min_interest_rate) + self.min_interest_rate
        market_adjustment = (state['default_rate'] - 0.05) * 0.5
        liquidity_adjustment = (1 - state['market_liquidity']) * 0.05

        random_factor = np.random.normal(0, 0.01)  # Introduce randomness

        if self.optimal_interest_rate is not None:
            interest_rate = max(self.min_interest_rate, min(self.max_interest_rate,
                                self.optimal_interest_rate + market_adjustment + liquidity_adjustment + random_factor))
        else:
            interest_rate = max(self.min_interest_rate, min(self.max_interest_rate,
                                base_rate + market_adjustment + liquidity_adjustment + random_factor))

        # action_item // 10 is the loan amount index and is % 5 to get the loan amount factor, finally, divide by 4 to get the loan amount.
        # This is multiplied by the min and max loan amount to get the loan amount, this is done to scale the loan amount to the min and max loan amount.
        # Followed by adding the min loan amount to get the final loan amount.
        loan_amount = ((action_item // 10) % 5) / 4 * \
            (self.max_loan_amount - self.min_loan_amount) + self.min_loan_amount
        # Term is calculated by taking the action item and dividing by 50 to get the term index, followed by multiplying by 12 to get the term.
        term = ((action_item // 50) % 3) * 12 + 12

        return interest_rate, loan_amount, term

    def assess_loan(self, loan, borrower):
        # Asses the creditworthiness of the borrower
        # TODO: Modify this to be more realistic.
        # NOTE: A possible idea is to include credit history?
        credit_score_factor = (borrower.credit_score - 300) / 550
        # Debt to income ratio (important factor in loan assessment)
        dti_factor = 1 - borrower.debt_to_income_ratio()
        # Loan amount factor is calculated by taking the loan amount and dividing by the capital of the lender.
        loan_amount_factor = 1 - (loan.amount / self.capital)
        # Many banks use loan scoring models to assess the creditworthiness of borrowers.
        # This is a very simplified version of a loan scoring model.
        # But, complex loan scoring models can take into account many more factors like employment history, loan purpose, etc.
        loan_score = (credit_score_factor * 0.3 + dti_factor * 0.3 +
                      loan_amount_factor * 0.4) * (self.risk_tolerance + 0.2)
        # The loan is granted if the loan score is greater than a random number between 0 and 1 (Mood of the lender ? Randomness)
        return np.random.random() < loan_score

    def grant_loan(self, loan):
        # If the lender has enough capital, the loan is granted
        if self.capital >= loan.amount:
            self.capital -= loan.amount
            self.loans.append(loan)
            return True
        return False

    def calculate_recovery_rate(self, credit_score):
        # Depending on the borrower's credit score, 300 - 850 maps to a recovery rate of 5%-50%
        min_credit_score = 300
        max_credit_score = 850
        min_default_return_rate = 0.05
        max_default_return_rate = 0.5

        recovery_rate = min_default_return_rate + (max_default_return_rate - min_default_return_rate) * (
            credit_score - min_credit_score) / (max_credit_score - min_credit_score)

        return recovery_rate

    def recover_loan(self, loan):
        # NOTE: Originally, in case of default, the lender always recovers 50% of the loan amount. This is not very realistic.
        # NOTE: Improvement - depending on the borrower's credit score, 300 - 850 maps to a recovery rate of 5%-50%
        self.loans.remove(loan)
        recovery = loan.balance * \
            self.calculate_recovery_rate(loan.borrower.credit_score[-1])
        self.capital += recovery

    def update_state(self, state, action, reward, next_state):
        # Update the state of the lender and train the DQN
        action_index = self.action_to_index(action)
        self.memory.push(self.state_to_tensor(state),
                         torch.tensor([[action_index]],
                                      device=self.device, dtype=torch.long),
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
        # Explain: The Q-Learning algorithm is used to update the Q-Values of the DQN model.
        # Here, the self.policy_net is used to predict the Q-Values for the state-action pairs, and then gathered
        # So state_action_values would look like this: [[Q-Value for action 1], [Q-Value for action 2], ...]
        # Similarly, the self.target_net is used to predict the Q-Values for the next state, and then the maximum Q-Value is taken.
        # So next_state_values would look like this: [Max Q-Value for next state]
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)
        # Detach the tensor from the computation graph why? Because we don't want to update the target network. (AutoGrad)
        next_state_values = self.target_net(
            next_state_batch).max(1)[0].detach()

        # Main equation of Q-Learning
        # Q-Learning equation: Q(s, a) = r + γ * max(Q(s', a'))
        # In our context, we use the policy_net to predict Q(s, a) and the target_net to predict Q(s', a')
        # state_action_values: Q(s, a) predicted by policy_net
        # next_state_values: max(Q(s', a')) predicted by target_net
        # reward_batch: r
        # self.gamma: γ
        # Here, we apply the discount factor to the next state values and add the reward to get the expected state-action values.
        # In DQN, we find the future reward by taking the maximum Q-Value of the next state, and then we apply the discount factor to it.
        # The expected state-action values are calculated by adding the reward (immediate reward) to the future reward.
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        # Apply the smooth L1 loss function to calculate the loss
        # Smooth L1 loss is a combination of L1 loss and L2 loss (Why this loss function?)
        # We use this particular loss function because it is less sensitive to outliers and provides a smooth gradient.
        # Why is it less sensitive to outliers? Because it uses a quadratic function for small errors and a linear function for large errors.
        # Why do we need a smooth gradient? To prevent the model from diverging (very high rewards) during training.
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def action_to_index(self, action):
        interest_rate, loan_amount, term = action
        interest_rate_index = int((interest_rate - self.min_interest_rate) /
                                  (self.max_interest_rate - self.min_interest_rate) * 9)
        loan_amount_index = int((loan_amount - self.min_loan_amount) /
                                (self.max_loan_amount - self.min_loan_amount) * 4)
        term_index = (term - 12) // 12
        return interest_rate_index + loan_amount_index * 10 + term_index * 50

    def reset(self):
        self.capital = self.initial_capital
        self.loans = []
        self.risk_tolerance = np.random.uniform(0.1, 0.9)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
