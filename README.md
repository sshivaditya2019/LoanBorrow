# Equiloan Marketplace Simulation

## Description
The objective of this project is to create a loan marketplace consisting of borrowers and lenders. As expected, the lenders and the borrowers have competing interests. The lenders are interested in maximizing returns on their loans while minimizing losses from defaults. The borrowers are
interested in lowest interest rates and maximizing success rates of loan repayment.

Eventually, the expected result is to see that this marketplace becomes economically stable over time. The default rates should be low and the credit scores should be average
(i.e.: 600-700 range).
To model the loan marketplace environment, a Deep Reinforcement Learning (DRL) approach was taken. The actions that the borrowers and lenders take are modeled using
the Deep Q-Network (DQN) models.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sshivaditya2019/Equiloan.git
   cd Equiloan
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Simulation
There are two ways to run the simulation:

1. Standard mode (without using best values):
   ```bash
   python loan_market_simulation/main.py
   ```

2. Using best values from previous runs:
   ```bash
   python loan_market_simulation/main.py --use_best_values
   ```

## Implementation Details
- The project uses Deep Q-Network (DQN) to train both lender and borrower agents.
- The environment simulates various economic factors, including:
  - Economic cycles (Boom, Recession, Neutral)
  - Average credit score
  - Average income
  - Average debt
  - Number of active loans
  - Default rate
  - Average interest rate
  - Market liquidity
  - Total lender capital
  - Total borrower debt

## Document Overview
- `loan_market_simulation/`:
  - `main.py`: Main script to run the simulation.
  - `environment.py`: Environment class for the loan marketplace.
  - `lender.py`: Lender agent class.
  - `borrower.py`: Borrower agent class.
  - `gui.py`: GUI class to visualize the simulation.

