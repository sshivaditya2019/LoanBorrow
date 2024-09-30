# Loan Borrow Marketplace Simulation

## Description
A Loan Borrow Marketplace simulation where agents (lenders and borrowers) interact with each other to procure loans at better rates. Lenders try to offer safer loans to borrowers while maximizing their returns. The simulation uses reinforcement learning to train agents to make optimal decisions in a dynamic economic environment.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-borrow-marketplace.git
   cd loan-borrow-marketplace
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

## Key Features
- **Reinforcement Learning**: Agents learn optimal strategies through interaction with the environment.
- **Best Values Tracking**: The simulation tracks and saves the best values achieved for various metrics across multiple runs.
- **Improved Agent Initialization**: Lenders and borrowers can be initialized with best values from previous runs, allowing for better starting conditions.
- **Dynamic Interest Rates**: Lenders adjust their interest rates based on market conditions, default rates, and liquidity.
- **Enhanced Economic Effects**: The simulation applies realistic economic effects on borrowers' income and credit scores during different economic cycles.
- **Visualization**: Real-time visualization of the marketplace using Pygame.

## Sample Results
Results may vary depending on the simulation parameters and whether best values are used. Here's an example of results from a simulation run:

### Initial State
- Average credit score: 526.25
- Average income: $73,064.93
- Average debt: $42,865.18
- Number of active loans: 9
- Default rate: 0.00%
- Average interest rate: 8.33%
- Market liquidity: 1.00
- Total lender capital: $4,991,000.00
- Total borrower debt: $857,303.69

### Final State
- Average credit score: 551.50
- Average income: $72,559.63
- Average debt: $42,959.88
- Number of active loans: 21
- Default rate: 0.00%
- Average interest rate: 6.79%
- Market liquidity: 1.00
- Total lender capital: $4,979,000.00
- Total borrower debt: $859,197.66

## Inference
- The average credit score increased, indicating that borrowers learned to manage their debt better.
- More loans were issued, suggesting increased market liquidity and borrower access to credit.
- The average interest rate decreased, implying a more competitive lending market benefiting borrowers.

