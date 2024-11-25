import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from lender import Lender
from borrower import Borrower
from environment import LoanMarketEnvironment
from gui import Visualization

def plot_results(history):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Loan Market Simulation Results')

    axs[0, 0].plot(history['avg_credit_score'])
    axs[0, 0].set_title('Average Credit Score')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Credit Score')

    axs[0, 1].plot(history['avg_interest_rate'])
    axs[0, 1].set_title('Average Interest Rate')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Interest Rate')

    axs[1, 0].plot(history['num_loans'])
    axs[1, 0].set_title('Number of Active Loans')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Number of Loans')

    axs[1, 1].plot(history['default_rate'])
    axs[1, 1].set_title('Default Rate')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Default Rate')

    axs[2, 0].plot(history['market_liquidity'])
    axs[2, 0].set_title('Market Liquidity')
    axs[2, 0].set_xlabel('Time Step')
    axs[2, 0].set_ylabel('Liquidity')

    axs[2, 1].plot(history['total_lender_capital'])
    axs[2, 1].set_title('Total Lender Capital')
    axs[2, 1].set_xlabel('Time Step')
    axs[2, 1].set_ylabel('Capital')

    plt.tight_layout()
    plt.savefig('loan_market_simulation_results.png')
    plt.close()

def save_best_values(best_values):
    with open('best_values.json', 'w') as f:
        json.dump(best_values, f)

def load_best_values():
    try:
        with open('best_values.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main(use_best_values=False):
    num_lenders = 5
    num_borrowers = 20 # Also be changed in the lender.py's for the DQN's output size
    num_episodes = 1
    max_time_steps = 720

    best_values = load_best_values() if use_best_values else None

    # All the lenders start with $1,000,000
    lenders = [
        Lender(i, initial_capital=1_000_000, risk_tolerance=np.random.uniform(0.1, 0.9), best_values=best_values)
        for i in range(num_lenders)
    ]

    borrowers = [
        Borrower(i, best_values=best_values)
        for i in range(num_borrowers)
    ]

    # GUI
    env = LoanMarketEnvironment(lenders, borrowers, best_values=best_values)

    vis = Visualization()

    history = {
        'avg_credit_score': [],
        'avg_income': [],
        'avg_debt': [],
        'num_loans': [],
        'default_rate': [],
        'avg_interest_rate': [],
        'market_liquidity': [],
        'total_lender_capital': []
    }

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        time_step = 0

        while not done and time_step < max_time_steps:
            next_state, lender_rewards, borrower_rewards, done = env.step()
            # Could be logged not implemented (W&B) or tensorboard
            vis.update(env)

            if vis.check_quit():
                done = True

            for key in history.keys():
                if key == 'total_lender_capital':
                    history[key].append(sum(lender.capital for lender in env.lenders))
                else:
                    history[key].append(next_state[key])

            env.update_best_values()

            if time_step % 60 == 0:
                print(f"Episode {episode}, Year {time_step // 12}")
                env.print_statistics()

            if time_step % 100 == 0:
                for lender in lenders:
                    lender.update_target_network()
                for borrower in borrowers:
                    borrower.update_target_network()

            state = next_state
            time_step += 1

        if done:
            break

    vis.close()

    plot_results(history)

    save_best_values(env.best_values)

    print("Simulation complete. Results saved to 'loan_market_simulation_results.png'")
    print("Best values saved to 'best_values.json'")

if __name__ == "__main__":
    # Add an argument to use the best values from previous runs
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_best_values', action='store_true', help='Use best values from previous runs')
    args = parser.parse_args()
    
    main(use_best_values=args.use_best_values)
