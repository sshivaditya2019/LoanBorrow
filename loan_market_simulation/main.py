import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from lender import Lender
from borrower import Borrower
from environment import LoanMarketEnvironment
from gui import Visualization

def plot_results(history):
    fig, axs = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle('Loan Market Simulation Results')

    # Market Health Metrics
    axs[0, 0].plot(history['avg_credit_score'])
    axs[0, 0].set_title('Average Credit Score')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Credit Score')

    axs[0, 1].plot(history['avg_interest_rate'])
    axs[0, 1].set_title('Average Interest Rate')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Interest Rate')

    # Loan Activity Metrics
    axs[1, 0].plot(history['num_loans'])
    axs[1, 0].set_title('Number of Active Loans')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Number of Loans')

    axs[1, 1].plot(history['default_rate'])
    axs[1, 1].set_title('Default Rate')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Default Rate')

    # Economic Indicators
    axs[2, 0].plot(history['economic_cycle'])
    axs[2, 0].set_title('Economic Cycle')
    axs[2, 0].set_xlabel('Time Step')
    axs[2, 0].set_ylabel('Economic Cycle')

    axs[2, 1].plot(history['total_lender_capital'])
    axs[2, 1].set_title('Total Lender Capital')
    axs[2, 1].set_xlabel('Time Step')
    axs[2, 1].set_ylabel('Capital')

    # Borrower Financial Health
    axs[3, 0].plot(history['avg_income'])
    axs[3, 0].set_title('Average Income')
    axs[3, 0].set_xlabel('Time Step')
    axs[3, 0].set_ylabel('Income')

    axs[3, 1].plot(history['avg_debt'])
    axs[3, 1].set_title('Average Debt')
    axs[3, 1].set_xlabel('Time Step')
    axs[3, 1].set_ylabel('Debt')

    # Loan Decision Metrics
    axs[4, 0].plot(history['loan_decisions']['accepted'], label='Accepted', color='green')
    axs[4, 0].plot(history['loan_decisions']['rejected'], label='Rejected', color='red')
    axs[4, 0].set_title('Loan Decisions')
    axs[4, 0].set_xlabel('Time Step')
    axs[4, 0].set_ylabel('Count')
    axs[4, 0].legend()

    axs[4, 1].plot(history['loan_outcomes']['successful'], label='Successful', color='green')
    axs[4, 1].plot(history['loan_outcomes']['defaulted'], label='Defaulted', color='red')
    axs[4, 1].set_title('Loan Outcomes')
    axs[4, 1].set_xlabel('Time Step')
    axs[4, 1].set_ylabel('Count')
    axs[4, 1].legend()

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
    num_borrowers = 20
    num_episodes = 1
    max_time_steps = 720

    best_values = load_best_values() if use_best_values else None

    lenders = [
        Lender(i, initial_capital=1_000_000, risk_tolerance=np.random.uniform(0.1, 0.9), best_values=best_values)
        for i in range(num_lenders)
    ]

    borrowers = [
        Borrower(i, best_values=best_values)
        for i in range(num_borrowers)
    ]

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
        'total_lender_capital': [],
        'economic_cycle': [],
        'loan_decisions': {
            'accepted': [],
            'rejected': []
        },
        'loan_outcomes': {
            'successful': [],
            'defaulted': []
        }
    }

    # Track loan decisions and outcomes
    accepted_loans = 0
    rejected_loans = 0
    successful_loans = 0
    defaulted_loans = 0

    for episode in range(num_episodes):
        done = False
        time_step = 0

        lender_rewards = {lender.id: 0 for lender in lenders}
        borrower_rewards = {borrower.id: 0 for borrower in borrowers}

        while not done and time_step < max_time_steps:
            next_state, lender_rewards, borrower_rewards, done = env.step(lender_rewards, borrower_rewards)
            vis.update(env)

            if vis.check_quit():
                done = True

            # Update history
            for key in ['avg_credit_score', 'avg_income', 'avg_debt', 'num_loans', 
                       'default_rate', 'avg_interest_rate', 'market_liquidity', 'economic_cycle']:
                history[key].append(next_state[key])

            history['total_lender_capital'].append(sum(lender.capital for lender in env.lenders))

            # Track loan decisions
            current_accepted = len(env.loans)
            current_rejected = sum(1 for b in borrowers if not b.can_borrow())
            history['loan_decisions']['accepted'].append(current_accepted - accepted_loans)
            history['loan_decisions']['rejected'].append(current_rejected - rejected_loans)
            accepted_loans = current_accepted
            rejected_loans = current_rejected

            # Track loan outcomes
            current_defaulted = sum(1 for loan in env.loans if loan.is_defaulted())
            current_successful = sum(1 for loan in env.loans if not loan.is_active and not loan.is_defaulted())
            history['loan_outcomes']['defaulted'].append(current_defaulted - defaulted_loans)
            history['loan_outcomes']['successful'].append(current_successful - successful_loans)
            defaulted_loans = current_defaulted
            successful_loans = current_successful

            env.update_best_values()

            if time_step % 60 == 0:
                print(f"Episode {episode}, Year {time_step // 12}")
                env.print_statistics()

            if time_step % 100 == 0:
                for lender in lenders:
                    lender.update_target_network()
                for borrower in borrowers:
                    if borrower.debt == 0:
                        borrower.loans = []
                    borrower.update_target_network()

            time_step += 1

        if done:
            break

    vis.close()
    plot_results(history)
    save_best_values(env.best_values)

    print("Simulation complete. Results saved to 'loan_market_simulation_results.png'")
    print("Best values saved to 'best_values.json'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_best_values', action='store_true', help='Use best values from previous runs')
    args = parser.parse_args()
    
    main(use_best_values=args.use_best_values)
