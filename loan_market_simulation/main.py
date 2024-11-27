import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
from datetime import datetime
from lender import Lender
from borrower import Borrower
from environment import LoanMarketEnvironment
from gui import Visualization

def calculate_moving_average(data, window=30):
    """Calculate moving average with the specified window"""
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

def plot_loan_metrics(history):
    """Plot loan metrics without economic cycle"""
    time_steps = np.array(range(len(history['economic_cycle'])))
    window = 30

    # Calculate moving averages
    ma_interest = calculate_moving_average(np.array(history['avg_interest_rate']), window)
    ma_accepted = calculate_moving_average(np.array(history['loan_decisions']['accepted']), window)
    ma_rejected = calculate_moving_average(np.array(history['loan_decisions']['rejected']), window)

    plt.figure(figsize=(15, 8))
    
    # Plot interest rate on primary y-axis
    ax1 = plt.gca()
    ax1.plot(time_steps[window-1:], ma_interest, label='Interest Rate', color='blue', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Interest Rate', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot loan decisions on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(time_steps[window-1:], ma_accepted, label='Accepted Loans', color='green', linewidth=2)
    ax2.plot(time_steps[window-1:], ma_rejected, label='Rejected Loans', color='red', linewidth=2)
    ax2.set_ylabel('Number of Loans')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Interest Rate vs Loan Decisions (30-period Moving Average)')
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/loan_metrics_{timestamp}.png')
    plt.close()

def plot_rewards(history):
    """Plot lender and borrower rewards"""
    time_steps = np.array(range(len(history['lender_rewards'])))
    window = 30

    # Calculate moving averages
    ma_lender = calculate_moving_average(np.array(history['lender_rewards']), window)
    ma_borrower = calculate_moving_average(np.array(history['borrower_rewards']), window)

    plt.figure(figsize=(15, 8))
    
    plt.plot(time_steps[window-1:], ma_lender, label='Lender Rewards', color='purple', linewidth=2)
    plt.plot(time_steps[window-1:], ma_borrower, label='Borrower Rewards', color='orange', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Rewards')
    plt.title('Agent Rewards Over Time (30-period Moving Average)')
    plt.legend()
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/rewards_{timestamp}.png')
    plt.close()

def plot_results(history):
    """Plot simulation results"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Loan Market Simulation Results')

    # Create time steps array for x-axis
    time_steps = np.array(range(len(history['economic_cycle'])))
    
    # Calculate moving averages
    window = 30  # 2.5 years window
    ma_start = window - 1  # Index where moving average starts
    
    # Market Health Metrics with Economic Cycle
    ax1 = axs[0, 0]
    credit_scores = np.array(history['avg_credit_score'])
    ma_credit = calculate_moving_average(credit_scores, window)
    ax1.plot(time_steps, credit_scores, label='Credit Score', color='blue', alpha=0.5)
    ax1.plot(time_steps[ma_start:], ma_credit, label='MA Credit Score', color='blue', linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_steps, history['economic_cycle'], label='Economic Cycle', color='red', linestyle='--', alpha=0.8)
    ax1.set_title('Average Credit Score vs Economic Cycle')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Credit Score', color='blue')
    ax1_twin.set_ylabel('Economic Cycle', color='red')
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax2 = axs[0, 1]
    interest_rates = np.array(history['avg_interest_rate'])
    ma_interest = calculate_moving_average(interest_rates, window)
    ax2.plot(time_steps, interest_rates, label='Interest Rate', color='blue', alpha=0.5)
    ax2.plot(time_steps[ma_start:], ma_interest, label='MA Interest Rate', color='blue', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time_steps, history['economic_cycle'], label='Economic Cycle', color='red', linestyle='--', alpha=0.8)
    ax2.set_title('Average Interest Rate vs Economic Cycle')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Interest Rate', color='blue')
    ax2_twin.set_ylabel('Economic Cycle', color='red')
    ax2.grid(True)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Loan Decisions - Accepted vs Rejected
    ax3 = axs[1, 0]
    accepted = np.array(history['loan_decisions']['accepted'])
    ma_accepted = calculate_moving_average(accepted, window)
    ax3.plot(time_steps, accepted, label='Accepted Loans', color='green', alpha=0.5)
    ax3.plot(time_steps[ma_start:], ma_accepted, label='MA Accepted', color='green', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_steps, history['economic_cycle'], label='Economic Cycle', color='red', linestyle='--', alpha=0.8)
    ax3.set_title('Accepted Loans vs Economic Cycle')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Number of Accepted Loans', color='green')
    ax3_twin.set_ylabel('Economic Cycle', color='red')
    ax3.grid(True)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax4 = axs[1, 1]
    rejected = np.array(history['loan_decisions']['rejected'])
    ma_rejected = calculate_moving_average(rejected, window)
    ax4.plot(time_steps, rejected, label='Rejected Loans', color='orange', alpha=0.5)
    ax4.plot(time_steps[ma_start:], ma_rejected, label='MA Rejected', color='orange', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time_steps, history['economic_cycle'], label='Economic Cycle', color='red', linestyle='--', alpha=0.8)
    ax4.set_title('Rejected Loans vs Economic Cycle')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Number of Rejected Loans', color='orange')
    ax4_twin.set_ylabel('Economic Cycle', color='red')
    ax4.grid(True)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Borrower Financial Health
    ax5 = axs[2, 0]
    income = np.array(history['avg_income'])
    ma_income = calculate_moving_average(income, window)
    ax5.plot(time_steps, income, label='Income', color='green', alpha=0.5)
    ax5.plot(time_steps[ma_start:], ma_income, label='MA Income', color='green', linewidth=2)
    ax5_twin = ax5.twinx()
    ax5_twin.plot(time_steps, history['economic_cycle'], label='Economic Cycle', color='red', linestyle='--', alpha=0.8)
    ax5.set_title('Average Income vs Economic Cycle')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Income', color='green')
    ax5_twin.set_ylabel('Economic Cycle', color='red')
    ax5.grid(True)
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax6 = axs[2, 1]
    debt = np.array(history['avg_debt'])
    ma_debt = calculate_moving_average(debt, window)
    ax6.plot(time_steps, debt, label='Debt', color='orange', alpha=0.5)
    ax6.plot(time_steps[ma_start:], ma_debt, label='MA Debt', color='orange', linewidth=2)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(time_steps, history['economic_cycle'], label='Economic Cycle', color='red', linestyle='--', alpha=0.8)
    ax6.set_title('Average Debt vs Economic Cycle')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Debt', color='orange')
    ax6_twin.set_ylabel('Economic Cycle', color='red')
    ax6.grid(True)
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/simulation_results_{timestamp}.png')
    plt.close()

    # Plot additional visualizations
    plot_loan_metrics(history)
    plot_rewards(history)

def save_best_values(best_values):
    """Save best values with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    # Save with timestamp
    with open(f'results/best_values_{timestamp}.json', 'w') as f:
        json.dump(best_values, f, indent=4)
    
    # Also save to standard location
    with open('best_values.json', 'w') as f:
        json.dump(best_values, f, indent=4)

def load_best_values():
    """Load best values with fallback options"""
    try:
        with open('best_values.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Try to find most recent best values in results directory
        try:
            result_files = [f for f in os.listdir('results') if f.startswith('best_values_')]
            if result_files:
                most_recent = max(result_files)
                with open(f'results/{most_recent}', 'r') as f:
                    return json.load(f)
        except (FileNotFoundError, OSError):
            pass
    return None

def main(use_best_values=False, num_lenders=5, num_borrowers=20, num_episodes=1, max_time_steps=720, use_credit_history=False):
    """Main simulation function"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load best values if requested
    best_values = load_best_values() if use_best_values else None

    # Initialize agents
    lenders = [
        Lender(i, initial_capital=1_000_000, risk_tolerance=np.random.uniform(0.3, 0.7), best_values=best_values, use_credit_history=use_credit_history)
        for i in range(num_lenders)
    ]

    borrowers = [
        Borrower(i, best_values=best_values, use_credit_history=use_credit_history)
        for i in range(num_borrowers)
    ]

    # Initialize environment and visualization
    env = LoanMarketEnvironment(lenders, borrowers, best_values=best_values, use_credit_history=use_credit_history)
    vis = Visualization()

    # Initialize history tracking
    history = {
        'avg_credit_score': [],
        'avg_income': [],
        'avg_debt': [],
        'num_loans': [],
        'avg_interest_rate': [],
        'total_lender_capital': [],
        'economic_cycle': [],
        'loan_decisions': {
            'accepted': [],
            'rejected': []
        },
        'lender_rewards': [],
        'borrower_rewards': []
    }

    for episode in range(num_episodes):
        done = False
        time_step = 0

        lender_rewards = {lender.id: 0 for lender in lenders}
        borrower_rewards = {borrower.id: 0 for borrower in borrowers}

        while not done and time_step < max_time_steps:
            next_state, lender_rewards, borrower_rewards, done = env.step(lender_rewards, borrower_rewards)
            
            # Update visualization and check for quit
            quit_requested = vis.update(env)
            if quit_requested:
                done = True

            # Update history
            for key in ['avg_credit_score', 'avg_income', 'avg_debt', 'num_loans', 
                       'avg_interest_rate', 'economic_cycle']:
                history[key].append(next_state[key])

            history['total_lender_capital'].append(sum(lender.capital for lender in env.lenders))

            # Track loan decisions
            history['loan_decisions']['accepted'].append(next_state['num_loans'])
            history['loan_decisions']['rejected'].append(next_state['num_rejected_loans'])

            # Track rewards
            history['lender_rewards'].append(sum(lender_rewards.values()))
            history['borrower_rewards'].append(sum(borrower_rewards.values()))

            env.update_best_values()

            # Print periodic statistics
            if time_step % 60 == 0:
                print(f"\nEpisode {episode + 1}/{num_episodes}, Year {time_step // 12}")
                env.print_statistics()

            # Update networks periodically
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

    # Cleanup and save results
    vis.close()
    plot_results(history)
    save_best_values(env.best_values)

    print("\n=== Simulation Complete ===")
    print(f"Results saved in 'results' directory")
    print(f"Duration: {time_step} time steps")
    print("\nFinal market state:")
    env.print_statistics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loan Market Simulation')
    parser.add_argument('--use_best_values', action='store_true', help='Use best values from previous runs')
    parser.add_argument('--num_lenders', type=int, default=5, help='Number of lenders')
    parser.add_argument('--num_borrowers', type=int, default=20, help='Number of borrowers')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--max_time_steps', type=int, default=720, help='Maximum time steps per episode')
    parser.add_argument('--use_credit_history', action='store_true', help='Use credit history for borrowers')
    
    args = parser.parse_args()
    
    main(use_best_values=args.use_best_values,
         num_lenders=args.num_lenders,
         num_borrowers=args.num_borrowers,
         num_episodes=args.num_episodes,
         max_time_steps=args.max_time_steps,
         use_credit_history=args.use_credit_history)
