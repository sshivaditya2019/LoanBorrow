import numpy as np
from loan import Loan

class LoanMarketEnvironment:
    def __init__(self, lenders, borrowers, best_values=None):
        self.lenders = lenders
        self.borrowers = borrowers
        self.loans = []
        self.time_step = 0
        self.economic_cycle = 0  # 0: neutral, 1: boom, -1: recession
        self.cycle_duration = 0 # Duration of the current economic cycle
        self.max_cycle_duration = 60  # 5 years
        self.min_interest_rate = 0.03  # 1% minimum interest rate
        self.state = self.get_state() 
        self.best_values = best_values # Store the best values for each feature
        self.inflation = 0.04  # 4% annual inflation rate

    def calculate_lender_reward(self, loan, is_default=False):
        """Calculate reward for lender based on loan performance"""
        if is_default:
            # Penalize defaults heavily
            return -loan.amount * 2
        
        # Immediate reward for giving out a loan
        immediate_reward = loan.amount * 0.05
        
        # Calculate return on investment
        interest_earned = loan.total_interest()
        risk_adjusted_return = interest_earned * (1 - self.get_default_rate())
        
        # Reward for good risk assessment
        credit_score_factor = (loan.borrower.credit_score - 300) / 550
        dti_factor = 1 - loan.borrower.debt_to_income_ratio()
        risk_assessment_bonus = loan.amount * 0.1 * (credit_score_factor + dti_factor) / 2
        
        return immediate_reward + risk_adjusted_return + risk_assessment_bonus

    def calculate_borrower_reward(self, loan, is_default=False):
        """Calculate reward for borrower based on loan terms and outcome"""
        if is_default:
            # Severe penalty for defaulting
            return -loan.amount * 3
        
        # Immediate reward for getting a loan
        immediate_reward = loan.amount * 0.1
        
        # Interest rate optimization (reward lower rates)
        rate_factor = 1 - (loan.interest_rate - self.min_interest_rate) / (0.075 - self.min_interest_rate)
        interest_reward = loan.amount * 0.2 * rate_factor
        
        # Financial health factors
        credit_score_bonus = (loan.borrower.credit_score - 300) / 550 * loan.amount * 0.1
        dti_ratio = loan.borrower.debt_to_income_ratio()
        
        # Penalize high DTI ratios exponentially
        dti_penalty = -np.exp(max(0, dti_ratio - 0.4)) * loan.amount * 0.1
        
        # Affordability check
        monthly_payment = loan.monthly_payment()
        monthly_income = loan.borrower.income / 12
        affordability_ratio = monthly_payment / monthly_income
        
        # Penalize taking loans that are too large relative to income
        if affordability_ratio > 0.5:
            affordability_penalty = -loan.amount * (affordability_ratio - 0.5) * 2
        else:
            affordability_penalty = 0
        
        return immediate_reward + interest_reward + credit_score_bonus + dti_penalty + affordability_penalty

    def get_state(self):
        # Get the current debts
        for borrower in self.borrowers:
            print("Income :", borrower.income, "Debt :", borrower.debt)
        for borrower in self.borrowers:
            print(f"Borrower {borrower.id} debt: {borrower.debt:.2f} load ids: {[loan.id for loan in borrower.loans]}")
        if self.loans:
            return {
            'avg_credit_score': np.mean([b.credit_score for b in self.borrowers]),
            'avg_income': np.mean([b.income for b in self.borrowers]),
            'avg_debt': np.mean([l.amount for l in self.loans]),
            'num_loans': len(self.loans),
            'default_rate': self.get_default_rate(),
            'avg_interest_rate': self.get_avg_interest_rate(),
            'market_liquidity': self.get_market_liquidity(),
            'economic_cycle': self.economic_cycle,
            'time_step': self.time_step,
            'should_interest_rate_increase': 1
            }
        else:
            return {
            'avg_credit_score': np.mean([b.credit_score for b in self.borrowers]),
            'avg_income': np.mean([b.income for b in self.borrowers]),
            'avg_debt': 0,
            'num_loans': 0,
            'default_rate': 0,
            'avg_interest_rate': self.min_interest_rate,
            'market_liquidity': self.get_market_liquidity(),
            'economic_cycle': self.economic_cycle,
            'time_step': self.time_step,
            'should_interest_rate_increase': 0
            }

    def get_default_rate(self):
        # Default rate is the ratio of defaulted loans to total loans
        # Useful to measure the risk of the loan market
        if not self.loans:
            return 0
        print("Total Number of default loans: ", sum(loan.is_defaulted() for loan in self.loans))
        return sum(loan.is_defaulted() for loan in self.loans) / len(self.loans)

    def get_avg_interest_rate(self):
        # Average interest rate of all loans offered in the market
        if not self.loans:
            return self.min_interest_rate
        return max(self.min_interest_rate, np.mean([loan.interest_rate for loan in self.loans]))

    def get_market_liquidity(self):
        # Market liquidity is the ratio of total lender capital to total borrower debt capacity
        # Allows to measure the ability of the market to provide loans
        total_lender_capital = sum(lender.capital for lender in self.lenders)
        total_borrower_debt_capacity = sum(max(0, borrower.income * 0.5 - borrower.debt) for borrower in self.borrowers)
        return min(1, total_lender_capital / max(1, total_borrower_debt_capacity))

    def update_economic_cycle(self):
        # Randomly change the economic cycle every 5 years
        self.cycle_duration += 1
        if self.cycle_duration >= self.max_cycle_duration:
            self.economic_cycle = np.random.choice([-1, 0, 1])
            self.cycle_duration = 0

    def apply_economic_effects(self):
        # Apply economic effects on borrowers
        # By increasing or decreasing their income and credit score
        for borrower in self.borrowers:
            if self.economic_cycle == 1:  # boom
                borrower.income *= 1.0 * (1 + (self.inflation)/12)
                borrower.improve_credit_score(1)
            elif self.economic_cycle == -1:  # recession
                borrower.income *= 1.0 * (1 - (0.01/12))
                borrower.improve_credit_score(-1)

    def step(self, lender_rewards=None, borrower_rewards=None):
        # A single step in the simulation
        self.time_step += 1
        self.update_economic_cycle() # Update the economic cycle every 5 years
        self.apply_economic_effects() # Apply economic effects on borrowers

        if lender_rewards is None:
            lender_rewards = {lender.id: 0 for lender in self.lenders}
        if borrower_rewards is None:
            borrower_rewards = {borrower.id: 0 for borrower in self.borrowers}
        
        # Lenders make loan offers
        loan_offers = [] # List of all loan offers
        for lender in self.lenders:
            if lender.capital > lender.min_loan_amount: # Only make offers if the lender has enough capital
                action = lender.get_action(self.state) # Get the action from the policy network
                interest_rate, loan_amount, term = action 
                interest_rate = max(interest_rate, self.min_interest_rate)  # Enforce minimum interest rate
                loan_offers.append((lender, (interest_rate, loan_amount, term))) # Add the loan offer to the list
        print("Loan offers: ", loan_offers)
        
        # Borrowers evaluate loan offers
        for borrower in self.borrowers:
            if borrower.can_borrow(): # Only evaluate loan offers if the borrower can borrow
                for lender, offer in loan_offers:
                    interest_rate, loan_amount, term = offer
                    loan = Loan(lender, borrower, loan_amount, interest_rate, term) # Create a loan object
                    
                    if lender.assess_loan(loan, borrower): # Check if the lender can grant the loan
                        decision = borrower.evaluate_loan(loan, self.state) # Evaluate the loan offer
                        
                        if decision:
                            if borrower.apply_for_loan(loan) and lender.grant_loan(loan): # Apply for the loan and grant it
                                self.loans.append(loan)
                                # Calculate rewards based on loan terms and financial health
                                borrower_rewards[borrower.id] += self.calculate_borrower_reward(loan)
                                lender_rewards[lender.id] += self.calculate_lender_reward(loan)
                                self.state['should_interest_rate_increase'] = 1
                                print(f"Loan created: Lender {lender.id}, Borrower {borrower.id}, Amount: {loan_amount:.2f}, Interest: {interest_rate:.2%}, Term: {term}")
                        else:
                            self.state['should_interest_rate_increase'] = 0
                            # Small penalty for rejected offers
                            lender_rewards[lender.id] -= loan_amount * 0.01

        # Process existing loans
        for loan in self.loans[:]:
            if loan.make_payment():
                # Reward successful payments
                payment = loan.monthly_payment()
                interest_portion = payment * loan.interest_rate / 12
                borrower_rewards[loan.borrower.id] += payment * 0.05  # Small reward for timely payment
                lender_rewards[loan.lender.id] += interest_portion * 0.1  # Reward for earning interest
            elif loan.is_active == False and loan.is_defaulted() == False:
                # Reward successful loan completion
                borrower_rewards[loan.borrower.id] += self.calculate_borrower_reward(loan) * 2
                lender_rewards[loan.lender.id] += self.calculate_lender_reward(loan) * 2
                self.loans.remove(loan)
                print(f"Loan paid off: Lender {loan.lender.id}, Borrower {loan.borrower.id}, Amount: {loan.amount:.2f}")
            elif loan.is_defaulted():
                # Penalize defaults
                borrower_rewards[loan.borrower.id] += self.calculate_borrower_reward(loan, True)
                lender_rewards[loan.lender.id] += self.calculate_lender_reward(loan, True)
                loan.lender.recover_loan(loan)
                loan.borrower.recover_loan(loan)
                self.loans.remove(loan)
                print(f"Loan defaulted: Lender {loan.lender.id}, Borrower {loan.borrower.id}, Amount: {loan.amount:.2f}")
            else:
                print(f"Loan active: Lender {loan.lender.id}, Borrower {loan.borrower.id}, Amount: {loan.amount:.2f}, Balance: {loan.balance:.2f}")

        # Update states and policies
        next_state = self.get_state()
        for lender in self.lenders:
            lender.update_state(self.state, lender.get_action(self.state), lender_rewards[lender.id], next_state)
        for borrower in self.borrowers:
            # Use a dummy loan offer when there's no actual loan to evaluate
            dummy_loan_offer = (self.min_interest_rate, 0, 12)
            action = borrower.evaluate_loan(Loan(None, borrower, 0, self.min_interest_rate, 12), self.state)
            borrower.update_state(self.state, action, borrower_rewards[borrower.id], next_state, dummy_loan_offer)

        self.state = next_state

        # Stop when the simulation reaches the maximum time step or when there are no more loans or capital
        done = self.time_step >= 720 or len(self.loans) == 0 and all(lender.capital <= lender.min_loan_amount for lender in self.lenders)

        return self.state, lender_rewards, borrower_rewards, done

    def reset(self):
        self.loans = []
        self.time_step = 0
        self.economic_cycle = 0
        self.cycle_duration = 0
        for borrower in self.borrowers:
            borrower.reset()
        for lender in self.lenders:
            lender.reset()
        self.state = self.get_state()
        return self.state

    def print_statistics(self):
        print(f"Time step: {self.time_step}")
        print(f"Economic cycle: {'Boom' if self.economic_cycle == 1 else 'Recession' if self.economic_cycle == -1 else 'Neutral'}")
        print(f"Average credit score: {self.state['avg_credit_score']:.2f}")
        print(f"Average income: ${self.state['avg_income']:.2f}")
        print(f"Average debt: ${self.state['avg_debt']:.2f}")
        print(f"Number of active loans: {self.state['num_loans']}")
        print(f"Default rate: {self.state['default_rate']:.2%}")
        print(f"Average interest rate: {self.state['avg_interest_rate']:.2%}")
        print(f"Market liquidity: {self.state['market_liquidity']:.2f}")
        print(f"Total lender capital: ${sum(lender.capital for lender in self.lenders):.2f}")
        print(f"Total borrower debt: ${sum(borrower.debt for borrower in self.borrowers):.2f}")

    def update_best_values(self):
        if self.best_values is None:
            self.best_values = {}

        self.best_values['highest_credit_score'] = max(self.best_values.get('highest_credit_score', 0), 
                                                       max(b.credit_score for b in self.borrowers))
        self.best_values['lowest_debt'] = min(self.best_values.get('lowest_debt', float('inf')), 
                                              min(b.debt for b in self.borrowers))
        self.best_values['highest_income'] = max(self.best_values.get('highest_income', 0), 
                                                 max(b.income for b in self.borrowers))
        self.best_values['highest_lender_capital'] = max(self.best_values.get('highest_lender_capital', 0), 
                                                         max(l.capital for l in self.lenders))
        self.best_values['lowest_default_rate'] = min(self.best_values.get('lowest_default_rate', float('inf')), 
                                                      self.get_default_rate())
        self.best_values['optimal_interest_rate'] = self.get_avg_interest_rate() 
        self.best_values['optimal credit score'] = self.state['avg_credit_score']
        self.best_values['optimal income'] = self.state['avg_income']
        self.best_values['optimal debt'] = self.state['avg_debt']
