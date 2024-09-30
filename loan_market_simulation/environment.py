import numpy as np
from loan import Loan

class LoanMarketEnvironment:
    def __init__(self, lenders, borrowers, best_values=None):
        self.lenders = lenders
        self.borrowers = borrowers
        self.loans = []
        self.time_step = 0
        self.economic_cycle = 0  # 0: neutral, 1: boom, -1: recession
        self.cycle_duration = 0
        self.max_cycle_duration = 60  # 5 years
        self.min_interest_rate = 0.01  # 1% minimum interest rate
        self.state = self.get_state()
        self.best_values = best_values

    def get_state(self):
        return {
            'avg_credit_score': np.mean([b.credit_score for b in self.borrowers]),
            'avg_income': np.mean([b.income for b in self.borrowers]),
            'avg_debt': np.mean([b.debt for b in self.borrowers]),
            'num_loans': len(self.loans),
            'default_rate': self.get_default_rate(),
            'avg_interest_rate': self.get_avg_interest_rate(),
            'market_liquidity': self.get_market_liquidity(),
            'economic_cycle': self.economic_cycle,
            'time_step': self.time_step
        }

    def get_default_rate(self):
        if not self.loans:
            return 0
        return sum(loan.is_defaulted() for loan in self.loans) / len(self.loans)

    def get_avg_interest_rate(self):
        if not self.loans:
            return self.min_interest_rate
        return max(self.min_interest_rate, np.mean([loan.interest_rate for loan in self.loans]))

    def get_market_liquidity(self):
        total_lender_capital = sum(lender.capital for lender in self.lenders)
        total_borrower_debt_capacity = sum(max(0, borrower.income * 0.5 - borrower.debt) for borrower in self.borrowers)
        return min(1, total_lender_capital / max(1, total_borrower_debt_capacity))

    def update_economic_cycle(self):
        self.cycle_duration += 1
        if self.cycle_duration >= self.max_cycle_duration:
            self.economic_cycle = np.random.choice([-1, 0, 1])
            self.cycle_duration = 0

    def apply_economic_effects(self):
        for borrower in self.borrowers:
            if self.economic_cycle == 1:  # boom
                borrower.income *= 1.01
                borrower.improve_credit_score(1)
            elif self.economic_cycle == -1:  # recession
                borrower.income *= 0.99
                borrower.improve_credit_score(-1)

    def step(self):
        self.time_step += 1
        self.update_economic_cycle()
        self.apply_economic_effects()

        lender_rewards = {lender.id: 0 for lender in self.lenders}
        borrower_rewards = {borrower.id: 0 for borrower in self.borrowers}

        # Lenders make loan offers
        loan_offers = []
        for lender in self.lenders:
            if lender.capital > lender.min_loan_amount:
                action = lender.get_action(self.state)
                interest_rate, loan_amount, term = action
                interest_rate = max(interest_rate, self.min_interest_rate)  # Enforce minimum interest rate
                loan_offers.append((lender, (interest_rate, loan_amount, term)))

        # Borrowers evaluate loan offers
        for borrower in self.borrowers:
            if borrower.can_borrow():
                for lender, offer in loan_offers:
                    interest_rate, loan_amount, term = offer
                    loan = Loan(lender, borrower, loan_amount, interest_rate, term)
                    
                    if lender.assess_loan(loan, borrower):
                        decision = borrower.evaluate_loan(loan, self.state)
                        
                        if decision:
                            if borrower.apply_for_loan(loan) and lender.grant_loan(loan):
                                self.loans.append(loan)
                                borrower_rewards[borrower.id] += loan_amount
                                lender_rewards[lender.id] -= loan_amount
                                print(f"Loan created: Lender {lender.id}, Borrower {borrower.id}, Amount: {loan_amount:.2f}, Interest: {interest_rate:.2%}, Term: {term}")

        # Process existing loans
        for loan in self.loans[:]:
            if loan.make_payment():
                payment = loan.monthly_payment()
                borrower_rewards[loan.borrower.id] -= payment
                lender_rewards[loan.lender.id] += payment
            elif loan.is_defaulted():
                recovery = loan.current_value()
                lender_rewards[loan.lender.id] -= (loan.balance - recovery)
                loan.lender.recover_loan(loan)
                self.loans.remove(loan)
                print(f"Loan defaulted: Lender {loan.lender.id}, Borrower {loan.borrower.id}, Amount: {loan.amount:.2f}")

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