import numpy as np

# A Simple Loan Class
# It represents the loan that is given by the lender to the borrower.
class Loan:
    def __init__(self, lender, borrower, amount, interest_rate, term):
        self.lender = lender
        self.id = np.random.randint(1000000)
        self.borrower = borrower
        self.amount = amount
        self.interest_rate = interest_rate
        self.term = 1
        self.balance = amount
        self.payments_made = 0
        self.missed_payments = 0 # Could be used to check credit worthiness of the borrower (Not used in the current implementation)
        self.is_active = True # Could be used to judge for future loans if the borrower is active or not (Not used in the current implementation)
        self.total_interest_paid = 0 # Could be used to check credit worthiness of the borrower (Not used in the current implementation)

    def reset(self):
        self.balance = self.amount
        self.payments_made = 0
        self.missed_payments = 0
        self.is_active = True
        self.total_interest_paid = 0

    def monthly_payment(self):
        # Monthly payment calculation uses compound interest formula
        r = self.interest_rate / 12
        mon =  self.amount * (1 + self.interest_rate * self.term)/12
        # print("Monthly Payment: ", self.amount, self.interest_rate, self.term, mon)
        return mon

    def make_payment(self):
        if self.is_active:
            payment = self.monthly_payment()
            if self.borrower.can_pay(self):
                self.balance -= payment
                self.payments_made += 1
                self.lender.capital += payment
                self.borrower.debt -= max(0, self.borrower.debt - payment)
                self.borrower.annual_income -= max(0, self.borrower.annual_income - payment)
                interest_portion = self.balance * (self.interest_rate / 12)
                self.total_interest_paid += interest_portion
                if self.balance <= 0:
                    self.lender.loans.remove(self)
                    self.borrower.loans.remove(self)
                    self.is_active = False
                    return False
                return True
            else:
                self.missed_payments += 1
                return False
        return False

    def is_defaulted(self):
        # A loan is defaulted if the borrower has missed 3 or more payments or if the debt to income ratio is greater than 0.6
        # That is, the borrower is not able to pay the loan
        if not self.borrower.can_borrow():
            self.is_active = False
            
            return True
        return False

    def current_value(self):
        # Value of the loan at the current time
        # Value is the remaining balance of the loan if the loan is not defaulted
        if self.is_defaulted():
            return self.balance * 0.5 # If the loan is defaulted, the value is 50% of the remaining balance (Always)
        else:
            remaining_payments = self.term - self.payments_made
            return self.monthly_payment() * remaining_payments

    def risk_score(self):
        # If the loan is defaulted, the risk score is 1
        if self.is_defaulted():
            return 1.0
        
        dti_ratio = self.borrower.debt_to_income_ratio()
        payment_history = self.missed_payments / (self.payments_made + self.missed_payments + 1)
        credit_score_factor = 1 - (self.borrower.credit_score - 300) / 300
        term_factor = self.term / 60
        
        risk_score = (dti_ratio * 0.7 + payment_history * 0.1 + credit_score_factor * 0.1 + term_factor * 0.1)
        # Clamp the risk score between 0 and 1
        return min(risk_score, 1.0)

    def expected_return(self):
        if self.is_defaulted():
            return -self.balance * 0.5
        
        total_expected_payments = self.monthly_payment() * (self.term - self.payments_made)
        expected_loss = total_expected_payments * self.risk_score()
        return total_expected_payments - expected_loss - (self.balance - self.amount)

    def total_interest(self):
        return self.monthly_payment() * self.term - self.amount

    def __str__(self):
        return f"Loan: Amount=${self.amount}, Interest={self.interest_rate:.2%}, Term={self.term} months, Balance=${self.balance:.2f}, Risk Score={self.risk_score():.2f}"
