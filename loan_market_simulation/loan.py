import numpy as np

class Loan:
    def __init__(self, lender, borrower, amount, interest_rate, term):
        self.lender = lender
        self.id = np.random.randint(1000000)
        self.borrower = borrower
        self.amount = amount
        self.interest_rate = interest_rate
        self.term = term  # Term in months
        self.balance = amount
        self.payments_made = 0
        self.missed_payments = 0
        self.consecutive_missed_payments = 0
        self.is_active = True
        self.total_interest_paid = 0
        
        # Payment tracking
        self.payment_history = []  # Track all payment amounts
        self.payment_dates = []    # Track payment timing
        self.early_payments = 0    # Track early payments
        self.late_payments = 0     # Track late payments

    def reset(self):
        """Reset loan to initial state"""
        self.balance = self.amount
        self.payments_made = 0
        self.missed_payments = 0
        self.consecutive_missed_payments = 0
        self.is_active = True
        self.total_interest_paid = 0
        self.payment_history.clear()
        self.payment_dates.clear()
        # self.early_payments = 0
        # self.late_payments = 0

    def monthly_payment(self):
        """Calculate monthly payment using amortization formula"""
        if self.term <= 0:
            return 0
            
        r = self.interest_rate / 12  # Monthly interest rate
        n = self.term               # Total number of payments
        
        # Standard amortization formula
        if r > 0:
            payment = (self.amount * r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            payment = self.amount / n
            
        return payment

    def make_payment(self):
        """Process monthly payment"""
        if not self.is_active:
            return False

        payment = self.monthly_payment()
        
        if self.borrower.can_pay(self):
            # Process successful payment
            self.balance = max(0, self.balance - payment)
            self.payments_made += 1
            self.consecutive_missed_payments = 0
            
            # Update borrower financials
            self.borrower.debt = max(0, self.borrower.debt - payment)
            self.borrower.annual_income = max(0, self.borrower.annual_income - payment)
            
            # Calculate interest portion
            interest_portion = self.balance * (self.interest_rate / 12)
            self.total_interest_paid += interest_portion
            
            # Track payment
            self.payment_history.append(payment)
            self.payment_dates.append(self.payments_made)
            
            # Check if loan is fully paid
            if self.balance <= 0 or self.payments_made >= self.term:
                self.handle_loan_completion()
                return False
                
            return True
            
        else:
            # Handle missed payment
            self.missed_payments += 1
            self.consecutive_missed_payments += 1
            self.payment_history.append(0)
            self.payment_dates.append(self.payments_made)
            # self.late_payments += 1
            return False

    def handle_loan_completion(self):
        """Handle successful loan completion"""
        self.lender.capital += (self.amount + self.total_interest_paid)
        print(f"Lender: {self.lender.id} | Capital after Loan {self.id} paid: {self.lender.capital}")
        
        # Remove loan from both parties
        if self in self.lender.loans:
            self.lender.loans.remove(self)
        if self in self.borrower.loans:
            self.borrower.loans.remove(self)
        self.is_active = False
        
        # Improve borrower's credit score for successful completion
        self.borrower.improve_credit_score(20)

    def is_defaulted(self):
        """Check if loan is defaulted"""
        # Check multiple default conditions
        conditions = [
            self.consecutive_missed_payments >= 3,  # Three consecutive missed payments
            self.missed_payments / max(1, self.payments_made) > 0.3,  # More than 30% missed payments
            self.borrower.debt_to_income_ratio() > 0.5,  # DTI too high
            not self.borrower.can_borrow()  # Borrower's overall financial health
        ]
        
        if any(conditions):
            self.is_active = False
            return True
            
        return False

    def current_value(self):
        """Calculate current value of the loan"""
        if self.is_defaulted():
            recovery_rate = 0.5  # Base recovery rate
            # Adjust recovery rate based on borrower's credit score
            credit_factor = (self.borrower.credit_score - 300) / 550
            recovery_rate += credit_factor * 0.2  # Up to 20% better recovery for good credit
            return self.balance * recovery_rate
        else:
            remaining_payments = self.term - self.payments_made
            future_value = self.monthly_payment() * remaining_payments
            # Discount future value based on risk
            risk_score = self.risk_score()
            return future_value * (1 - risk_score * 0.2)

    def risk_score(self):
        """Calculate risk score"""
        if self.is_defaulted():
            return 1.0
        
        # Calculate component risk factors
        dti_ratio = self.borrower.debt_to_income_ratio()
        payment_history = self.missed_payments / (self.payments_made + self.missed_payments + 1)
        credit_score_factor = 1 - (self.borrower.credit_score - 300) / 550
        term_factor = self.term / 60
        
        # Payment consistency factor
        if self.payments_made > 0:
            consistency = len([p for p in self.payment_history if p > 0]) / len(self.payment_history)
        else:
            consistency = 0.5
        
        # Calculate weighted risk score
        risk_score = (
            dti_ratio * 0.3 +
            payment_history * 0.2 +
            credit_score_factor * 0.2 +
            term_factor * 0.1 +
            (1 - consistency) * 0.2
        )
        
        # Adjust for consecutive missed payments
        if self.consecutive_missed_payments > 0:
            risk_score += 0.1 * min(1.0, self.consecutive_missed_payments / 3)
        
        return min(risk_score, 1.0)

    def expected_return(self):
        """Calculate expected return"""
        if self.is_defaulted():
            return -self.balance * 0.5
        
        # Calculate expected payments
        total_expected_payments = self.monthly_payment() * (self.term - self.payments_made)
        
        # Calculate risk-adjusted loss
        risk_score = self.risk_score()
        expected_loss = total_expected_payments * risk_score
        
        # Consider payment history in return calculation
        if self.payments_made > 0:
            payment_reliability = len([p for p in self.payment_history if p > 0]) / len(self.payment_history)
            reliability_bonus = total_expected_payments * 0.1 * payment_reliability
        else:
            reliability_bonus = 0
        
        return total_expected_payments - expected_loss + reliability_bonus - (self.balance - self.amount)

    def total_interest(self):
        """Calculate total interest over loan term"""
        return self.monthly_payment() * self.term - self.amount

    def __str__(self):
        status = "Defaulted" if self.is_defaulted() else "Active" if self.is_active else "Completed"
        return (f"Loan {self.id}: Amount=${self.amount:.2f}, Interest={self.interest_rate:.2%}, "
                f"Term={self.term} months, Balance=${self.balance:.2f}, "
                f"Risk Score={self.risk_score():.2f}, Status={status}, "
                f"Payments Made={self.payments_made}, Missed={self.missed_payments}")

    def update_credit_length(self, term):
        self.credit_length += term