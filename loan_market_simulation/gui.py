import pygame
import random

class Visualization:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Loan Market Simulation")
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()

        self.lender_positions = {}
        self.borrower_positions = {}

    def draw_lender(self, lender, x, y):
        pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 10)
        text = self.font.render(f"L{lender.id}: ${lender.capital:.0f}", True, (0, 0, 0))
        self.screen.blit(text, (x + 15, y - 10))

    def draw_borrower(self, borrower, x, y):
        pygame.draw.circle(self.screen, (0, 0, 255), (x, y), 10)
        text = self.font.render(f"B{borrower.id}: ${borrower.debt:.0f}", True, (0, 0, 0))
        self.screen.blit(text, (x + 15, y - 10))

    def draw_loan(self, loan):
        lender_pos = self.lender_positions[loan.lender.id]
        borrower_pos = self.borrower_positions[loan.borrower.id]
        pygame.draw.line(self.screen, (255, 0, 0), lender_pos, borrower_pos, 2)

    def draw_statistics(self, env):
        stats = [
            f"Time step: {env.time_step}",
            f"Avg credit score: {env.state['avg_credit_score']:.2f}",
            f"Avg income: ${env.state['avg_income']:.2f}",
            f"Avg debt: ${env.state['avg_debt']:.2f}",
            f"Active loans: {env.state['num_loans']}",
            f"Default rate: {env.state['default_rate']:.2%}",
            f"Avg interest rate: {env.state['avg_interest_rate']:.2%}",
            f"Market liquidity: {env.state['market_liquidity']:.2f}"
        ]

        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (0, 0, 0))
            self.screen.blit(text, (10, 10 + i * 25))

    def update(self, env):
        self.screen.fill((255, 255, 255))

        # Update positions if not set
        if not self.lender_positions:
            self.lender_positions = {lender.id: (random.randint(50, self.width - 50), random.randint(50, self.height // 2 - 50)) for lender in env.lenders}
        if not self.borrower_positions:
            self.borrower_positions = {borrower.id: (random.randint(50, self.width - 50), random.randint(self.height // 2 + 50, self.height - 50)) for borrower in env.borrowers}

        # Draw lenders and borrowers
        for lender in env.lenders:
            self.draw_lender(lender, *self.lender_positions[lender.id])
        for borrower in env.borrowers:
            self.draw_borrower(borrower, *self.borrower_positions[borrower.id])

        # Draw loans
        for loan in env.loans:
            self.draw_loan(loan)

        # Draw statistics
        self.draw_statistics(env)

        pygame.display.flip()
        self.clock.tick(60)

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def close(self):
        pygame.quit()