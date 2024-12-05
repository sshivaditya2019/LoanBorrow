import pygame
import random
import numpy as np
from pygame import gfxdraw

class Visualization:
    def __init__(self, width=1200, height=800, time_delay=1000):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.time_delay = time_delay
        pygame.display.set_caption("Loan Market Simulation")
        
        # Enhanced fonts
        self.title_font = pygame.font.Font(None, 36)
        self.header_font = pygame.font.Font(None, 28)
        self.stats_font = pygame.font.Font(None, 24)
        self.detail_font = pygame.font.Font(None, 20)
        
        self.clock = pygame.time.Clock()
        
        # Color scheme
        self.colors = {
            'background': (240, 240, 240),
            'panel': (220, 220, 220),
            'text': (20, 20, 20),
            'lender': (46, 139, 87),  # Sea green
            'borrower': (70, 130, 180),  # Steel blue
            'loan_active': (50, 205, 50),  # Lime green
            'loan_risky': (255, 140, 0),  # Dark orange
            'loan_default': (220, 20, 60),  # Crimson
            'loan_rejected': (169, 169, 169),  # Dark gray
            'highlight': (255, 215, 0),  # Gold
            'grid': (200, 200, 200)
        }
        
        # Initialize positions
        self.lender_positions = {}
        self.borrower_positions = {}
        self.selected_agent = None
        self.hover_agent = None
        
        # History tracking
        self.history_length = 100
        self.metric_history = {
            'avg_interest_rate': [],
            'rejection_rate': [] 
        }
        
        # Mini-chart dimensions
        self.chart_width = 150
        self.chart_height = 50
        
        # Grid settings
        self.grid_spacing = 50

    def draw_grid(self):
        """Draw background grid"""
        for x in range(0, self.width, self.grid_spacing):
            pygame.draw.line(self.screen, self.colors['grid'], (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.grid_spacing):
            pygame.draw.line(self.screen, self.colors['grid'], (0, y), (self.width, y), 1)

    def draw_agent(self, pos, radius, color, highlight=False, selected=False):
        """Draw an agent with anti-aliasing and effects"""
        x, y = pos
        gfxdraw.aacircle(self.screen, x+2, y+2, radius, (100, 100, 100))
        gfxdraw.filled_circle(self.screen, x+2, y+2, radius, (100, 100, 100))
        
        gfxdraw.aacircle(self.screen, x, y, radius, color)
        gfxdraw.filled_circle(self.screen, x, y, radius, color)
        
        if highlight or selected:
            gfxdraw.aacircle(self.screen, x, y, radius+2, self.colors['highlight'])
            if selected:
                gfxdraw.aacircle(self.screen, x, y, radius+3, self.colors['highlight'])

    def draw_lender(self, lender, x, y):
        is_selected = self.selected_agent == ('lender', lender.id)
        is_hovered = self.hover_agent == ('lender', lender.id)
        base_size = 10
        capital_factor = min(2.0, max(0.5, lender.capital / 1_000_000))
        radius = int(base_size * capital_factor)        
        self.draw_agent((x, y), radius, self.colors['lender'], is_hovered, is_selected)
        text = self.stats_font.render(f"L{lender.id}", True, self.colors['text'])
        self.screen.blit(text, (x + 15, y - 20))
        text = self.detail_font.render(f"${lender.capital:,.0f}", True, self.colors['text'])
        self.screen.blit(text, (x + 15, y))
        if is_selected or is_hovered:
            self.draw_lender_details(lender, x, y)

    def draw_borrower(self, borrower, x, y):
        """Draw borrower with enhanced visuals"""
        is_selected = self.selected_agent == ('borrower', borrower.id)
        is_hovered = self.hover_agent == ('borrower', borrower.id)
        base_size = 10
        credit_factor = min(2.0, max(0.5, borrower.credit_score / 700))
        radius = int(base_size * credit_factor)     
        self.draw_agent((x, y), radius, self.colors['borrower'], is_hovered, is_selected)
        text = self.stats_font.render(f"B{borrower.id}", True, self.colors['text'])
        self.screen.blit(text, (x + 15, y - 20))
        text = self.detail_font.render(f"${borrower.debt:,.0f}", True, self.colors['text'])
        self.screen.blit(text, (x + 15, y))
        if is_selected or is_hovered:
            self.draw_borrower_details(borrower, x, y)

    def draw_loan(self, loan, is_rejected=False):
        if loan.lender.id in self.lender_positions and loan.borrower.id in self.borrower_positions:
            start_pos = self.lender_positions[loan.lender.id]
            end_pos = self.borrower_positions[loan.borrower.id]
            
            if is_rejected:
                color = self.colors['loan_rejected']
                dash_length = 5
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                distance = max(1, (dx**2 + dy**2)**0.5)
                dash_count = int(distance / (2 * dash_length))
                for i in range(dash_count):
                    start_fraction = (2 * i * dash_length) / distance
                    end_fraction = min(1, ((2 * i + 1) * dash_length) / distance)
                    dash_start = (start_pos[0] + dx * start_fraction,
                                start_pos[1] + dy * start_fraction)
                    dash_end = (start_pos[0] + dx * end_fraction,
                              start_pos[1] + dy * end_fraction)
                    pygame.draw.line(self.screen, color, dash_start, dash_end, 1)
            else:
                risk_score = loan.risk_score()
                if risk_score < 0.3:
                    color = self.colors['loan_active']
                elif risk_score < 0.7:
                    color = self.colors['loan_risky']
                else:
                    color = self.colors['loan_default']
                thickness = max(1, min(5, int(loan.amount / 20000)))
                pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)

    def draw_mini_chart(self, data, title, x, y, color):
        """Draw a mini line chart"""
        if not data:
            return
            
        pygame.draw.rect(self.screen, self.colors['panel'], 
                        (x, y, self.chart_width, self.chart_height + 20))
        
        # Draw title
        title_text = self.detail_font.render(title, True, self.colors['text'])
        self.screen.blit(title_text, (x, y))
        
        # Draw chart
        data = data[-self.chart_width:]
        max_val = max(data) if data else 1
        min_val = min(data) if data else 0
        range_val = max(0.001, max_val - min_val)
        
        points = []
        for i, val in enumerate(data):
            px = x + i
            py = y + 20 + self.chart_height - int((val - min_val) / range_val * self.chart_height)
            points.append((px, py))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

    def draw_statistics_panel(self, env):
        """Draw enhanced statistics panel"""
        panel_width = 300
        panel_height = self.height
        panel_x = self.width - panel_width
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.colors['panel'], 
                        (panel_x, 0, panel_width, panel_height))
        
        # Draw title
        title = self.title_font.render("Market Statistics", True, self.colors['text'])
        self.screen.blit(title, (panel_x + 10, 10))
        
        # Calculate rejection rate
        total_loans = env.state['num_loans'] + env.state['num_rejected_loans']
        rejection_rate = env.state['num_rejected_loans'] / max(1, total_loans)
        
        # Update metric history
        self.metric_history['avg_interest_rate'].append(env.state['avg_interest_rate'])
        self.metric_history['rejection_rate'].append(rejection_rate)
        
        for metric in self.metric_history:
            if len(self.metric_history[metric]) > self.history_length:
                self.metric_history[metric].pop(0)
        
        # Draw mini charts
        self.draw_mini_chart(self.metric_history['avg_interest_rate'], 
                           "Avg Interest Rate", panel_x + 10, 60, (0, 0, 255))
        self.draw_mini_chart(self.metric_history['rejection_rate'],
                           "Rejection Rate", panel_x + 10, 140, (169, 169, 169))
        
        # Draw detailed statistics
        stats = [
            ("Economic Cycle", f"{'Boom' if env.economic_cycle == 1 else 'Recession' if env.economic_cycle == -1 else 'Neutral'}"),
            ("Time Step", f"{env.time_step}"),
            ("Avg Credit Score", f"{env.state['avg_credit_score']:.2f}"),
            ("Avg Income", f"${env.state['avg_income']:,.2f}"),
            ("Avg Debt", f"${env.state['avg_debt']:,.2f}"),
            ("Active Loans", f"{env.state['num_loans']}"),
            ("Rejected Loans", f"{env.state['num_rejected_loans']}"),
            ("Rejection Rate", f"{rejection_rate:.2%}"),
            ("Avg Interest Rate", f"{env.state['avg_interest_rate']:.2%}")
        ]
        
        y = 220
        for label, value in stats:
            text = self.stats_font.render(f"{label}:", True, self.colors['text'])
            self.screen.blit(text, (panel_x + 10, y))
            text = self.stats_font.render(value, True, self.colors['text'])
            self.screen.blit(text, (panel_x + 150, y))
            y += 30

    def draw_lender_details(self, lender, x, y):
        """Draw detailed lender information"""
        details = [
            f"Capital: ${lender.capital:,.2f}",
            f"Risk Tolerance: {lender.risk_tolerance:.2f}",
            f"Active Loans: {len(lender.loans)}",
            f"Avg Return: {np.mean([loan.expected_return() for loan in lender.loans]) if lender.loans else 0:.2%}"
        ]
        
        for i, detail in enumerate(details):
            text = self.detail_font.render(detail, True, self.colors['text'])
            self.screen.blit(text, (x + 15, y + 20 + i * 20))

    def draw_borrower_details(self, borrower, x, y):
        """Draw detailed borrower information"""
        details = [
            f"Credit Score: {borrower.credit_score}",
            f"Income: ${borrower.income:,.2f}",
            f"Debt: ${borrower.debt:,.2f}",
            f"DTI: {borrower.debt_to_income_ratio():.2%}"
        ]
        
        for i, detail in enumerate(details):
            text = self.detail_font.render(detail, True, self.colors['text'])
            self.screen.blit(text, (x + 15, y + 20 + i * 20))

    def handle_events(self, env):
        """Handle mouse events for interaction"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_clicked = True
        
        # Check for hover/click on agents
        self.hover_agent = None
        for lender_id, pos in self.lender_positions.items():
            if self.is_point_near(mouse_pos, pos, 15):
                self.hover_agent = ('lender', lender_id)
                if mouse_clicked:
                    self.selected_agent = ('lender', lender_id)
                    
        for borrower_id, pos in self.borrower_positions.items():
            if self.is_point_near(mouse_pos, pos, 15):
                self.hover_agent = ('borrower', borrower_id)
                if mouse_clicked:
                    self.selected_agent = ('borrower', borrower_id)
        
        return False

    def is_point_near(self, point, target, threshold):
        """Check if a point is near a target position"""
        return ((point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2) <= threshold ** 2

    def update(self, env):
        """Update visualization"""
        # Handle events
        quit_requested = self.handle_events(env)
        if quit_requested:
            return True
            
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw background grid
        self.draw_grid()
        
        # Initialize positions if not set
        if not self.lender_positions:
            self.lender_positions = {
                lender.id: (random.randint(50, self.width - 350), 
                          random.randint(50, self.height // 2 - 50)) 
                for lender in env.lenders
            }
        if not self.borrower_positions:
            self.borrower_positions = {
                borrower.id: (random.randint(50, self.width - 350), 
                            random.randint(self.height // 2 + 50, self.height - 50)) 
                for borrower in env.borrowers
            }
        
        # Draw rejected loans first (background)
        for loan in env.rejected_loans:
            self.draw_loan(loan, is_rejected=True)
            
        # Draw active loans
        for loan in env.loans:
            self.draw_loan(loan, is_rejected=False)
        
        # Draw agents
        for lender in env.lenders:
            self.draw_lender(lender, *self.lender_positions[lender.id])
        for borrower in env.borrowers:
            self.draw_borrower(borrower, *self.borrower_positions[borrower.id])
        
        # Draw statistics panel
        self.draw_statistics_panel(env)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
        
        return False

    def close(self):
        """Clean up pygame"""
        pygame.quit()
