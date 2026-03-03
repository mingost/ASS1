import pygame
import numpy as np
from game_2048 import Game2048, Direction
from dqn_model import DQNAgent

# Colors
BACKGROUND_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
TEXT_COLOR = (119, 110, 101)
BRIGHT_TEXT_COLOR = (249, 246, 242)

class Game2048UI:
    def __init__(self, use_ai=False, model_path=None):
        pygame.init()
        self.cell_size = 100
        self.cell_padding = 10
        self.board_size = 4
        self.window_size = self.cell_size * self.board_size + self.cell_padding * (self.board_size + 1)
        self.header_height = 100
        
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + self.header_height))
        pygame.display.set_caption("2048 Game with Q-Learning")
        
        self.game = Game2048()
        self.use_ai = use_ai
        self.agent = None
        
        if use_ai and model_path:
            self.agent = DQNAgent(16, 4)
            self.agent.load(model_path)
            self.agent.epsilon = 0  # No exploration during play
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.clock = pygame.time.Clock()
        self.running = True
    
    def draw_board(self):
        """Draw the game board"""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw header
        score_text = self.font_medium.render(f"Score: {self.game.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (20, 20))
        
        max_tile_text = self.font_medium.render(f"Max: {self.game.get_max_tile()}", True, TEXT_COLOR)
        self.screen.blit(max_tile_text, (20, 60))
        
        mode_text = self.font_small.render("AI Mode" if self.use_ai else "Manual Mode", True, TEXT_COLOR)
        self.screen.blit(mode_text, (self.window_size - 120, 30))
        
        # Draw tiles
        for i in range(self.board_size):
            for j in range(self.board_size):
                value = self.game.board[i, j]
                x = j * (self.cell_size + self.cell_padding) + self.cell_padding
                y = i * (self.cell_size + self.cell_padding) + self.cell_padding + self.header_height
                
                color = TILE_COLORS.get(value, TILE_COLORS[2048])
                pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size), border_radius=5)
                
                if value != 0:
                    text_color = BRIGHT_TEXT_COLOR if value > 4 else TEXT_COLOR
                    text = self.font_large.render(str(value), True, text_color)
                    text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                    self.screen.blit(text, text_rect)
        
        if self.game.game_over:
            overlay = pygame.Surface((self.window_size, self.window_size + self.header_height))
            overlay.set_alpha(200)
            overlay.fill((255, 255, 255))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("Game Over!", True, TEXT_COLOR)
            text_rect = game_over_text.get_rect(center=(self.window_size // 2, self.window_size // 2))
            self.screen.blit(game_over_text, text_rect)
            
            restart_text = self.font_small.render("Press R to Restart or Q to Quit", True, TEXT_COLOR)
            text_rect = restart_text.get_rect(center=(self.window_size // 2, self.window_size // 2 + 50))
            self.screen.blit(restart_text, text_rect)
        
        pygame.display.flip()
    
    def handle_input(self):
        """Handle keyboard input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.game.reset()
                elif not self.game.game_over and not self.use_ai:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.game.move(Direction.UP)
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.game.move(Direction.DOWN)
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.game.move(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.game.move(Direction.RIGHT)
    
    def ai_move(self):
        """Let AI make a move"""
        if not self.game.game_over and self.agent:
            state = self.game.get_state()
            action = self.agent.act(state)
            _, _, done = self.game.move(Direction(action))
            # Update game_over status from move result
            if done:
                print(f"Game Over detected! Score: {self.game.score}, Max: {self.game.get_max_tile()}")
    
    def run(self):
        """Main game loop"""
        game_over_timer = 0
        ai_move_timer = 0
        ai_move_delay = 12  # 0.2 seconds at 60 FPS
        
        while self.running:
            self.handle_input()
            
            if self.use_ai:
                if not self.game.game_over:
                    ai_move_timer += 1
                    if ai_move_timer >= ai_move_delay:
                        self.ai_move()
                        ai_move_timer = 0
                    game_over_timer = 0
                else:
                    # Auto restart after 2 seconds
                    game_over_timer += 1
                    if game_over_timer > 120:  # 2 seconds at 60 FPS
                        print(f"Game Over! Score: {self.game.score}, Max Tile: {self.game.get_max_tile()}")
                        self.game.reset()
                        game_over_timer = 0
                        ai_move_timer = 0
            
            self.draw_board()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    import sys
    
    print("2048 Game")
    print("1. Play manually (WASD or Arrow keys)")
    print("2. Watch AI play (requires trained model)")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "2":
        ui = Game2048UI(use_ai=True, model_path="model_2048_final.pth")
    else:
        ui = Game2048UI(use_ai=False)
    
    ui.run()
