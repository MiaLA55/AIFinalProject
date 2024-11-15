import pygame, sys, time, random

# Difficulty settings
difficulty = 25

# Window size
frame_size_x = 720
frame_size_y = 480

# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

class SnakeGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Snake Eater')
        self.game_window = pygame.display.set_mode((frame_size_x, frame_size_y))
        self.fps_controller = pygame.time.Clock()

        self.reset()

    def reset(self):
        # Initialize game variables
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = [random.randrange(1, (frame_size_x//10)) * 10,
                         random.randrange(1, (frame_size_y//10)) * 10]
        self.food_spawn = True

        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0

    def show_score(self, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (frame_size_x / 10, 15)
        self.game_window.blit(score_surface, score_rect)

    def game_over(self):
        my_font = pygame.font.SysFont('times new roman', 90)
        game_over_surface = my_font.render('YOU DIED', True, red)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (frame_size_x / 2, frame_size_y / 4)
        self.game_window.fill(black)
        self.game_window.blit(game_over_surface, game_over_rect)
        self.show_score(red, 'times', 20)
        pygame.display.flip()
        time.sleep(2)
        self.reset()

    def play_step(self):
        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == ord('w'):
                    self.change_to = 'UP'
                elif event.key == pygame.K_DOWN or event.key == ord('s'):
                    self.change_to = 'DOWN'
                elif event.key == pygame.K_LEFT or event.key == ord('a'):
                    self.change_to = 'LEFT'
                elif event.key == pygame.K_RIGHT or event.key == ord('d'):
                    self.change_to = 'RIGHT'
                elif event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))

        # Move snake in the specified direction
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        elif self.direction == 'DOWN':
            self.snake_pos[1] += 10
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += 10

        # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos == self.food_pos:
            self.score += 1
            self.food_spawn = False
        else:
            self.snake_body.pop()

        # Spawning new food
        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (frame_size_x // 10)) * 10,
                             random.randrange(1, (frame_size_y // 10)) * 10]
        self.food_spawn = True

        # Refresh game window
        self.game_window.fill(black)
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.game_window, white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        self.show_score(white, 'consolas', 20)
        pygame.display.update()
        self.fps_controller.tick(difficulty)

        # Check for collisions
        if self.snake_pos[0] < 0 or self.snake_pos[0] > frame_size_x - 10 or \
           self.snake_pos[1] < 0 or self.snake_pos[1] > frame_size_y - 10:
            self.game_over()
        for block in self.snake_body[1:]:
            if self.snake_pos == block:
                self.game_over()

# Main function to run the game
if __name__ == "__main__":
    game = SnakeGame()
    while True:
        game.play_step()
