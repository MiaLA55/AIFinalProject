import pygame
from pygame.locals import *
import sys
import random

class FlappyBirdEnv:
    def __init__(self):
        # Constants
        self.SCREEN_WIDTH = 280
        self.SCREEN_HEIGHT = 511
        self.BASE_Y = self.SCREEN_HEIGHT * 0.8
        self.FPS = 32
        
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("comicsans", 30)
        
        # Load images
        self.images = {
            'base': pygame.image.load('base.png').convert_alpha(),
            'pipe': (
                pygame.transform.rotate(pygame.image.load('pipe.png').convert_alpha(), 180),
                pygame.image.load('pipe.png').convert_alpha()
            ),
            'background': pygame.image.load('bg.png').convert(),
            'bird': pygame.image.load('bird1.png').convert_alpha()
        }
        
        # Game state
        self.reset()

    def reset(self):
        """Reset the game state for a new episode"""
        self.score = 0
        self.bird_pos = [int(self.SCREEN_WIDTH/5), int(self.SCREEN_HEIGHT/2)]
        self.bird_velocity = -9
        self.bird_flapped = False
        
        # Initialize pipes
        self.pipe1 = self._get_new_pipe()
        self.pipe2 = self._get_new_pipe()
        self.upper_pipes = [
            {'x': self.SCREEN_WIDTH + 200, 'y': self.pipe1[0]['y']},
            {'x': self.SCREEN_WIDTH + 500, 'y': self.pipe2[0]['y']}
        ]
        self.lower_pipes = [
            {'x': self.SCREEN_WIDTH + 200, 'y': self.pipe1[1]['y']},
            {'x': self.SCREEN_WIDTH + 500, 'y': self.pipe2[1]['y']}
        ]
        
        return self._get_state()

    def _get_new_pipe(self):
        """Generate a new pipe position"""
        pipe_height = self.images['pipe'][1].get_height()
        gap = int(self.SCREEN_HEIGHT/4)
        y2 = int(gap + random.randrange(0, int(self.SCREEN_HEIGHT - self.images['base'].get_height() - 1.2*gap)))
        pipe_x = int(self.SCREEN_WIDTH + 300)
        y1 = int(pipe_height - y2 + gap)
        
        return [
            {'x': pipe_x, 'y': -y1},
            {'x': pipe_x, 'y': y2}
        ]

    def _check_collision(self):
        """Check if the bird has collided with pipes or boundaries"""
        if (self.bird_pos[1] >= self.BASE_Y - self.images['bird'].get_height() or 
            self.bird_pos[1] < 0):
            return True
            
        for pipe in self.upper_pipes:
            pipe_height = self.images['pipe'][0].get_height()
            if (self.bird_pos[1] < pipe_height + pipe['y'] and 
                abs(self.bird_pos[0] - pipe['x']) < self.images['pipe'][0].get_width()):
                return True

        for pipe in self.lower_pipes:
            if (self.bird_pos[1] + self.images['bird'].get_height() > pipe['y'] and 
                abs(self.bird_pos[0] - pipe['x']) < self.images['pipe'][0].get_width()):
                return True
                
        return False

    def step(self, action, generation):
        """Execute one time step within the environment"""
        reward = 15
        done = False
        
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Apply action
        if action:
            if self.bird_pos[1] > 0:
                self.bird_velocity = -8
                self.bird_flapped = True

        # Update bird position
        if self.bird_velocity < 10 and not self.bird_flapped:
            self.bird_velocity += 1
        if self.bird_flapped:
            self.bird_flapped = False
            
        self.bird_pos[1] = self.bird_pos[1] + min(
            self.bird_velocity, 
            self.BASE_Y - self.bird_pos[1] - self.images['bird'].get_height()
        )

        # Update pipes
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            upper_pipe['x'] += -4
            lower_pipe['x'] += -4

        # Add new pipe when first pipe is about to go off screen
        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = self._get_new_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # Remove pipes that are off screen
        if self.upper_pipes[0]['x'] < -self.images['pipe'][0].get_width():
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # Check for point scoring
        player_mid_pos = self.bird_pos[0] + self.images['bird'].get_width()/2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe['x'] + self.images['pipe'][0].get_width()/2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self.score += 1

        # Check for collision
        if self._check_collision():
            reward = -1000
            done = True

        # Render if it's a visualization generation
        if generation % 10 == 0:
            self.render(generation)
            self.clock.tick(self.FPS)
        
        return self._get_state(), reward, done

    def _get_state(self):
        """Get the current state for the agent"""
        x = min(280, self.lower_pipes[0]['x'])
        y = self.lower_pipes[0]['y'] - self.bird_pos[1]
        if y < 0:
            y = abs(y) + 408
        return int(x/40-1), int(y/40)

    def render(self, generation):
        """Render the current game state"""
        # Draw background
        self.window.blit(self.images['background'], (0, 0))
        
        # Draw pipes
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            self.window.blit(self.images['pipe'][0], (upper_pipe['x'], upper_pipe['y']))
            self.window.blit(self.images['pipe'][1], (lower_pipe['x'], lower_pipe['y']))
        
        # Draw base
        self.window.blit(self.images['base'], (0, self.BASE_Y))
        
        # Draw bird
        self.window.blit(self.images['bird'], (self.bird_pos[0], self.bird_pos[1]))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", 1, (255, 255, 255))
        self.window.blit(score_text, (self.SCREEN_WIDTH - 10 - score_text.get_width(), 10))
        
        pygame.display.update()