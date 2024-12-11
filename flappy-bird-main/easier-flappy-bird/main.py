import matplotlib.pyplot as plt
from game import FlappyBirdEnv
from agent import QLearningAgent
import pygame

def main():
    # Initialize environment and agent
    env = FlappyBirdEnv()
    agent = QLearningAgent()
    
    # Training history
    generations = []
    scores = []
    
    # Show welcome screen
    pygame.display.set_caption("Flappy Bird Q-Learning")
    show_welcome_screen(env)
    
    # Training loop
    generation = 1
    while True:
        state = env.reset()
        done = False
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done = env.step(action, generation)
            
            # Update agent
            agent.update(state, action, reward, next_state)
            
            state = next_state
            
        # Record training history
        generations.append(generation)
        scores.append(env.score)
        
        # Print progress
        print(f"Generation {generation}: Score {env.score}")
        
        generation += 1

def show_welcome_screen(env):
    """Show welcome screen until space is pressed"""
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                waiting = False
        
        # Draw welcome screen
        env.window.blit(env.images['background'], (0, 0))
        env.window.blit(env.images['bird'], (int(env.SCREEN_WIDTH/5), int(env.SCREEN_HEIGHT/2)))
        env.window.blit(env.images['base'], (0, env.BASE_Y))
        
        # Draw text
        title = env.font.render("Click space to start", 1, (255, 255, 255))
        env.window.blit(title, (env.SCREEN_WIDTH/2 - title.get_width()/2, env.SCREEN_HEIGHT/3))
        
        pygame.display.update()
        env.clock.tick(env.FPS)

if __name__ == "__main__":
    main()