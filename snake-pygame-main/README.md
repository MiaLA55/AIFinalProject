# AI Agent Plays Snake 
We trained an agent to play snake by using Q-learning techniques.
The snake game is written in Python using the Pygame library adn the source used for the base game is here: https://github.com/rajatdiptabiswas/snake-pygame

## Running the application
There are two files that can be run. Run 
```
python SnakeGame.py
```
if you want to play the game yourself as a human.

Run 
```
python ai_snake_game.py
```
if you want to watch the AI agent continuously play the game.

## Important Files and Their General Purpose
___Note__: Other files not listed  were used purely for debugging purposes_

* __agent.py__: Contains the agent and methods for saving experiences and other data 
* __ai_snake_game__: Run this to have the AI agent play Snake on its own continuously 
* __model.py__: Contains the Q-network used by the agent
* __SnakeGame.py__: Run this to play the game yourself (without the AI agent)
* __snake_model.npy__: This file is used to store the agent's knowledge

## Screenshots

![2](https://user-images.githubusercontent.com/32998741/33873437-2780ed2a-df45-11e7-9776-b1f151fa4e02.png)
*Active game screen*

![3](https://user-images.githubusercontent.com/32998741/33873440-28647360-df45-11e7-8291-b82d5646352f.png)
*Game over screen*


