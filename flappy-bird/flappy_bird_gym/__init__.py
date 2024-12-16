# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Registers the gym environments and exports the `gym.make` function.
"""

# Silencing pygame:
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Exporting envs:
from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple

# Exporting original game:
from flappy_bird_gym import original_game

# Exporting gym.make:
from gym import make

# Registering environments:
from gym.envs.registration import register
from flappy_bird_gym import recorder

register(
    id="FlappyBird-v0",
    entry_point="flappy_bird_gym:FlappyBirdEnvSimple",
)

register(
    id="FlappyBird-rgb-v0",
    entry_point="flappy_bird_gym:FlappyBirdEnvRGB",
)

# Main names:
__all__ = [
    make.__name__,
    FlappyBirdEnvRGB.__name__,
    FlappyBirdEnvSimple.__name__,
]