import numpy as np
from collections import deque
from SnakeGame import SnakeGame
import random

import matplotlib.pyplot as plot
from IPython import display




def plot(scores, mean_scores):
    #This method is for plotting the training data to see if the agent is doing better
    display.clear_output(wait=True)
    display.display(plot.gcf())

    plot.clf()
    plot.xlabel("n Games")
    plot.ylabel("Score")
    plot.title("The Effect of Number of Games on Score")
    plot.plot(mean_scores)
    plot.plot(scores)

    plot.text(len(scores)-1, scores[-1], str(scores[-1]))
    plot.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plot.ylim(ymin = 0)
    plot.show(block=False)
    plot.pause(0.09)


plot.ion()