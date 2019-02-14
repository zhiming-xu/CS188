# myAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

opt= []
target = []

class MyAgent(Agent):
    """
    Implementation of your agent.
    """
    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        global opt, target
        startPosition = state.getPacmanPosition(self.index)
        food = state.getFood()
        problem = AnyFoodSearchProblem(state, self.index)
        index = self.index
        target_i, target_j = target[index]
        "*** YOUR CODE HERE ***"
        flag = 0
        if food[target_i][target_j] is False or\
           (target_i, target_j)== (-1, -1) or len(opt[index])<=1:
            for i in range(food.width):
                for j in range(food.height):
                    if food[i][j]:
                        path = search.bfs(problem)
                        opt[index] = path
                        target[index] = (i, j)
                        cost = len(path)
                        count = 0
                        for i in range(len(target)):
                            if target[i] == [i, j]:
                                count += 1
                        flag = (count<=1)
                        if flag:
                            break
                if flag:
                    break
        else:
            del(opt[index][0])
        return opt[index][0]

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        global opt, target
        opt=[[1,] for i in range(10)]
        target=[[-1, -1] for i in range(10)]
"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""


class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"
       	cost = 1000000000
        global opt
        flag = 0
        if len(opt[self.index]) <= 1:
            for i in range(food.width):
                for j in range(food.height):
                    if food[i][j]:
                        path = search.bfs(problem)
                        if len(path)<cost:
                            opt[self.index] = path
                            cost = len(path)
                            if self.index!=0:
                                flag=1
                                break
                if flag:
                    break
        else:
            del(opt[self.index][0])
        return opt[self.index]

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y]

