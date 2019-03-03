# baselineTeam.py
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.values = util.Counter()
    self.epsilon = .4
    self.alpha = .3
    self.discount = .9

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    # for this part, state is the observation of last state
    # and nextState is the current state
    old_state = self.getPreviousObservation()
    if old_state:
        max_new_q = self.computeValueFromQValues(gameState)
        old_pos = old_state.getAgentState(self.index).getPosition()
        cur_pos = gameState.getAgentState(self.index).getPosition()
        action = (cur_pos[0]-old_pos[0], cur_pos[1]-old_pos[1])
        old_value = self.getQValue(old_state, action)
        reward = self.getScore(gameState) - self.getScore(old_state)
        self.values[(old_pos, action)] = (1 - self.alpha) * old_value\
                                       + self.alpha * (reward + \
                                       self.discount * max_new_q)
    return self.getQAction(gameState)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getQValue(self, state, action):
      pos = state.getAgentState(self.index).getPosition()
      return float(self.values[(pos, action)])

  def computeValueFromQValues(self, state):
      actions = state.getLegalActions(self.index)
      if not actions:
          return 0.0
      best_action, best_reward = '', -1e9
      for action in actions:
          reward = self.getQValue(state, action)
          if reward > best_reward:
              best_reward = reward
              best_action = action
      return best_reward

  def computeActionFromQValues(self, state):
      actions = state.getLegalActions(self.index)
      if not actions:
          return None
      best_action, best_reward = '', -1e9
      for action in actions:
          reward = self.getQValue(state, action)
          if reward > best_reward:
              best_reward = reward
              best_action = action
      return best_action
  
  def getQAction(self, state):
      legalActions = state.getLegalActions(self.index)
      action = None
      if util.flipCoin(self.epsilon):
           action = random.choice(legalActions)
      else:
          action = self.computeActionFromQValues(state)
      return action

  def getPolicy(self, state):
      return state.computeActionFromQValues(state)

  def getValue(self, state):
      return self.computeValueFromQValues(state)

PENDING_FOOD = 0

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    # add more features for evaluating
    # initialize feature list
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foods = self.getFood(successor)
    grid_size = foods.height
    foodList = foods.asList()
    features['successorScore'] = -len(foodList)
    # calculate score change
    # retrieve the last state
    global PENDING_FOOD
    old_game_state = self.getPreviousObservation()
    if PENDING_FOOD <= 3 and old_game_state:
        old_score = -len(old_game_state.getBlueFood().asList() if gameState.isOnRedTeam\
                         else old_game_state.getRedFood().asList())
        PENDING_FOOD += max(features['successorScore'] - old_score, 0)
    else:
        PENDING_FOOD = 0
    # Compute distance to the nearest ghost
    opponents = gameState.getBlueTeamIndices()\
               if gameState.isOnRedTeam\
               else gameState.getRedTeamIndices()
    
    my_pos = successor.getAgentState(self.index).getPosition()
    tmp_pos = [successor.getAgentState(3).getPosition() for opponent in opponents]
    min_dis_to_oppo = min([self.getMazeDistance(my_pos, oppo) for oppo in tmp_pos])
    features['distance_to_opponent'] = 1 / (min_dis_to_oppo + 1)
    
    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      my_pos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(my_pos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.8 / (PENDING_FOOD * 0.5 + 1), 'distanceToFood': -0.5,\
            'distance_to_opponent': -5 * (1 + PENDING_FOOD)}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
