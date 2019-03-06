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
from game import Directions, Actions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='QLearningAgent', second='QLearningAgent'):
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

def closestFood(pos, food, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None
##########
# Agents #
##########

class QLearningAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    weights = util.Counter()
    # for defensive
    weights['oppo_in_dead_end'] = .6
    weights['oppo_in_tunnel'] = .8
    weights['oppo_in_crossing'] = -.6
    weights['oppo_in_open_area'] = -1
    weights['closest_distance_to_pacman'] = -.7
    weights['average_distance_to_pacman'] = -1.2
    weights['is-pacman-1-step-away'] = -.5
    weights['is-pacman-2-step-away'] = -.5
    weights['eat_pacman'] = 6
    # for offensive
    weights['closest_distance_to_ghost'] = 1.5
    weights['average_distance_to_ghost'] = 3.2
    weights['min_distance_to_food'] = -1
    weights['carry_food'] = .1
    weights['eat_food'] = 3.2
    weights['return_distance'] = -.1
    weights['is-ghosts-1-step-away'] = -2
    weights['is-ghosts-2-step-away'] = -2
    weights['is_pacman'] = 2.7
    # for both
    weights['is_dead_end'] = -.6
    weights['is_tunnel'] = -.8
    weights['is_crossing'] = 2.2
    weights['is_open_area'] = 2.2
    weights['bias'] = 1
    self.weights = weights
    self.epsilon = .2
    self.alpha = .5
    self.discount = 1
    self.pre_action = None
    self.is_dead_end = util.Counter()
    self.is_tunnel = util.Counter()
    self.is_crossing = util.Counter()
    self.is_open_area = util.Counter()
    self.pre_score = 0
    self.pre_features = util.Counter()
    self.pre_value = 0
    self.carry_food = 0
    self.pre_calculate(gameState)

  def pre_calculate(self, gameState):
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    for i in range(width):
        for j in range(height):
            count = len(Actions.getLegalNeighbors((i, j), walls))
            if count == 2:
                self.is_dead_end[(i, j)] = 1
            elif count == 3:
                self.is_tunnel[(i, j)] = 1
            elif count == 4:
                self.is_crossing[(i, j)] = 1
            else:
                self.is_open_area[(i, j)] = 1
    return

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    # for this part, state is the observation of last state
    # and nextState is the current state
    if self.getPreviousObservation() is not None:
            self.update_value(gameState)
    self.pre_action = self.getQAction(gameState)
    # print("-------------FEATURES AND WEIGHTS---------------")
    # for feature in self.pre_features:
    #    print(feature, self.pre_features[feature], self.weights[feature])
    return self.pre_action

  def update_value(self, state):
      pre_features = self.pre_features
      pre_value = self.pre_value
      diff = self.getReward(state) + self.discount * self.computeValueFromQValues(state)\
             - pre_value
      print('------------------------THIS IS DIFF---------------------------')
      print(diff)
      print('WEIGHT BEFORE:', self.weights)
      for feature in pre_features:
          self.weights[feature] += self.alpha * diff * pre_features[feature]
      print('WEIGHT AFTER:', self.weights)

  def getReward(self, state):
        pre_state = self.getPreviousObservation()
        # food_bonus = 2 * self.pre_features['eat_food']
        # oppo_bonus = 4 * self.pre_features['eat_opponent']
        # score bonus
        score_bonus = 10 * (state.getScore() - pre_state.getScore())
        score_bonus = score_bonus if state.isOnRedTeam(self.index) else -score_bonus
        # food bonus
        pre_pos = pre_state.getAgentPosition(self.index)
        cur_pos = state.getAgentPosition(self.index)
        pre_food = pre_state.getBlueFood() if pre_state.isOnRedTeam(self.index)\
                   else pre_state.getRedFood()
        cur_food = state.getBlueFood() if state.isOnRedTeam(self.index)\
                   else state.getRedFood()
        food_bonus = 2 * (len(pre_food.asList()) - len(cur_food.asList()))
        # death penalty
        if abs(cur_pos[0] - pre_pos[0]) + abs(cur_pos[1] - pre_pos[1]) > 3:
            score_bonus -= 4
        # eat opponents
        oppo_index = self.getOpponents(state)
        old_pos = [pre_state.getAgentPosition(oppo) for oppo in oppo_index]
        new_pos = [state.getAgentPosition(oppo) for oppo in oppo_index]
        num_oppo_eat = 0
        for i in range(len(old_pos)):
            if abs(old_pos[i][0]-new_pos[i][0]) + abs(old_pos[i][1]-new_pos[i][1]) > 3:
                num_oppo_eat += 1
        oppo_bonus = num_oppo_eat * 4
        reward = food_bonus + score_bonus + oppo_bonus
        return reward

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentPosition(self.index)
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  
  # for both offensive and defensive agent
  def getFeatures(self, gameState, action):
      # Initiate
      features = util.Counter()
      next_state = self.getSuccessor(gameState, action)
      food_to_eat = self.getFood(gameState)
      opponent_index = self.getOpponents(next_state)
      oppo_position = [next_state.getAgentPosition(oppo)\
                       for oppo in opponent_index]
      walls = next_state.getWalls()
      features['bias'] = 1.0
      new_agent_state = next_state.getAgentState(self.index)
      next_x, next_y = new_agent_state.getPosition()

      # Offensive / distance to closest dot
      min_distance = closestFood((int(next_x), int(next_y)), food_to_eat, walls)
      if min_distance is not None:
          features['min_distance_to_food'] = min_distance / min(walls.width, walls.height)

      # Offensive / distance to closest ghost
      distances = []
      if next_state.getAgentState(self.index).isPacman:
          features['is_pacman'] = 1
          for opponent in opponent_index:
              if next_state.getAgentState(opponent).isPacman is False:
                  opponent_pos = next_state.getAgentPosition(opponent)
                  distances.append(self.getMazeDistance((next_x, next_y), opponent_pos))
          if distances:
              features['closest_distance to ghost'] = min(distances) /\
                                                      min(walls.width, walls.height)
              features['average_distance_to_ghost'] = sum(distances) /\
                       min(walls.width, walls.height) / len(distances)
          features['is_dead_end'] = self.is_dead_end[(next_x, next_y)]
          features['is_tunnel'] = self.is_tunnel[(next_x, next_y)]
          features['is_crossing'] = self.is_crossing[(next_x, next_y)]
          features['is_open_area'] = self.is_open_area[(next_x, next_y)]
          # Offensive / is ghosts 1 or 2 step away
          one_step_away = []
          two_step_away = []
          for oppo in oppo_position:
              one_step_away += Actions.getLegalNeighbors(oppo, walls)
          is_one_step_away = (next_x, next_y) in one_step_away
          for oppo in one_step_away:
              two_step_away += Actions.getLegalNeighbors(oppo, walls)
          is_two_step_away = (next_x, next_y) in two_step_away
          features['is-ghosts-1-step-away'] = is_one_step_away
          features['is-ghosts-2-step-away'] = is_two_step_away
      # Offensive / eat food after taking action
      features['eat_food'] = food_to_eat[int(next_x)][int(next_y)]
      # Defensive / distance to closest opponent: pacman
      oppo_pacman_pos = []
      for oppo in opponent_index:
          if gameState.getAgentState(oppo).isPacman:
              oppo_pacman_pos.append(gameState.getAgentPosition(oppo))
      if oppo_pacman_pos:
          distances = []
          for oppo_pos in oppo_pacman_pos:
              # eat opponent in next move
              features['eat_pacman'] |= int(next_x) == int(oppo_pos[0]) and\
                                        int(next_y) == int(oppo_pos[1])
              # opponent: pacman's situation
              distances.append(self.getMazeDistance((next_x, next_y), oppo_pos))
              features['oppo_in_dead_end'] |= self.is_dead_end[oppo_pos]
              features['oppo_in_tunnel'] |= self.is_tunnel[oppo_pos]
              features['oppo_in_crossing'] |= self.is_crossing[oppo_pos]
              features['oppo_in_open_area'] |= self.is_open_area[oppo_pos]
              # is opponent: pacman nearby
              one_step_away = Actions.getLegalNeighbors((next_x, next_y), walls)
              is_one_step_away = oppo_pos in one_step_away
              two_step_away = []
              for oppo in one_step_away:
                  two_step_away += Actions.getLegalNeighbors(oppo, walls)
              is_two_step_away = oppo_pos in two_step_away
              features['is-pacman-1-step-away'] |= is_one_step_away
              features['is-pacman-2-step-away'] |= is_two_step_away
          features['closest_distance_to_pacman'] = min(distances) /\
                                                   min(walls.width, walls.height)
          features['average_distance_to_pacman'] = sum(distances) /\
                   min(walls.width, walls.height) / len(distances)
      features.divideAll(10.0)
      return features



























  def getQValue(self, state, action):
      features = self.getFeatures(state, action)
      weights = self.weights
      new_value = features * weights
      return new_value, features

  def computeValueFromQValues(self, state):
      actions = state.getLegalActions(self.index)
      if not actions:
          return 0.0
      best_reward = float('-inf')
      for action in actions:
          reward, features = self.getQValue(state, action)
          if reward > best_reward:
              best_reward = reward
              self.pre_features = features
      self.pre_value = best_reward
      return best_reward

  def computeActionFromQValues(self, state):
      actions = state.getLegalActions(self.index)
      if not actions:
          return None
      best_action, best_reward = '', float('-inf')
      for action in actions:
          reward, _ = self.getQValue(state, action)
          if reward > best_reward:
              best_reward = reward
              best_action = action
      return best_action
  
  def getQAction(self, state):
      legalActions = state.getLegalActions(self.index)
      if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
      else:
          action = self.computeActionFromQValues(state)
      return action
