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
import numpy as np

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

def closestDistance(pos, target, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find target at this location then exit
        if pos_x == target[0] and pos_y == target[1]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no path found
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
    weights['oppo_in_dead_end'] = 2.3
    weights['oppo_in_tunnel'] = 1.4
    weights['oppo_in_crossing'] = -4
    weights['oppo_in_open_area'] = -2.2
    weights['closest_distance_to_pacman'] = -4.2
    weights['average_distance_to_pacman'] = -5.4
    weights['is-pacman-1-step-away'] = 3.4
    weights['is-pacman-2-step-away'] = 2.8
    weights['eat_pacman'] = 6.4
    weights['opponent_closest_distance_to_food'] = -1.2
    # for offensive
    weights['min_distance_to_food'] = -6.7
    weights['carry_food'] = 3.4
    weights['eat_food'] = 4
    weights['return_distance'] = -5
    weights['is-ghosts-1-step-away'] = -4.3
    weights['is-ghosts-2-step-away'] = -5
    weights['is_pacman'] = 3.2
    weights['food_nearby'] = 1
    weights['closest_distance_to_ghost'] = -1
    # for both
    weights['is_dead_end'] = -1.1
    weights['is_tunnel'] = -2
    weights['is_crossing'] = 1
    weights['is_open_area'] = 2.0
    weights['bias'] = 1
    weights['is_stop'] = -3.7
    weights['is_wandering'] = -4.2
    self.weights = weights
    self.epsilon = .1
    self.alpha = .2
    self.discount = .95
    self.pre_action = []
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
    self.pre_action.append(\
                    self.alphabetaSearch(gameState, self.index,\
                                         2, -1e9, 1e9)[1])
    return self.pre_action[-1]

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
        pre_agent_state = pre_state.getAgentState(self.index)
        cur_agent_state = state.getAgentState(self.index)
        pre_pos = pre_agent_state.getPosition()
        cur_pos = cur_agent_state.getPosition()
        pre_food = pre_state.getBlueFood() if pre_state.isOnRedTeam(self.index)\
                   else pre_state.getRedFood()
        cur_food = state.getBlueFood() if state.isOnRedTeam(self.index)\
                   else state.getRedFood()
        food_bonus = 2 * (len(pre_food.asList()) - len(cur_food.asList()))
        if food_bonus > 0:
            self.carry_food += 1
        elif pre_agent_state.isPacman ^ cur_agent_state.isPacman:
            self.carry_food = 0
        food_bonus /= np.sqrt(self.carry_food + 1)
        # food loss
        pre_food = pre_state.getRedFood() if pre_state.isOnRedTeam(self.index)\
                   else pre_state.getBlueFood()
        cur_food = state.getRedFood() if state.isOnRedTeam(self.index)\
                   else state.getBlueFood()
        food_loss = 2 * (len(pre_food.asList()) - len(cur_food.asList()))
        # death penalty
        if abs(cur_pos[0] - pre_pos[0]) + abs(cur_pos[1] - pre_pos[1]) > 3:
            score_bonus -= 1
        # stop penalty
        stop_penalty = self.pre_action[-1] == Directions.STOP
        wandering_penalty =\
            self.pre_action[-1] == Actions.reverseDirection(self.pre_action[-2])\
                                   if len(self.pre_action) > 1 else 0
        # eat opponents
        oppo_index = self.getOpponents(state)
        old_pos = [pre_state.getAgentPosition(oppo) for oppo in oppo_index]
        new_pos = [state.getAgentPosition(oppo) for oppo in oppo_index]
        num_oppo_eat = 0
        for i in range(len(old_pos)):
            if abs(old_pos[i][0]-new_pos[i][0]) + abs(old_pos[i][1]-new_pos[i][1]) > 3:
                num_oppo_eat += 1
        oppo_bonus = num_oppo_eat * 1
        reward = food_bonus + score_bonus + oppo_bonus - food_loss - stop_penalty -\
                 wandering_penalty
        return reward

  def alphabetaSearch(self, gameState, agentIndex, depth, alpha, beta):
      if depth == 0 or gameState.isOver():
          ret = self.computeValueFromQValues(gameState), Directions.STOP
      elif agentIndex % 2 == self.index % 2:
          ret = self.alphasearch(gameState, agentIndex, depth, alpha, beta)
      else:
          ret = self.betasearch(gameState, agentIndex, depth, alpha, beta)
      return ret

  def alphasearch(self, gameState, agentIndex, depth, alpha, beta):
      actions = gameState.getLegalActions(agentIndex)
      if agentIndex == gameState.getNumAgents() - 1:
          next_agent, next_depth = 0, depth - 1
      else:
          next_agent, next_depth = agentIndex + 1, depth
      max_score, max_action = -1e9, Directions.STOP
      for action in actions:
          successor_game_state = gameState.generateSuccessor(agentIndex, action)
          new_score = self.alphabetaSearch(successor_game_state, next_agent, next_depth,\
                                             alpha, beta)[0]
          if new_score > max_score:
              max_score, max_action = new_score, action
          if new_score > beta:
              return new_score, action
          alpha = max(alpha, max_score)
      return max_score, max_action

  def betasearch(self, gameState, agentIndex, depth, alpha, beta):
      actions = gameState.getLegalActions(agentIndex)
      if agentIndex == gameState.getNumAgents() - 1:
          next_agent, next_depth = 0, depth - 1
      else:
          next_agent, next_depth = agentIndex + 1, depth
      min_score, min_action = 1e9, Directions.STOP
      for action in actions:
          successor_game_state = gameState.generateSuccessor(agentIndex, action)
          new_score = self.alphabetaSearch(successor_game_state, next_agent, next_depth,\
                                             alpha, beta)[0]
          if new_score < min_score:
              min_score, min_action = new_score, action
          if new_score < alpha:
              return new_score, action
          beta = min(beta, min_score)
      return min_score, min_action
  
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
      food_to_defend = self.getFoodYouAreDefending(gameState)
      opponent_index = self.getOpponents(gameState)
      walls = next_state.getWalls()
      features['bias'] = 1.0
      new_agent_state = next_state.getAgentState(self.index)
      next_x, next_y = new_agent_state.getPosition()
      # if action is stop
      features['is_stop'] = action == Directions.STOP
      features['is_wandering'] =\
              self.pre_action[-1] == Actions.reverseDirection(action)\
              if self.pre_action else 0
      # if currently carry food
      features['carry_food'] = self.carry_food
      # Offensive / distance to closest dot
      min_distance = closestFood((int(next_x), int(next_y)), food_to_eat, walls)
      if min_distance is not None:
          features['min_distance_to_food'] = min_distance /\
                   min(walls.width, walls.height) * 1.5

      # Offensive / distance to closest ghost
      distances = []
      opponent_pos = []
      if next_state.getAgentState(self.index).isPacman:
          features['is_pacman'] = 1
          for opponent in opponent_index:
              if gameState.getAgentState(opponent).isPacman is False:
                  opponent_pos.append(gameState.getAgentPosition(opponent))
                  distances.append(closestDistance((next_x, next_y),\
                                   opponent_pos[-1], walls))
          if distances:
              features['closest_distance_to_ghost'] = min(distances) /\
                                      min(walls.width, walls.height)
          features['is_dead_end'] = self.is_dead_end[(next_x, next_y)]
          features['is_tunnel'] = self.is_tunnel[(next_x, next_y)]
          features['is_crossing'] = self.is_crossing[(next_x, next_y)]
          features['is_open_area'] = self.is_open_area[(next_x, next_y)]
          # Offensive / is ghosts 1 or 2 step away
          if opponent_pos:
              one_step_away = []
              two_step_away = []
              for oppo in opponent_pos:
                  one_step_away += Actions.getLegalNeighbors(oppo, walls)
              is_one_step_away = (next_x, next_y) in one_step_away
              for oppo in one_step_away:
                  two_step_away += Actions.getLegalNeighbors(oppo, walls)
              is_two_step_away = (next_x, next_y) in two_step_away
              features['is-ghosts-1-step-away'] = is_one_step_away
              features['is-ghosts-2-step-away'] = is_two_step_away
          # Offensive / eat food after taking action
              if not is_one_step_away:
                  features['eat_food'] = food_to_eat[int(next_x)][int(next_y)]
                  features['return_distance'] = abs(next_x / (walls.width/2) - 1) *\
                  features['carry_food']
      # Defensive / distance to closest opponent: pacman
      oppo_pacman_pos = []
      for oppo in opponent_index:
          if gameState.getAgentState(oppo).isPacman:
              oppo_pacman_pos.append(gameState.getAgentPosition(oppo))
      if oppo_pacman_pos:
          distances = []
          pacman_distance_to_food = []
          for oppo_pos in oppo_pacman_pos:
              # opponent: pacman's min distance to self defending food
              pacman_distance_to_food.append(closestFood(oppo_pos,\
                                             food_to_eat, walls))
              # eat opponent in next move
              features['eat_pacman'] |= int(next_x) == int(oppo_pos[0]) and\
                                        int(next_y) == int(oppo_pos[1])
              # opponent: pacman's situation
              distances.append(closestDistance((next_x, next_y), oppo_pos, walls))
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
              for pos in two_step_away:
                  if food_to_eat[pos[0]][pos[1]]:
                      features['food_nearby'] = 1
                      break
          features['closest_distance_to_pacman'] = min(distances) /\
                                                   min(walls.width, walls.height)
          features['average_distance_to_pacman'] = sum(distances) /\
                   min(walls.width, walls.height) / len(distances)
          features['opponent_closest_distance_to_food'] =\
                   min(pacman_distance_to_food) / min(walls.width, walls.height)
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
