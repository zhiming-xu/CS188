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
    weights['oppo_in_dead_end'] = .2
    weights['oppo_in_tunnel'] = .1
    weights['oppo_in_crossing'] = -.1
    weights['oppo_in_open_area'] = -.2
    weights['surrounded_x'] = .1
    weights['surrounded_y'] = .1
    weights['surrounded_both'] = .2
    weights['frontier_distance'] = -.1
    weights['is_scared'] = -.1
    weights['closest_distance_to_pacman'] = -.1
    weights['average_distance_to_pacman'] = -.1
    weights['min_distance_to_defend_food'] = -.1
    weights['dot_loss'] = -.2
    # for offensive
    weights['closest_distance_to_ghost'] = .1
    weights['average_distance_to_ghost'] = .1
    weights['is_surrounded_by_ghost'] = -.1
    weights['min_distance_to_food'] = -.1
    weights['eat_food'] = .5
    weights['carry_food'] = .1
    weights['return_distance'] = -.1
    # for both
    weights['is_dead_end'] = -.1
    weights['is_tunnel'] = -.1
    weights['is_crossing'] = .1
    weights['is_open_area'] = .2
    weights['bias'] = .1
    self.weights = weights
    self.epsilon = .1
    self.alpha = .3
    self.discount = .8
    self.pre_action = None
    self.is_dead_end = util.Counter()
    self.is_tunnel = util.Counter()
    self.is_crossing = util.Counter()
    self.is_open_area = util.Counter()
    self.pre_score = 0
    self.pre_features = util.Counter()
    self.pre_value = 0
    self.pre_calculate(gameState)

  def pre_calculate(self, gameState):
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    temp = gameState
    for i in range(width):
        for j in range(height):
            count = 0
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST,
                           Directions.WEST, Directions.STOP]:
                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(i + dx), int(j + dy)
                if 0 <= next_x < width and 0 <= next_y < height\
                and walls[next_x][next_y] is False:
                    count += 1
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
            self.update(gameState)
    self.pre_action = self.getQAction(gameState)
    return self.pre_action

  def update(self, state):
      pre_features = self.pre_features
      pre_value = self.pre_value
      diff = self.getReward(state) + self.discount * self.computeValueFromQValues(state)\
             - pre_value
      for feature in pre_features:
          self.weights[feature] += self.alpha * diff * pre_features[feature]
 
  def getReward(self, state):
        pre_state = self.getPreviousObservation()
        food_bonus = 5 * self.pre_features['eat_food']
        score_bonus = 10 * (state.getScore() - pre_state.getScore())
        pre_pos = pre_state.getAgentState(self.index).getPosition()
        cur_pos = state.getAgentState(self.index).getPosition()
        if abs(cur_pos[0] - pre_pos[0]) + abs(cur_pos[1] - pre_pos[1]) > 3:
            score_bonus -= 30
        reward = food_bonus + score_bonus
        return reward

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

# for offensive agent
  def getFeaturesPacman(self, gameState, action):
    # return a counter of features for this state
    successor = self.getSuccessor(gameState, action)
    old_state = self.getPreviousObservation()
    eat_food = self.getFood(successor)
    food_list = eat_food.asList()
    defend_food = self.getFoodYouAreDefending(successor)
    opponents_index = self.getOpponents(successor)
    walls = successor.getWalls()
    features = util.Counter()
    features['bias'] = 1.0
    # compute the location of pacman after the action
    new_state = successor.getAgentState(self.index)
    next_x, next_y = new_state.getPosition()
    # calculate distance to opponents
    oppo_position = [successor.getAgentState(oppo).getPosition() for oppo in\
                     opponents_index]
    distance_to_oppo = [self.getMazeDistance((next_x, next_y), oppo) for oppo\
                        in oppo_position]
    closest_distance = min(distance_to_oppo)
    avg_distance = sum(distance_to_oppo) / len(distance_to_oppo)
    features['closest_distance_to_ghost'] = closest_distance / 10.0
    features['average_distance_to_ghost'] = avg_distance / 10.0
    num_of_ghost = 0
    for distance_oppo in distance_to_oppo:
        if distance_oppo < 3:
            num_of_ghost += 1
    features['num_of_ghost_nearby'] = num_of_ghost / 2
    is_surrounded = oppo_position[0][0] <= next_x <= oppo_position[1][0] or\
                    oppo_position[0][1] <= next_y <= oppo_position[1][1]
    features['is_surrounded_by_ghost'] = is_surrounded
    # closest to food
    if food_list:
        min_distance = min([self.getMazeDistance((next_x, next_y), food)\
                            for food in food_list])
        features['min_distance_to_food'] = min_distance / 10.0
    get_food = eat_food[int(next_x)][int(next_y)]
    features['eat_food'] = get_food
    # carrying food
    last_role = old_state.getAgentState(self.index).isPacman
    cur_role = gameState.getAgentState(self.index).isPacman
    if last_role ^ cur_role:
        self.carry_food = 0
    elif get_food:
        self.carry_food += 1
    features['carry_food'] = self.carry_food
    # distance to frontier
    frontier_x = walls.width / 2
    return_dis = abs(frontier_x - next_x)
    features['return_distance'] = return_dis / (walls.width / 2)
    # tunnel/crossing/dead end/open
    features['is_dead_end'] = self.is_dead_end[(next_x, next_y)]
    features['is_tunnel'] = self.is_tunnel[(next_x, next_y)]
    features['is_crossing'] = self.is_crossing[(next_x, next_y)]
    features['is_open_area'] = self.is_open_area[(next_x, next_y)]
    # TODO: power capsules, scared time
    return features

# for defensive agent
  def getFeaturesGhost(self, gameState, action):
    # return a counter of feature for new state
    successor = self.getSuccessor(gameState, action)
    old_state = self.getPreviousObservation()
    eat_food = self.getFood(successor)
    defend_food = self.getFoodYouAreDefending(successor)
    food_list = defend_food.asList()
    opponents_index = self.getOpponents(successor)
    oppo_position = [gameState.getAgentState(oppo).getPosition() for oppo\
                     in opponents_index]
    walls = successor.getWalls()
    new_state = successor.getAgentState(self.index)
    next_x, next_y = new_state.getPosition()
    features = util.Counter()
    features['bias'] = 1.0
    # opponent in tunnel/crossing/openarea
    features['oppo_in_dead_end'] = self.is_dead_end[(oppo_position[0])] or\
                                   self.is_dead_end[(oppo_position[1])]
    features['oppo_in_tunnel'] = self.is_tunnel[(oppo_position[0])] or\
                                 self.is_tunnel[(oppo_position[1])]
    features['oppo_in_crossing'] = self.is_crossing[(oppo_position[0])] or\
                                   self.is_crossing[(oppo_position[1])]
    features['oppo_in_open_area'] = self.is_open_area[(oppo_position[0])] or\
                                    self.is_open_area[(oppo_position[1])]
    # opponent location relative to self
    teammate_index = (self.index + 2) % 4
    teammate_x, teammate_y = gameState.getAgentState(teammate_index).getPosition()
    for pos in oppo_position:
        surrounded_x = (min(teammate_x, next_x) <= oppo_position[0][0] <=
                       max(teammate_x, next_x) or
                       min(teammate_x, next_x) <= oppo_position[1][0] <=
                       max(teammate_x, next_x))

        surrounded_y = (min(teammate_y, next_y) <= oppo_position[0][1] <=
                       max(teammate_y, next_y) or
                       min(teammate_y, next_y) <= oppo_position[1][1] <=
                       max(teammate_y, next_y))
        surrounded_both = surrounded_x and surrounded_y
        features['surrounded_x'] = surrounded_x
        features['surrounded_y'] = surrounded_y
        features['surrounded_both'] = surrounded_both
    # distance to frontier
    frontier_x = walls.width / 2
    frontier_dis = abs(frontier_x - next_x)
    features['frontier_distance'] = frontier_dis / (walls.width / 2)
    # self in tunnel/crossing/dead end/open
    features['is_dead_end'] = self.is_dead_end[(next_x, next_y)]
    features['is_tunnel'] = self.is_tunnel[(next_x, next_y)]
    features['is_crossing'] = self.is_crossing[(next_x, next_y)]
    features['is_open_area'] = self.is_open_area[(next_x, next_y)]
    # self is scared
    features['is_scared'] = gameState.getAgentState(self.index).scaredTimer > 0
    # calculate distance to opponents
    distance_to_oppo = [self.getMazeDistance((next_x, next_y), oppo) for oppo\
                        in oppo_position]
    closest_distance = min(distance_to_oppo)
    avg_distance = sum(distance_to_oppo) / len(distance_to_oppo)
    features['closest_distance_to_pacman'] = closest_distance / 10.0
    features['average_distance_to_pacman'] = avg_distance / 10.0
    # calculate opponent distance to defending dots
    if food_list:
        min_distance = 1e9
        for oppo in oppo_position:
            min_distance = min(min_distance, min([self.getMazeDistance(oppo, food)\
                            for food in food_list]))
        features['min_distance_to_defend_food'] = min_distance / 10.0
    # get dot loss
    if old_state:
        if self.red:
            old_defending_food = old_state.getRedFood()
        else:
            old_defending_food = old_state.getBlueFood()
        dot_loss = 0
        for oppo in oppo_position:
            dot_loss += old_defending_food[int(oppo[0])][int(oppo[1])]
            features['dot_loss'] = dot_loss
    return features

  def getFeatures(self, gameState, action):
      if gameState.getAgentState(self.index).isPacman:
          self.pre_features = self.getFeaturesPacman(gameState, action)
      else:
          self.pre_features = self.getFeaturesGhost(gameState, action)
      return self.pre_features

  def getQValue(self, state, action):
      features = self.getFeatures(state, action)
      weights = self.weights
      print(weights)
      features.normalize()
      return features * weights

  def computeValueFromQValues(self, state):
      actions = state.getLegalActions(self.index)
      if not actions:
          return 0.0
      best_action, best_reward = Directions.STOP, float('-inf')
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
      best_action, best_reward = Directions.STOP, float('-inf')
      for action in actions:
          reward = self.getQValue(state, action)
          if reward > best_reward:
              best_reward = reward
              best_action = action
      if best_action == '':
          print(best_reward, reward)
      return best_action
  
  def getQAction(self, state):
      legalActions = state.getLegalActions(self.index)
      action = Directions.STOP
      if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
      else:
          action = self.computeActionFromQValues(state)
      return action
