# myAgentP3.py
# ---------
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
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util, sys, heapq, math
from game import Directions, Actions
from util import nearestPoint


def breadth_first_search(pos, target, walls):
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
        neighbours = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in neighbours:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no path found
    return None


def actionsWithoutReverse(legalActions, gameState, agentIndex):
    """
    Filters actions by removing REVERSE, i.e. the opposite action to the previous one
    """
    legalActions = list(legalActions)
    reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
    if len(legalActions) > 1 and reverse in legalActions:
        legalActions.remove(reverse)
    return legalActions

##########
# Agents #
##########
class MyAgent(CaptureAgent):
    """
    YOUR DESCRIPTION HERE
    Re-planning agent that attempts to find a desirable sequence of actions
    for the next 20 steps at each time step.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        # Make sure you do not delete the following line.
        # If you would like to use Manhattan distances instead
        # of maze distances in order to save on initialization
        # time, please take a look at:
        # CaptureAgent.registerInitialState in captureAgents.py.
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.weights = util.Counter()
         # for offensive
        weights = util.Counter()
        weights['min_distance_to_food'] = -7.2
        weights['eat_food'] = 4.2
        weights['is-ghosts-1-step-away'] = -4.3
        weights['is-ghosts-2-step-away'] = -5
        weights['closest_distance_to_ghost'] = -1.0
        
        weights['food_nearby'] = 1.2
        # weights['is_dead_end'] = -1.1
        # weights['is_tunnel'] = -1.6
        # weights['is_crossing'] = .8
        # weights['is_open_area'] = 1.7
        weights['bias'] = 1
        self.weights = weights
        team_index = self.getTeam(gameState)
        team_index.remove(self.index)
        self.walls = gameState.getWalls()
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
    
    def closestFood(self, pos, food, walls):
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

    def closestDistance(self, pos, target, walls):
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

    def chooseAction(self, gameState):
        if self.getPreviousObservation() is not None:
            self.update_value(gameState)
        self.pre_action.append(self.getQAction(gameState))
        actionsWithoutReverse(self.pre_action, gameState, self.index)
        return self.pre_action[-1]

    def update_value(self, state):
        pre_features = self.pre_features
        pre_value = self.pre_value
        diff = self.getReward(state) + self.discount * self.computeValueFromQValues(state)\
               - pre_value
        # print('WEIGHT BEFORE:', self.weights)
        change, change_sum = 0, 0
        for feature in pre_features:
            change = self.alpha * diff * pre_features[feature]
            self.weights[feature] += change
            change_sum += abs(change)
        if change_sum > .1:
            print('WEIGHTS: ', self.weights,'VALUES: ', pre_features)
        # print('WEIGHT AFTER:', self.weights)

    def getReward(self, state):
        pre_state = self.getPreviousObservation()
        # score bonus
        score_bonus = 10 * (state.getScore() - pre_state.getScore()) * self.pre_features['eat_food']
        # food bonus
        pre_agent_state = pre_state.getAgentState(self.index)
        cur_agent_state = state.getAgentState(self.index)
        pre_pos = pre_agent_state.getPosition()
        cur_pos = cur_agent_state.getPosition()
        # death penalty
        if abs(cur_pos[0] - pre_pos[0]) + abs(cur_pos[1] - pre_pos[1]) > 3:
            score_bonus -= 1
        # stop penalty
        stop_penalty = self.pre_action[-1] == Directions.STOP
        wandering_penalty =\
            self.pre_action[-1] == Actions.reverseDirection(self.pre_action[-2])\
                                   if len(self.pre_action) > 1 else 0
        reward = score_bonus - stop_penalty - wandering_penalty
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
  
    def getFeatures(self, gameState, action):
        # Initiate
        features = util.Counter()
        next_state = self.getSuccessor(gameState, action)
        food_to_eat = self.getFood(gameState)
        opponent_index = self.getOpponents(gameState)
        walls = next_state.getWalls()
        features['bias'] = 1.0
        new_agent_state = next_state.getAgentState(self.index)
        next_x, next_y = new_agent_state.getPosition()
        min_distance = self.closestFood((int(next_x), int(next_y)), food_to_eat, walls)
        if min_distance is not None:
            features['min_distance_to_food'] = min_distance /\
                     min(walls.width, walls.height) * 1.5
        one_step_away = Actions.getLegalNeighbors((next_x, next_y), walls)
        two_step_away = []
        for oppo in one_step_away:
            two_step_away += Actions.getLegalNeighbors(oppo, walls)
        for pos in two_step_away:
            if food_to_eat[pos[0]][pos[1]]:
                features['food_nearby'] = 1
                break
        distances = []
        opponent_pos = []
        for opponent in opponent_index:
            opponent_pos.append(gameState.getAgentPosition(opponent))
            distances.append(self.getMazeDistance((next_x, next_y), opponent_pos[-1]))
        if distances:
            features['closest_distance_to_ghost'] = min(distances) /\
                                    min(walls.width, walls.height)
        features['is_dead_end'] = self.is_dead_end[(next_x, next_y)]
        features['is_tunnel'] = self.is_tunnel[(next_x, next_y)]
        features['is_crossing'] = self.is_crossing[(next_x, next_y)]
        features['is_open_area'] = self.is_open_area[(next_x, next_y)]
        # is ghosts 1 or 2 step away
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
            # eat food after taking action
            if not is_one_step_away:
                features['eat_food'] = food_to_eat[int(next_x)][int(next_y)]
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