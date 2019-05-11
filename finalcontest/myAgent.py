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
        self.weights['closest_food'] = -7
        self.weights['eat_food'] = 4.5
        self.weights['eaten_by_ghost'] = -3
        self.weights['is_ghost_1_step_away'] = -4.3
        self.weights['is_ghost_2_step_away'] = -5
        self.weights['is_dead_end'] = -.1
        self.weights['is_tunnel'] = -.2
        self.weights['is_crossing'] = .3
        self.weights['is_open_area'] = .2
        self.weights['bias'] = 1.4
        self.is_dead_end = util.Counter()
        self.is_tunnel = util.Counter()
        self.is_crossing = util.Counter()
        self.is_open_area = util.Counter()
        self.team_index = self.getTeam(gameState)
        self.team_index.remove(self.index)
        self.team_index = self.team_index[0]
        self.ghost_index = self.getOpponents(gameState)[0]
        self.threshold = .6
        self.depth = 5
        self.pre_calculate(gameState)

    def closest_food(self, pos, food):
        if food is None:
            return None
        food_list = food.asList()
        if len(food_list) == 0:
            return None
        min_distance = float("inf")
        for foodLoc in food_list:
            min_distance = min(min_distance, self.getMazeDistance(pos, foodLoc))
        return min_distance
    
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
        self.food_num = len(self.getFood(gameState).asList())
        # teammate actions at this stage might be incomplete
        return self.alphabetaSearch(gameState, self.index, self.depth, float('-inf'), float('inf'))[1]

    def alphabetaSearch(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth and agentIndex == self.index:
            self.pre_score = gameState.getScore()
            if self.receivedBroadcast:
                for action in self.receivedBroadcast:
                    if action in gameState.getLegalActions(self.team_index):
                        gameState = gameState.generateSuccessor(self.team_index, action)
            else:
                for _ in range(self.depth):
                    legal_actions = gameState.getLegalActions(self.team_index)
                    new_game_state = None
                    for action in legal_actions:
                        next_state = gameState.generateSuccessor(self.team_index, action)
                        if next_state.getScore()>gameState.getScore():
                            new_game_state = next_state
                            break
                    if new_game_state:
                        gameState = new_game_state
        if depth == 0:
            ret = self.compute_value(gameState), Directions.STOP
        elif agentIndex == self.index:
            ret = self.alphasearch(gameState, agentIndex, depth, alpha, beta)
        elif agentIndex == self.ghost_index:
            ret = self.betasearch(gameState, agentIndex, depth, alpha, beta)
        return ret

    def alphasearch(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == self.index:
            next_agent, next_depth = self.ghost_index, depth - 1
        else:
            next_agent, next_depth = self.index, depth - 1
        max_score, max_action = float('-inf'), Directions.STOP
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
        if agentIndex == self.index:
            next_agent, next_depth = self.ghost_index, depth - 1
        else:
            next_agent, next_depth = self.index, depth - 1
        min_score, min_action = float('inf'), Directions.STOP
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

    def compute_value(self, gameState):
        features = util.Counter()
        # get my next position
        my_pos = gameState.getAgentPosition(self.index)
        team_pos = gameState.getAgentPosition(self.team_index)
        ghost_pos = gameState.getAgentPosition(self.ghost_index)
        walls = gameState.getWalls()
        food = self.getFood(gameState)
        # get ghost's next position
        ghost_legal_neighbors = Actions.getLegalNeighbors(ghost_pos, gameState.getWalls())
        ghost_min_dis, ghost_min_pos = float('inf'), (-1, -1)
        for pos in (my_pos, team_pos):
            for neighbor in ghost_legal_neighbors:
                tmp_dis = self.getMazeDistance(neighbor,pos)
                if tmp_dis < ghost_min_dis:
                    ghost_min_dis = tmp_dis
                    ghost_min_pos = neighbor
        ghost_next_pos = ghost_min_pos
        features['eaten_by_ghost'] = my_pos == ghost_next_pos
        closest_dis_to_food = self.closest_food(my_pos, food)
        if closest_dis_to_food:
            features['closest_food'] = closest_dis_to_food * 1.5 / min(walls.width, walls.height)
        features['eat_food'] = (gameState.getScore()!=self.pre_score)<<(len(food.asList())<=2)
        prob = random.uniform(0, 1)
        one_step_away = Actions.getLegalNeighbors(ghost_pos, walls)
        two_step_away = []
        for nbr in one_step_away:
            two_step_away += Actions.getLegalNeighbors(nbr, walls)
        features['is_ghost_1_step_away'] = my_pos in one_step_away if prob <= self.threshold else 0
        prob = random.uniform(0, 1)
        features['is_ghost_2_step_away'] = my_pos in two_step_away if prob <= self.threshold ** 2 else 0
        features['is_dead_end'] = self.is_dead_end[(int(my_pos[0]), int(my_pos[1]))]
        features['is_tunnel'] = self.is_tunnel[(int(my_pos[0]), int(my_pos[1]))]
        features['is_crossing'] = self.is_crossing[(int(my_pos[0]), int(my_pos[1]))]
        features['is_open_area'] = self.is_open_area[(int(my_pos[0]), int(my_pos[1]))]
        features['bias'] = 1
        features.divideAll(10.0)
        qvalue = features * self.weights + (gameState.getScore() - self.pre_score) * 10
        return qvalue
