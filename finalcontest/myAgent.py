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
# Global #
##########
dist_map = util.Counter()


##########
# Helper #
##########
class Deque:
    def __init__(self):
        self.list = []

    def push(self, item):
        self.list.insert(0, item)

    def pop(self):
        return self.list.pop()

    def pop_back(self):
        return self.list.pop(0)

    def peek_item(self):
        return self.list[0]

    def is_empty(self):
        return len(self.list) == 0


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def peek_priority(self):
        return min(self.heap)


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


def closest_food(pos, food):
    if food is None:
        return None
    food_list = food.asList()
    if len(food_list) == 0:
        return None
    min_distance = float("inf")
    for foodLoc in food_list:
        min_distance = min(min_distance, dist_map[pos][foodLoc])
    return min_distance


def closest_distance(pos1, pos2):
    return dist_map[pos1][pos2]

def elegant_search(cur_state):
    walls = cur_state.getWalls()
    fringe = Deque()
    closed = set()
    pos = cur_state.getAgentPosition(0)
    fringe.push(pos)
    for x in range(walls.width):
        for y in range(walls.height):
            dist_map[(x, y)] = util.Counter()
    while fringe.is_empty() is False:
        cur_pos = fringe.pop()
        cur_oppo_pos = (walls.width - cur_pos[0] - 1, walls.height - cur_pos[1] - 1)
        closed.add(cur_pos)

        # Calculate distances within half of the map
        neighbours = Actions.getLegalNeighbors(cur_pos, walls)
        for neighbour in neighbours:
            if neighbour not in closed and neighbour[0] < round(walls.width / 2):
                fringe.push(neighbour)
        positions = set()
        for pos_x in range(round(walls.width / 2)):
            for pos_y in range(walls.height):
                if walls[pos_x][pos_y] == 0:
                    positions.add((pos_x, pos_y))
        for position in positions:
            if position not in closed:
                oppo_position = (walls.width - position[0] - 1, walls.height - position[1] - 1)
                dist_map[cur_pos][position] = dist_map[position][cur_pos] = \
                    dist_map[cur_oppo_pos][oppo_position] = dist_map[oppo_position][cur_oppo_pos] = \
                    breadth_first_search(cur_pos, position, walls)
    closed.clear()

    # Calculate distances between points that are in different colors
    x = round(walls.width / 2) - 1
    for y in range(walls.height):
        if walls[x][y] != 0:
            continue
        fringe.push(pos)
        while fringe.is_empty() is False:
            cur_pos = fringe.pop()
            closed.add(cur_pos)
            neighbours = Actions.getLegalNeighbors(cur_pos, walls)
            for neighbour in neighbours:
                if neighbour not in closed:
                    fringe.push(neighbour)
            positions = set()
            for pos_x in range(round(walls.width / 2), walls.width):
                for pos_y in range(walls.height):
                    if walls[pos_x][pos_y] == 0:
                        positions.add((pos_x, pos_y))
            for position in positions:
                if position not in closed:
                    distance = dist_map[cur_pos][(x, y)] + dist_map[(x + 1, y)][position] + 1
                    if dist_map[cur_pos][position] == 0 or distance < dist_map[cur_pos][position]:
                        dist_map[cur_pos][position] = dist_map[position][cur_pos] = distance
    closed.clear()

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
        self.weights['closest_food'] = -7.2
        self.weights['eat_food'] = 4.6
        self.weights['distance_to_ghost'] = -.6
        self.weights['eaten_by_ghost'] = -4
        self.weights['is_ghost_1_step_away'] = -4.3
        self.weights['is_ghost_2_step_away'] = -5
        self.weights['food_nearby'] = 1.2
        self.weights['is_dead_end'] = -.8
        self.weights['is_tunnel'] = -.4
        self.weights['is_crossing'] = .4
        self.weights['is_open_area'] = .8
        self.weights['bias'] = 1.2
        self.is_dead_end = util.Counter()
        self.is_tunnel = util.Counter()
        self.is_crossing = util.Counter()
        self.is_open_area = util.Counter()
        self.food_num = len(self.getFood(gameState).asList())
        self.team_index = self.getTeam(gameState)
        self.team_index.remove(self.index)
        self.team_index = self.team_index[0]
        self.ghost_index = self.getOpponents(gameState)[0]
        self.depth = 5
        # THIS IS A VERY ELEGANT SEARCH #
        elegant_search(gameState)
        # self.pre_calculate(gameState)

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
            for action in self.receivedBroadcast:
                if action in gameState.getLegalActions(self.team_index):
                    gameState = gameState.generateSuccessor(self.team_index, action)
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
                tmp_dis = closest_distance(neighbor, pos)
                if tmp_dis < ghost_min_dis:
                    ghost_min_dis = tmp_dis
                    ghost_min_pos = neighbor
        ghost_next_pos = ghost_min_pos
        # calculate features
        # eat food or not, if so, remove the food
        features['eaten_by_ghost'] = my_pos == ghost_next_pos
        closest_dis_to_food = closest_food(my_pos, food)
        if closest_dis_to_food:
            features['closest_food'] = closest_dis_to_food * 1.5 / min(walls.width, walls.height)
        features['eat_food'] = gameState.getScore()!=self.pre_score
        # print(gameState.getScore(), self.pre_score)
        # closest distance to ghost (current ghost position)
        closest_dis_to_ghost = closest_distance(my_pos, ghost_next_pos)
        features['distance_to_ghost'] = closest_dis_to_ghost / max(walls.width, walls.height)
        # if ghost is one or two steps away
        features['is_ghost_1_step_away'] = closest_dis_to_ghost <= 1
        features['is_ghost_2_step_away'] = closest_dis_to_ghost <= 2
        # features['is_dead_end'] = self.is_dead_end[(int(my_pos[0]), int(my_pos[1]))]
        # features['is_tunnel'] = self.is_tunnel[(int(my_pos[0]), int(my_pos[1]))]
        # features['is_crossing'] = self.is_crossing[(int(my_pos[0]), int(my_pos[1]))]
        # features['is_open_area'] = self.is_open_area[(int(my_pos[0]), int(my_pos[1]))]
        features['bias'] = 1
        features.divideAll(10.0)
        qvalue = features * self.weights + (gameState.getScore() - self.pre_score) * 10
        return qvalue