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
import random, time, util, sys, heapq
from game import Directions, Actions
import numpy as np

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
    try:
        food_list = food.asList()
    except:
        food_list = food
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
        self.weights['closest_food'] = -2
        self.weights['eat_food'] = 4
        self.weights['distance_to_ghost'] = 1
        self.weights['eaten_by_ghost'] = -5
        self.weights['is_ghost_1_step_away'] = -3
        self.weights['is_ghost_2_step_away'] = -2

        team_index = self.getTeam(gameState)
        team_index.remove(self.index)
        ghost_index = self.getOpponents(gameState)
        walls = gameState.getWalls()
        # THIS IS A VERY ELEGANT SEARCH #
        elegant_search(gameState)
        self.register_search(self.index, team_index[0], ghost_index[0], walls)

    def chooseAction(self, gameState):
        teammateActions = []
        for i in self.receivedBroadcast:
            teammateActions.append(i)
        action = self.choose_action(gameState, teammateActions)
        return action

    def get_position_and_value(self, my_pos, team_pos, ghost_pos, food, action, team_action):
        features = util.Counter()
        # get my next position
        my_next_pos = Actions.getSuccessor(my_pos, action)
        team_next_pos = Actions.getSuccessor(team_pos, team_action)
        # get ghost's next position
        ghost_legal_action = self.get_legal_actions(ghost_pos)
        ghost_min_dis, ghost_min_pos = float('inf'), (-1, -1)
        for action in ghost_legal_action:
            tmp_pos = Actions.getSuccessor(ghost_pos, action)
            tmp_dis = closest_distance(tmp_pos, my_next_pos)
            if tmp_dis < ghost_min_dis:
                ghost_min_dis = tmp_dis
                ghost_min_pos = tmp_pos
        ghost_next_pos = ghost_min_pos
        # calculate features
        # eat food or not, if so, remove the food
        features['eaten_by_ghost'] = my_next_pos == ghost_next_pos
        features['closest_food'] = closest_food(my_next_pos, food)
        if food[int(my_next_pos[0])][int(my_next_pos[1])]:
            food[int(my_next_pos[0])][int(my_next_pos[1])] = 0
            features['eat_food'] = 1
        else:
            features['eat_food'] = 0
        # closest distance to ghost (current ghost position)
        closest_dis_to_ghost = closest_distance(my_next_pos, ghost_pos)
        features['distance_to_ghost'] = closest_dis_to_ghost
        # if ghost is one or two steps away
        features['is_ghost_1_step_away'] = closest_dis_to_ghost <= 1
        features['is_ghost_2_step_away'] = closest_dis_to_ghost <= 2
        qvalue = features * self.weights
        return my_next_pos, team_next_pos, ghost_next_pos, food, qvalue, features['eat_food']


    def register_search(self, self_index, team_index, ghost_index, walls):
        self.self_index = self_index
        self.team_index = team_index
        self.ghost_index = ghost_index
        self.walls = walls
        self.discounts = 0.9
        self.time_interval = 0.9
        self.depth = 8
        self.stats = util.Counter()
        self.stats["Root"] = MCTNodes()

    def search(self, root_key, root_node, depth, team_plan):
        cur_key = root_key
        cur_node = root_node
        path = []
        rewards = []

        # remove food eaten by teammate
        cur_team_pos = cur_node.get_team_pos()
        cur_food = cur_node.get_food()
        tmp_team_pos = cur_team_pos
        for action in team_plan:
            tmp_team_pos = Actions.getSuccessor(tmp_team_pos, action)
            if cur_food[int(tmp_team_pos[0])][int(tmp_team_pos[1])]:
                cur_food[int(tmp_team_pos[0])][int(tmp_team_pos[1])] = 0

        simulated_actions = []
        for i in range(self.depth - len(team_plan)):
            min_distance = float("inf")
            best_action = "Stop"
            for action in self.get_legal_actions(tmp_team_pos):
                new_team_pos = Actions.getSuccessor(tmp_team_pos, action)
                distance = closest_food(new_team_pos, cur_food)
                if distance < min_distance:
                    min_distance = distance
                    tmp_team_pos = new_team_pos
                    best_action = action
            simulated_actions.append(best_action)
            if cur_food[int(tmp_team_pos[0])][int(tmp_team_pos[1])]:
                cur_food[int(tmp_team_pos[0])][int(tmp_team_pos[1])] = 0
        team_plan = team_plan + simulated_actions

        for i in range(depth):
            # Extracting self_pos from current tree node to compare previous simulation and teammate's new actions
            cur_self_pos = cur_node.get_self_pos()
            legal_actions = self.get_legal_actions(cur_self_pos)
            children = []

            # Generate the children of the current node
            for action in legal_actions:
                new_key = self.make_key(cur_key, action, team_plan[i])
                # Extract remaining values from current tree node to compute q-value
                cur_team_pos = cur_node.get_team_pos()
                cur_ghost_pos = cur_node.get_ghost_pos()
                cur_food = cur_node.get_food()

                params = self.get_position_and_value(cur_self_pos, cur_team_pos, cur_ghost_pos, cur_food, action, team_plan[i])
                self.stats[new_key] = MCTNodes(*params)
                children.append((action, new_key))

            # Pick the next action
            values = np.array([self.stats[key].get_value() for _, key in children])
            values = list(np.exp(values) / np.exp(values).sum())
            prob = np.random.rand()
            cum_sum, idx = 0, 0
            for idx in range(len(values)):
                if cum_sum <= prob < cum_sum + values[idx]:
                    break
                else:
                    cum_sum += values[idx]
            best_child = children[idx]
            cur_key = best_child[1]
            cur_node = self.stats[cur_key]
            path.append(best_child[0])
            rewards.append(self.stats[cur_key].eats_pellet())
        return path, rewards, cur_key

    def choose_action(self, cur_state, team_plan):
        # Start timing
        start_time = time.time()

        # Extracting values from game state
        self_pos = cur_state.getAgentPosition(self.self_index)
        team_pos = cur_state.getAgentPosition(self.team_index)
        ghost_pos = cur_state.getAgentPosition(self.ghost_index)
        food = self.getFood(cur_state)

        self.stats = util.Counter()
        self.stats["Root"] = MCTNodes(self_pos, team_pos, ghost_pos, food, 0)

        # Main search loop
        best_path = []
        best_reward = float("-inf")
        flag = True
        while flag:
            # MCT nodes expanding process
            path, rewards, cur_key = self.search("Root", self.stats["Root"], self.depth, team_plan)

            # Compare the current best path and the newly found path
            weighted_reward = self.compute_reward(rewards)
            if best_reward < weighted_reward:
                best_path = path
                best_reward = weighted_reward
            if time.time() - start_time >= self.time_interval:
                flag = False
        return best_path[0]

    def get_legal_actions(self, pos):
        legal_actions = ["North", "South", "East", "West"]
        pos = (int(pos[0]), int(pos[1]))
        # Check if North is available
        if pos[1] == self.walls.height - 2:
            legal_actions.remove("North")
        elif self.walls[pos[0]][pos[1] + 1]:
            legal_actions.remove("North")

        # Check if South is available
        if pos[1] == 1:
            legal_actions.remove("South")
        elif self.walls[pos[0]][pos[1] - 1]:
            legal_actions.remove("South")

        # Check if East is available
        if pos[0] == self.walls.width - 2:
            legal_actions.remove("East")
        elif self.walls[pos[0] + 1][pos[1]]:
            legal_actions.remove("East")

        # Check if West is available
        if pos[0] == 1:
            legal_actions.remove("West")
        elif self.walls[pos[0] - 1][pos[1]]:
            legal_actions.remove("West")

        return legal_actions

    def make_key(self, cur_key, self_action, team_action):
        if cur_key == "Root":
            return self_action + team_action
        else:
            return (*cur_key, self_action + team_action)

    def compute_reward(self, rewards):
        sum = 0
        for i in range(len(rewards)):
            if rewards[i] == 1:
                sum += self.discounts ** i
        return sum


class MCTNodes:
    def __init__(self, self_pos=(), team_pos=(), ghost_pos=(), food=(), value=0, pellet=0):
        self.self_pos = self_pos
        self.team_pos = team_pos
        self.ghost_pos = ghost_pos
        self.food = food
        self.value = value
        self.pellet = pellet

    def __eq__(self, other):
        return self.team_pos == other[0] and self.ghost_pos == other[1]

    def get_self_pos(self):
        return self.self_pos

    def get_team_pos(self):
        return self.team_pos

    def get_ghost_pos(self):
        return self.ghost_pos

    def get_food(self):
        return self.food

    def get_value(self):
        return self.value

    def eats_pellet(self):
        return self.pellet


def actionsWithoutStop(legalActions):
    """
    Filters actions by removing the STOP action
    """
    legalActions = list(legalActions)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    return legalActions


def actionsWithoutReverse(legalActions, gameState, agentIndex):
    """
    Filters actions by removing REVERSE, i.e. the opposite action to the previous one
    """
    legalActions = list(legalActions)
    reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
    if len(legalActions) > 1 and reverse in legalActions:
        legalActions.remove(reverse)
    return legalActions
