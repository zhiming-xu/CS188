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
        self.weights['closest_food'] = -2.4
        self.weights['eat_food'] = 6
        self.weights['distance_to_ghost'] = .2
        self.weights['eaten_by_ghost'] = -4
        self.weights['is_ghost_1_step_away'] = -3
        self.weights['is_ghost_2_step_away'] = -2
        self.weights['food_nearby'] = 1.2
        self.weights['is_dead_end'] = -.8
        self.weights['is_tunnel'] = -.4
        self.weights['is_crossing'] = .4
        self.weights['is_open_area'] = .8
        self.weights['bias'] = 3
        self.is_dead_end = util.Counter()
        self.is_tunnel = util.Counter()
        self.is_crossing = util.Counter()
        self.is_open_area = util.Counter()
        self.food_num = len(self.getFood(gameState).asList())
        self.epsilon = .2
        self.best_path = None
        self.best_reward = 0
        self.threshold = 3
        team_index = self.getTeam(gameState)
        team_index.remove(self.index)
        ghost_index = self.getOpponents(gameState)
        walls = gameState.getWalls()
        # THIS IS A VERY ELEGANT SEARCH #
        elegant_search(gameState)
        self.register_search(self.index, team_index[0], ghost_index[0], walls)
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
        self.food_num = len(self.getFood(gameState).asList())
        teammateActions = list(self.receivedBroadcast)
        # teammate actions at this stage might be incomplete
        self.food_num = len(self.getFood(gameState).asList())
        action = self.choose_action(gameState, teammateActions)
        self.stats = util.Counter()
        return self.best_path.pop(0)

    def get_position_and_value(self, my_pos, team_pos, ghost_pos, food, action, team_action):
        features = util.Counter()
        # get my next position
        my_next_pos = Actions.getSuccessor(my_pos, action)
        # get teammate's next position
        team_next_pos = Actions.getSuccessor(team_pos, team_action)
        if team_next_pos not in Actions.getLegalNeighbors(team_pos, self.walls):
            team_next_pos = (1., 1.)
        # get ghost's next position
        ghost_legal_neighbors = Actions.getLegalNeighbors(ghost_pos, self.walls)
        ghost_min_dis, ghost_min_pos = float('inf'), (-1, -1)
        for pos in (my_next_pos, team_next_pos):
            for neighbor in ghost_legal_neighbors:
                tmp_dis = closest_distance(neighbor, pos)
                if tmp_dis < ghost_min_dis:
                    ghost_min_dis = tmp_dis
                    ghost_min_pos = neighbor
        ghost_next_pos = ghost_min_pos
        # calculate features
        # eat food or not, if so, remove the food
        features['eaten_by_ghost'] = my_next_pos == ghost_next_pos
        closest_dis_to_food = closest_food(my_next_pos, food)
        if closest_dis_to_food:
            features['closest_food'] = closest_dis_to_food * 2 / \
                                       min(self.walls.width, self.walls.height)
        if food is not None and food[int(my_next_pos[0])][int(my_next_pos[1])]:
            food[int(my_next_pos[0])][int(my_next_pos[1])] = 0
            features['eat_food'] = 1
        else:
            features['eat_food'] = 0
        # closest distance to ghost (current ghost position)
        closest_dis_to_ghost = closest_distance(my_next_pos, ghost_pos)
        # features['distance_to_ghost'] = math.log(closest_dis_to_ghost / min(self.walls.width, self.walls.height) + 1)
        # if ghost is one or two steps away
        features['is_ghost_1_step_away'] = closest_dis_to_ghost <= 1
        features['is_ghost_2_step_away'] = closest_dis_to_ghost <= 2
        features['is_dead_end'] = self.is_dead_end[(int(my_next_pos[0]), int(my_next_pos[1]))]
        features['is_tunnel'] = self.is_tunnel[(int(my_next_pos[0]), int(my_next_pos[1]))]
        features['is_crossing'] = self.is_crossing[(int(my_next_pos[0]), int(my_next_pos[1]))]
        features['is_open_area'] = self.is_open_area[(int(my_next_pos[0]), int(my_next_pos[1]))]
        features.divideAll(10.0)
        features['bias'] = 1
        qvalue = features * self.weights
        return my_next_pos, team_next_pos, ghost_next_pos, food, qvalue, features['eat_food']


    def register_search(self, self_index, team_index, ghost_index, walls):
        self.self_index = self_index
        self.team_index = team_index
        self.ghost_index = ghost_index
        self.walls = walls
        self.discounts = 0.9
        self.time_interval = 0.5
        self.depth = 15
        self.stats = util.Counter()
        self.stats["Root"] = MCTNodes()

    def search(self, root_key, root_node, depth, team_plan):
        # on first call to this function, root_key is "Root"
        cur_key = root_key
        cur_node = root_node
        path = []
        rewards = []

        cur_team_pos = cur_node.get_team_pos()
        cur_food = cur_node.get_food()
        tmp_team_pos = cur_team_pos
        # take down the foods' indices, to remove later
        removed_list = []
        for action in team_plan:
            tmp_team_pos = Actions.getSuccessor(tmp_team_pos, action)
            if tmp_team_pos in Actions.getLegalNeighbors(tmp_team_pos, self.walls) and \
               cur_food[int(tmp_team_pos[0])][int(tmp_team_pos[1])]:
                removed_list.append([int(tmp_team_pos[0]), int(tmp_team_pos[1])])
            else:
                removed_list.append(None)
        # padding teammate position
        simulated_actions = []
        for i in range(self.depth - len(team_plan)):
            min_distance = float("inf")
            best_action = "Stop"
            for action in self.get_legal_actions(tmp_team_pos):
                new_team_pos = Actions.getSuccessor(tmp_team_pos, action)
                distance = closest_food(new_team_pos, cur_food)
                if distance is not None and distance < min_distance:
                    min_distance = distance
                    tmp_team_pos = new_team_pos
                    best_action = action
            simulated_actions.append(best_action)
            if tmp_team_pos in Actions.getLegalNeighbors(tmp_team_pos, self.walls) and \
               cur_food[int(tmp_team_pos[0])][int(tmp_team_pos[1])]:
                removed_list.append([int(tmp_team_pos[0]), int(tmp_team_pos[1])])
            else:
                removed_list.append(None)
        team_plan = team_plan + simulated_actions

        for i in range(depth):
            # Extracting self_pos from current tree node to compare previous simulation and teammate's new actions
            cur_self_pos = cur_node.get_self_pos()
            legal_actions = self.get_legal_actions(cur_self_pos)
            children = []
            # Generate the children of the current node
            for action in legal_actions:
                new_key = self.make_key(cur_key, action, team_plan[i], self.food_num)
                # Extract remaining values from current tree node to compute q-value
                cur_team_pos = cur_node.get_team_pos()
                cur_ghost_pos = cur_node.get_ghost_pos()
                cur_food = cur_node.get_food()
                if removed_list[i] and cur_food:
                    cur_food[removed_list[i][0]][removed_list[i][1]] = 0
                params = self.get_position_and_value(cur_self_pos, cur_team_pos, cur_ghost_pos, \
                                                     cur_food, action, team_plan[i])
                self.stats[new_key] = MCTNodes(*params)
                children.append((action, new_key))

            # Pick the next action
            values = [self.stats[key].get_value() for _, key in children]
            prob = random.uniform(0, 1)
            if prob < self.epsilon:
                values = [math.exp(value) for value in values]
                value_sum = sum(values)
                for i in range(len(values)):
                    values[i] = values[i] / value_sum
                prob = random.uniform(0, 1)
                cum_sum, idx = 0, 0
                for idx in range(len(values)):
                    if cum_sum <= prob < cum_sum + values[idx]:
                        break
                    else:
                        cum_sum += values[idx]
            else:
                idx = values.index(max(values))
            best_child = children[idx]
            cur_key = best_child[1]
            cur_node = self.stats[cur_key]
            path.append(best_child[0])
            rewards.append(self.stats[cur_key].get_value() / self.weights['bias'] + self.stats[cur_key].eats_pellet())
        return path, rewards, cur_key

    def choose_action(self, cur_state, team_plan):
        # Start timing
        start_time = time.time()

        # Extracting values from game state
        self_pos = cur_state.getAgentPosition(self.self_index)
        team_pos = cur_state.getAgentPosition(self.team_index)
        ghost_pos = cur_state.getAgentPosition(self.ghost_index)
        # food now is an instance of class grid
        food = self.getFood(cur_state)

        self.stats = util.Counter()
        self.stats["Root"] = MCTNodes(self_pos, team_pos, ghost_pos, food, 0)

        # Main search loop
        best_path = []
        best_reward = float("-inf")
        while True:
            # MCT nodes expanding process
            path, rewards, _ = self.search("Root", self.stats["Root"], self.depth, team_plan)
            # Compare the current best path and the newly found path
            weighted_reward = self.compute_reward(rewards)
            if best_reward < weighted_reward:
                best_path = path
                best_reward = weighted_reward
            if time.time() - start_time >= self.time_interval:
                break
        if not self.best_path or abs(self.best_reward-best_reward) > self.threshold:
            self.best_path = best_path
            self.best_reward = best_reward
        return self.best_path

    def get_legal_actions(self, pos):
        legal_actions = ["North", "South", "East", "West"]
        pos = (int(pos[0]), int(pos[1]))
        # Check if North is available
        if pos[1] >= self.walls.height - 2:
            legal_actions.remove("North")
        elif self.walls[pos[0]][pos[1] + 1]:
            legal_actions.remove("North")

        # Check if South is available
        if pos[1] <= 1:
            legal_actions.remove("South")
        elif self.walls[pos[0]][pos[1] - 1]:
            legal_actions.remove("South")

        # Check if East is available
        if pos[0] >= self.walls.width - 2:
            legal_actions.remove("East")
        elif self.walls[pos[0] + 1][pos[1]]:
            legal_actions.remove("East")

        # Check if West is available
        if pos[0] <= 1:
            legal_actions.remove("West")
        elif self.walls[pos[0] - 1][pos[1]]:
            legal_actions.remove("West")

        return legal_actions

    def make_key(self, cur_key, self_action, team_action, food_num):
        if cur_key == "Root":
            return ((self_action, team_action, food_num),)
        else:
            return tuple(list(cur_key) + [(self_action, team_action, food_num)])

    def compute_reward(self, rewards):
        reward_sum = 0
        for i in range(len(rewards)):
            reward_sum += rewards[i] * self.discounts ** i
        return reward_sum


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

    def __str__(self):
        return str({"my position": self.self_pos, "team position": self.team_pos, "ghost_position": self.ghost_pos,\
                    "value": self.value, "pellet": self.pellet})

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