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
import game
from util import nearestPoint

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


#########
# Agent #
#########
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

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        teammateActions = self.receivedBroadcast
        # Process your teammate's broadcast!
        # Use it to pick a better action for yourself

        actions = gameState.getLegalActions(self.index)

        filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)

        currentAction = random.choice(actions) # Change this!
        return currentAction


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
    if len (legalActions) > 1 and reverse in legalActions:
        legalActions.remove(reverse)
    return legalActions
