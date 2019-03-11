from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, heapq
from game import Directions, Actions
import game
from util import nearestPoint
from numpy import exp, log10, sqrt


def createTeam(firstIndex, secondIndex, isRed,
               first='QLearningAgent', second='QLearningAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


distMap = None
nodeMap = None

agentIntentions = []
agentActions = []


def closestFood(pos, food):
    foodList = food.asList()
    if len(foodList) == 0:
        return None
    minDistance = float("inf")
    for foodLoc in foodList:
        minDistance = min(minDistance, distMap[pos][foodLoc])
    return minDistance


def adjNode(pos):
    global nodeMap
    return nodeMap[pos].getNeighbour()


def nodeType(pos):
    global nodeMap
    return nodeMap[pos].getNodeType()


def minDist(pos, target):
    global distMap
    return distMap[pos][target]


def softmax(counter):
    result = 0
    for element in counter:
        result += exp(counter[element])
    for element in counter:
        counter[element] = exp(counter[element]) / result
    return counter


def breadthFirstSearch(pos, target, walls):
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
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no path found
    return None


def hashMap(curState):
    global distMap
    global nodeMap
    if distMap:
        return

    # Brutal search
    distMap = util.Counter()
    nodeMap = util.Counter()  # For discretion of the map
    fringe = Deque()
    closed = set()
    walls = curState.getWalls()
    pos = curState.getAgentPosition(0)
    fringe.push(pos)
    for x in range(walls.width):
        for y in range(walls.height):
            distMap[(x, y)] = util.Counter()
    while fringe.isEmpty() is False:
        curPos = fringe.pop()
        curOppoPos = (walls.width - curPos[0] - 1, walls.height - curPos[1] - 1)
        closed.add(curPos)
        neighbours = Actions.getLegalNeighbors(curPos, walls)

        # For discretion of the map
        if len(neighbours) < 3 and curPos[0] != round(walls.width / 2) - 1:  # The node is an edge
            nodeMap[curPos] = Node(0)
            nodeMap[curOppoPos] = Node(0)
        elif curPos[0] == round(walls.width / 2) - 1:  # The node is a boarder vertex
            nodeMap[curPos] = Node(2)
            nodeMap[curOppoPos] = Node(2)
        else:  # The node is a vertex
            nodeMap[curPos] = Node(1)
            nodeMap[curOppoPos] = Node(1)

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
                oppoPosition = (walls.width - position[0] - 1, walls.height - position[1] - 1)
                distMap[curPos][position] = distMap[position][curPos] = distMap[curOppoPos][oppoPosition] = \
                    distMap[oppoPosition][curOppoPos] = breadthFirstSearch(curPos, position, walls)
                # print(curPos, "->", position, ":", distMap[curPos][position]) # FIXME for debug purpose
    closed.clear()

    # Calculate distances between points that are in different colors
    x = round(walls.width / 2) - 1
    for y in range(walls.height):
        if walls[x][y] != 0:
            continue
        fringe.push(pos)
        while fringe.isEmpty() is False:
            curPos = fringe.pop()
            closed.add(curPos)
            neighbours = Actions.getLegalNeighbors(curPos, walls)
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
                    distance = distMap[curPos][(x, y)] + distMap[(x + 1, y)][position] + 1
                    if distMap[curPos][position] == 0 or distance < distMap[curPos][position]:
                        distMap[curPos][position] = distMap[position][curPos] = distance
                        # print(curPos, "->", position, ":", distance) # FIXME for debug purpose
    closed.clear()

    # Finding connecting edges and vertices
    x = round(walls.width / 2) - 1
    for y in range(walls.height):
        if nodeMap[(x, y)] == 2 and walls[x + 1][y] == 0:
            pos = (x, y)
            break
    vertices = set()
    vertices.add(pos)
    curEdge = set()
    while not vertices:
        curVertex = vertices.pop()
        oppoCurVertex = (walls.width - curVertex[0] - 1, walls.height - curVertex[1] - 1)
        neighbours = Actions.getLegalNeighbors(curVertex, walls)
        for neighbour in neighbours:
            if neighbour not in closed and neighbour[0] < round(walls.width / 2):
                fringe.push(neighbour)
                while fringe.isEmpty() is False:
                    curPos = fringe.pop()
                    closed.add(curPos)
                    if nodeMap[curPos].getNodeType() > 0:  # The node is a vertex of any kind
                        # Add the vertex to vertices since we will expand it later
                        if curPos not in closed:
                            vertices.add(curPos)
                        oppoCurPos = (walls.width - curPos[0] - 1, walls.height - curPos[1] - 1)
                        if len(curEdge) == 0:
                            nodeMap[curVertex].addNeighbour(curPos)
                            nodeMap[curPos].addNeighbour(curVertex)
                            nodeMap[oppoCurVertex].addNeighbour(oppoCurPos)
                            nodeMap[oppoCurPos].addNeighbour(oppoCurVertex)
                        else:
                            for edge in curEdge:
                                oppoEdge = (walls.width - edge[0] - 1, walls.height - edge[1] - 1)
                                nodeMap[curPos].addNeighbour(edge)
                                nodeMap[curVertex].addNeighbour(edge)
                                nodeMap[edge].addNeighbour(curVertex)
                                nodeMap[edge].addNeighbour(curPos)
                                nodeMap[oppoCurPos].addNeighbour(oppoEdge)
                                nodeMap[oppoCurVertex].addNeighbour(oppoEdge)
                                nodeMap[oppoEdge].addNeighbour(oppoCurVertex)
                                nodeMap[oppoEdge].addNeighbour(oppoCurPos)
                        # Also determine if it is a boarder vertex and leads to the opponent's field
                        boardering = (curPos[0] + 1, curPos[1])
                        if nodeMap[curPos] == 2 and walls[boardering[0]][boardering[1]] == 0:
                            nodeMap[curPos].addNeighbour(boardering)
                            oppoBoardering = (walls.width - boardering[0] - 1, walls.height - boardering[1] - 1)
                            nodeMap[oppoCurPos].addNeighbour(oppoBoardering)
                        curEdge.clear()
                    else:  # The node is an edge
                        curEdge.add(curPos)
                # When the fringe is empty, we reached a dead end in an edge
                for edge in curEdge:
                    oppoEdge = (walls.width - edge[0] - 1, walls.height - edge[1] - 1)
                    nodeMap[curVertex].addNeighbour(edge)
                    nodeMap[edge].addNeighbour(curVertex)
                    nodeMap[oppoCurVertex].addNeighbour(oppoEdge)
                    nodeMap[oppoEdge].addNeighbour(oppoCurVertex)
                curEdge.clear()
        # When all neighbours of a vertex is expanded, we mark the vertex as expanded
        closed.add(curVertex)
    closed.clear()


class Deque:
    def __init__(self):
        self.list = []

    def push(self, item):
        self.list.insert(0, item)

    def pop(self):
        return self.list.pop()

    def popBack(self):
        return self.list.pop(0)

    def peekItem(self):
        return self.list[0]

    def isEmpty(self):
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

    def peekPriority(self):
        return min(self.heap)

    def popItemAndPriority(self):
        (priority, _, item) = heapq.heappop(self.heap)
        return item, priority

    def isEmpty(self):
        return len(self.heap) == 0


class Node:
    def __init__(self, nodeType):
        self.nodeType = nodeType  # 0 for edge, 1 for vertex, 2 for boarder vertex
        self.neighbours = set()  # record the neighbour of the node, an edge must have 2 or less

    def getNodeType(self):
        return self.nodeType

    def addNeighbour(self, pos):
        self.neighbours.add(pos)

    def getNeighbour(self):
        return self.neighbours


class QLearningAgent(CaptureAgent):

    def registerInitialState(self, curState):
        CaptureAgent.registerInitialState(self, curState)
        # Initiate agent intentions with dummy values
        global agentIntentions
        global agentActions
        agentIntentions.append((0, 0))
        agentIntentions.append((0, 0))
        agentActions.append('Stop')
        agentActions.append('Stop')

        # Training values
        self.weights = self.getWeights()
        self.epsilon = 0.1
        self.alpha = 0.3  # FIXME no use anymore
        self.discount = 0.9

        # Attributes of the previous state
        self.pre_state = None
        self.pre_action = None
        self.pre_features = None
        self.preQValue = None
        self.curQValue = None

        # Attributes for pacman to make decisions
        self.carry_food = 0
        self.beginningFood = 20
        self.maxMazeDistance = 512

        # Calculate in the initial 15 seconds
        hashMap(curState)

        # Values needed for MCTS
        self.actionsChosen = Deque()  # A pair of action and QValue (calculated wit features)
        self.tolerance = 1  # FIXME Set to actual tolerance
        self.depth = 12
        self.bias = 0
        self.timeInterval = 0.9  # FIXME time interval and depth should be reevaluated

        # Thresholds, all thresholds are exclusive
        self.Tsurvival = 6
        self.Tretreat = 12  # Must be less than 20
        self.TloseRate = 10
        self.TinvadeRate = 10
        self.Twinrate = 0  # TODO not implemented yet
        self.dangerRange = 5  # Pacman will be alerted if there is a ghost within the range
        self.foodRegret = 1  # The penalty for losing one food pallet (comparing to threshold TloseRate or TinvadeRate)
        self.escapeExpect = 2  # The expected chance for an enemy to escape (times map width / 2 * distance to boarder)
        self.foodGain = 5  # The expected reward for carrying a food pallet
        self.captureExpect = 3  # The threshold for the ghost to actively try to catch pacman
        self.trapExpect = 5  # The threshold for ghost to actively try to trap pacman
        self.blockExpect = 3  # The threshold for ghost to actively try to block the vertex near the pacman
        self.invadeExpect = 2  # The expected chance for an enemy to invade (times map width / 2 * distance to boarder)

    # Returns the reward the pacman gets from the previous action
    def getReward(self, state):
        pre_state = self.getPreviousObservation()
        # food_bonus = 2 * self.pre_features['eat_food']
        # oppo_bonus = 4 * self.pre_features['eat_opponent']
        # score bonus
        if pre_state is None:
            return 0
        score_bonus = 10 * (state.getScore() - pre_state.getScore())
        score_bonus = score_bonus if state.isOnRedTeam(self.index) else -score_bonus
        # food bonus
        pre_agent_state = pre_state.getAgentState(self.index)
        cur_agent_state = state.getAgentState(self.index)
        pre_pos = pre_agent_state.getPosition()
        cur_pos = cur_agent_state.getPosition()
        pre_food = pre_state.getBlueFood() if pre_state.isOnRedTeam(self.index) \
            else pre_state.getRedFood()
        cur_food = state.getBlueFood() if state.isOnRedTeam(self.index) \
            else state.getRedFood()
        food_bonus = 2 * (len(pre_food.asList()) - len(cur_food.asList()))
        if food_bonus > 0:
            self.carry_food += 1
        elif pre_agent_state.isPacman ^ cur_agent_state.isPacman:
            self.carry_food = 0
        food_bonus /= sqrt(self.carry_food + 1)
        # food loss
        pre_food = pre_state.getRedFood() if pre_state.isOnRedTeam(self.index) \
            else pre_state.getBlueFood()
        cur_food = state.getRedFood() if state.isOnRedTeam(self.index) \
            else state.getBlueFood()
        food_loss = 2 * (len(pre_food.asList()) - len(cur_food.asList()))
        # death penalty
        if abs(cur_pos[0] - pre_pos[0]) + abs(cur_pos[1] - pre_pos[1]) > 3:
            score_bonus -= 1
        # stop penalty
        stop_penalty, wandering_penalty = 0, 0
        if self.pre_action and len(self.pre_action) > 1:
            stop_penalty = self.pre_action[-1] == Directions.STOP
            wandering_penalty = \
                self.pre_action[-1] == Actions.reverseDirection(self.pre_action[-2]) \
                    if len(self.pre_action) > 1 else 0
        # eat opponents
        oppo_index = self.getOpponents(state)
        old_pos = [pre_state.getAgentPosition(oppo) for oppo in oppo_index]
        new_pos = [state.getAgentPosition(oppo) for oppo in oppo_index]
        num_oppo_eat = 0
        for i in range(len(old_pos)):
            if abs(old_pos[i][0] - new_pos[i][0]) + abs(old_pos[i][1] - new_pos[i][1]) > 3:
                num_oppo_eat += 1
        oppo_bonus = num_oppo_eat * 1
        reward = food_bonus + score_bonus + oppo_bonus - food_loss - stop_penalty - \
                 wandering_penalty
        return reward

    # Set the weights of every feature
    def getWeights(self):
        weights = util.Counter()
        weights['min_distance_to_pacman'] = -2
        weights['is-pacman-1-step-away'] = 3.4
        weights['is-pacman-2-step-away'] = 2.8
        weights['eat_pacman'] = 6.4
        weights['average_distance_to_retreat'] = -5
        weights['distance_to_teammate'] = 1
        weights['closest_distance_to_target'] = -2
        weights['average_distance_to_exits'] = -2
        weights['closest_distance_to_boarder'] = -2
        # for offensive
        weights['min_distance_to_food'] = -1
        weights['carry_food'] = 2
        weights['eat_food'] = 4
        weights['return_distance'] = -5
        weights['is-ghosts-1-step-away'] = -3
        weights['is-ghosts-2-step-away'] = -2
        weights['is-ghosts-3-step-away'] = -2
        weights['average_distance_to_retreat'] = -1
        weights['is_pacman'] = 3.2
        weights['food_nearby'] = 1
        weights['closest_distance_to_ghost'] = -1
        # for both
        weights['bias'] = 1
        weights['is_stop'] = -4
        weights['is_wandering'] = -5
        return weights

    # Find the next state after taking an action in curState
    def getNextState(self, curState, action):
        global agentActions
        newState = curState.generateSuccessor(self.index, action)
        newPos = newState.getAgentState(self.index).getPosition()
        # Check if only half a grid point is moved
        if newPos != nearestPoint(newPos):
            return newState.generateSuccessor(self.index, action)
        else:
            # Simulate teammate movement
            # teammate = self.getTeamMate(self.index)
            # newState = newState.generateSuccessor(teammate, agentActions[teammate])

            # Simulate opponent movements
            for oppo in self.getOpponents(curState):
                oppoState = curState.getAgentState(oppo)
                legalActions = curState.getLegalActions(oppo)
                maxQValue = float("-inf")
                bestAction = None
                if oppoState.isPacman:
                    for action in legalActions:
                        expectQValue = self.weights * self.getPacmanFeatures(curState, action, oppo)
                        if maxQValue < expectQValue:
                            maxQValue = expectQValue
                            bestAction = action
                else:
                    for action in legalActions:
                        expectQValue = self.weights * self.getGhostFeatures(curState, action, oppo)
                        if maxQValue < expectQValue:
                            maxQValue = expectQValue
                            bestAction = action
                curState = curState.generateSuccessor(oppo, bestAction)
            return newState

    def getOwnBoarder(self, walls):
        if self.red:
            return round(walls.width / 2) - 1
        else:
            return round(walls.width / 2)

    def getOppoBoarder(self, walls):
        if self.red:
            return round(walls.width / 2)
        else:
            return round(walls.width / 2) - 1

    # Returns the food to defend
    def getOwnFood(self, gameState):
        if self.red:
            return gameState.getRedFood()
        else:
            return gameState.getBlueFood()

    def getEnemies(self, index):
        if index == 0 or index == 2:
            return [1, 3]
        elif index == 1 or index == 3:
            return [0, 2]

    def getTeamMate(self, index):
        if index == 0:
            return 2
        elif index == 1:
            return 3
        elif index == 2:
            return 0
        return 1

    # Extracts the features for a given state and action pair under a given policy
    def getFeatures(self, gameState, action, tactic):
        if gameState.getAgentState(self.index).isPacman:
            return self.getPacmanFeatures(gameState, action, self.index, tactic, flag=True)
        else:
            return self.getGhostFeatures(gameState, action, self.index, tactic, flag=True)

    def getPacmanFeatures(self, curState, action, index, tactic="food_score", flag=False):
        global agentIntentions
        features = util.Counter()
        ghost_distance = []
        if flag:
            nextState = self.getNextState(curState, action)
        else:
            nextState = curState.generateSuccessor(index, action)
        nextPos = nextState.getAgentPosition(index)
        opponentIndex = self.getEnemies(index)
        ghostPos = []
        pacmanPos = []
        for oppo in opponentIndex:
            if curState.getAgentState(oppo).isPacman is False:
                ghostPos.append(curState.getAgentPosition(oppo))
            else:
                pacmanPos.append(curState.getAgentPosition(oppo))
        walls = curState.getWalls()
        foodToEat = self.getFood(curState).asList()
        if ghostPos:
            ghost_distance = [minDist(oppo, nextPos) for oppo in ghostPos]
            min_distance_ghost = min(ghost_distance)
            features['is-ghosts-3-step-away'] = min_distance_ghost <= 3
            # features['is-ghosts-2-step-away'] = min_distance <= 2
            # features['is-ghosts-1-step-away'] = min_distance <= 1
        if tactic == "food_score":
            food_distance = [minDist(food, nextPos) for food in foodToEat]
            min_distance_to_food = min(food_distance)
            features['min_distance_to_food'] = min_distance_to_food
            features['eat_food'] = min_distance_to_food == 0
            features['carry_food'] = self.carry_food
        elif tactic == "retreat":
            minDistance = self.maxMazeDistance
            subMinDistance = self.maxMazeDistance
            x = self.getOwnBoarder(walls)
            for y in range(walls.height):
                if walls[x][y] == 0:
                    distance = minDist(nextPos, (x, y))
                    if distance < minDistance:
                        subMinDistance = minDistance
                        minDistance = distance
                    elif distance < subMinDistance:
                        subMinDistance = distance
            if subMinDistance != self.maxMazeDistance:
                features["average_distance_to_retreat"] = 0.7 * minDistance + 0.3 * subMinDistance
        elif tactic == "defend_food":
            if pacmanPos:
                minDistance = self.maxMazeDistance
                subMinDistance = self.maxMazeDistance
                x = self.getOwnBoarder(walls)
                for y in range(walls.height):
                    if walls[x][y] == 0:
                        distance = minDist(nextPos, (x, y))
                        if distance < minDistance:
                            subMinDistance = minDistance
                            minDistance = distance
                        elif distance < subMinDistance:
                            subMinDistance = distance
                if subMinDistance != self.maxMazeDistance:
                    features["average_distance_to_retreat"] = 0.7 * minDistance + 0.3 * subMinDistance
                pacman_distance = [minDist(oppo, nextPos) for oppo in pacmanPos]
                min_distance_pacman = min(pacman_distance)
                features['min_distance_to_pacman'] = min_distance_pacman
                features['average_distance_to_retreat'] = 0.7 * minDistance + 0.3 * subMinDistance
                teammatePos = curState.getAgentPosition((self.index + 2) % 4)
                features['distance_to_teammate'] = minDist(teammatePos, nextPos)
        if ghost_distance:
            features['is-ghosts-2-step-away'] = min_distance_ghost <= 2
            features['is-ghosts-1-step-away'] = min_distance_ghost <= 1
            features['min_distance_to_ghost'] = min_distance_ghost
        return features

    def getGhostFeatures(self, curState, action, index, tactic="chase_pacman", flag=False):
        # FIXME probably should replace 30 with map size
        global agentIntentions
        features = util.Counter()
        if flag:
            nextState = self.getNextState(curState, action)
        else:
            nextState = curState.generateSuccessor(index, action)
        nextPos = nextState.getAgentPosition(index)
        opponentIndex = self.getEnemies(index)
        walls = curState.getWalls()
        if tactic == "chase_pacman":
            features["closest_distance_to_target"] = minDist(nextPos, agentIntentions[index])
        elif tactic == "block_pacman":
            features["closest_distance_to_target"] = minDist(nextPos, agentIntentions[index])
        elif tactic == "guard_exits":
            return util.Counter()
        elif tactic == "patrol_boarder":
            # TODO add penalty for being too close to the boarder
            avgDistance = 0
            i = 0
            x = self.getOwnBoarder(walls)
            for y in range(walls.height):
                if walls[x][y] == 0:
                    i += 1
                    avgDistance += minDist(nextPos, (x, y))
            features["average_distance_to_exits"] = avgDistance / i
            # for opponent in opponentIndex:
        elif tactic == "rush_to_food":
            minDistance = self.maxMazeDistance
            x = self.getOppoBoarder(walls)
            for y in range(walls.height):
                if walls[x][y] == 0:
                    minDistance = min(minDistance, minDist(nextPos, (x, y)))
            features["closest_distance_to_boarder"] = minDistance
            print('-------------------\n', minDistance, action)
        return features

    def getTactics(self, curState, index):
        global agentIntentions
        walls = curState.getWalls()
        agentState = curState.getAgentState(index)
        agentPos = agentState.getPosition()
        opponentIndex = self.getEnemies(index)
        opPos0 = curState.getAgentState(opponentIndex[0])
        opPos1 = curState.getAgentState(opponentIndex[1])

        if agentState.isPacman:
            # --------------------------------------------------------
            # the survival tactic
            # --------------------------------------------------------
            vertices = adjNode(agentPos)
            exits = []
            edges = []
            for vertex in vertices:
                if nodeType(vertex) > 0:  # The node is not an edge
                    exits.append(vertex)
                else:
                    edges.append(vertex)
            survivalRate = 0
            if len(exits) == 0:  # Pacman is at a crossing, and has no neighbour vertex
                # We select out all the next level vertices
                nextLayerVtx = set()
                for edge in edges:
                    for node in adjNode(edge):
                        if nodeType(node) > 0:
                            nextLayerVtx.add(node)
                nextLayerVtx.discard(agentPos)
                exits = list(nextLayerVtx)
                if len(exits) > 2:  # Pacman is at a crossing, and has neighbour vertices
                    # FIXME some repeating code here
                    survivalRate = 20
                elif len(exits) == 1:
                    survivalRate = min(minDist(opPos0, exits[0]),
                                       minDist(opPos1, exits[0])) - minDist(agentPos, exits[0])
                elif len(exits) == 2:
                    survivalRate = min(minDist(opPos0, exits[0]) + minDist(opPos1, exits[1]),
                                       minDist(opPos0, exits[1]) + minDist(opPos1, exits[0])) - \
                                   min(minDist(agentPos, exits[0]), minDist(agentPos, exits[1]))
            elif len(exits) > 2:  # Pacman is at a crossing, and has neighbour vertices
                # We think we are safe
                # TODO this could result in the pacman being over optimistic in edges of width 2 or larger
                # TODO implement a better algorithm if necessary
                survivalRate = 16
            elif len(exits) == 1:  # Pacman is in a dead end
                survivalRate = min(minDist(opPos0, exits[0]), minDist(opPos1, exits[0])) - minDist(agentPos, exits[0])
            elif len(exits) == 2:  # Pacman is in a tunnel
                survivalRate = min(minDist(opPos0, exits[0]) + minDist(opPos1, exits[1]), minDist(opPos0, exits[1]) + \
                                   minDist(opPos1, exits[0])) - min(minDist(agentPos, exits[0]),
                                                                    minDist(agentPos, exits[1]))
            # Determine whether there is a ghost nearby
            if minDist(agentPos, opPos0) < self.dangerRange:
                survivalRate -= 10
            if minDist(agentPos, opPos1) < self.dangerRange:
                survivalRate -= 10
            if survivalRate < self.Tsurvival:
                return "survival"
            # --------------------------------------------------------
            # the ghost score tactic TODO probably encourage pacman to travel deeper in the first 20 sec, and than leave
            # --------------------------------------------------------

            # --------------------------------------------------------
            # the defend food tactic TODO probably should consider the time to return to the boarder
            # --------------------------------------------------------
            invaders = []
            for opponent in opponentIndex:
                if curState.getAgentState(opponent).isPacman:
                    invaders.append(curState.getAgentPosition(opponent))
            if invaders:
                foodLoss = self.foodRegret * (self.beginningFood - len(self.getOwnFood(curState).asList()))
                minDistance = self.maxMazeDistance
                x = self.getOwnBoarder(walls)
                for y in range(walls.height):
                    if walls[x][y] == 0:
                        for invader in invaders:
                            minDistance = min(minDistance, minDist((x, y), invader))
                escapeChance = self.escapeExpect * walls.width / (2 * (minDistance + 1))
                lossRate = foodLoss + escapeChance
                if lossRate > self.TloseRate:
                    return "defend_food"
            # --------------------------------------------------------
            # the retreat tactic
            # --------------------------------------------------------
            expectedReward = self.foodGain * self.carry_food
            expectedReward += 3 * (self.beginningFood - len(curState.getFood().asList()))  # Count eaten food as well
            minDistance = 80  # This is set to prevent errors, and give some trustworthy information
            strictMinDistance = self.maxMazeDistance  # When pacman takes the risk to rush out from backup location
            evacuatePos = None
            backupPos = None
            x = self.getOwnBoarder(walls)
            for y in range(walls.height):
                if walls[x][y] == 0:
                    escapeTime = minDist((x, y), agentPos)
                    if escapeTime < strictMinDistance:
                        strictMinDistance = escapeTime
                        backupPos = (x, y)
                    blockTime = min(minDist(opPos0, (x, y)), minDist(opPos1, (x, y)))
                    if escapeTime < blockTime:
                        minDistance = min(minDistance, escapeTime)
                        evacuatePos = (x, y)
            retreatRate = expectedReward / (minDistance / walls.width)
            if retreatRate > self.Tretreat:
                if evacuatePos:
                    agentIntentions[index] = evacuatePos
                else:
                    agentIntentions[index] = backupPos
                return "retreat"
            # --------------------------------------------------------
            # the food score tactic
            # --------------------------------------------------------
            return "food_score"  # The default tactic for an agent as a pacman

        else:
            invaders = []
            for opponent in opponentIndex:
                if curState.getAgentState(opponent).isPacman:
                    invaders.append(curState.getAgentPosition(opponent))
            if invaders:  # FIXME performs poorly when two enemies are invading
                for invader in invaders:
                    # --------------------------------------------------------
                    # the dodging tactic  # TODO not implemented
                    # --------------------------------------------------------

                    # --------------------------------------------------------
                    # the chasing tactic
                    # --------------------------------------------------------
                    invaderPos = invader
                    vertices = adjNode(invaderPos)
                    if minDist(agentPos, invaderPos) < self.captureExpect:
                        agentIntentions[index] = invaderPos
                        return "chase_pacman"
                    elif len(vertices) == 1:
                        diff = minDist(agentPos, vertices[0]) - minDist(invaderPos, vertices[0])
                        if diff < self.trapExpect:
                            agentIntentions[index] = vertices[0]
                            return "chase_pacman"
                    # --------------------------------------------------------
                    # the blocking tactic
                    # --------------------------------------------------------
                    elif len(vertices) == 2 and not curState.getAgentState(self.getTeamMate(index)).isPacman:
                        diff = self.maxMazeDistance
                        target = None
                        for vertex in vertices:
                            tmpDiff = minDist(agentPos, vertex) - minDist(invaderPos, vertex)
                            if tmpDiff < diff and agentIntentions[self.getTeamMate(index)] != vertex:
                                diff = tmpDiff
                                target = vertex
                        if diff < self.blockExpect:
                            agentIntentions[index] = target
                            return "block_pacman"
                    # --------------------------------------------------------
                    # the guarding tactic
                    # --------------------------------------------------------
                    # The default and pessimistic single ghost defensive strategy
                    return "guard_exits"
            else:
                # --------------------------------------------------------
                # the patrolling tactic
                # --------------------------------------------------------
                foodLoss = self.foodRegret * (self.beginningFood - len(self.getOwnFood(curState).asList()))
                minDistance = self.maxMazeDistance
                x = self.getOwnBoarder(walls)
                for y in range(walls.height):
                    if walls[x][y] == 0:
                        for opponent in opponentIndex:
                            minDistance = min(minDistance, minDist((x, y), curState.getAgentPosition(opponent)))
                invadeChance = self.invadeExpect * walls.width / (2 * (minDistance + 1))
                invadeRate = foodLoss + invadeChance
                if invadeRate > self.TinvadeRate:
                    return "patrol_boarder"
                # --------------------------------------------------------
                # the rush_to_food tactic
                # --------------------------------------------------------
                # The default tactic for ghosts with no invaders
                return "rush_to_food"

    def getQValue(self, curState, action, tactic):
        features = self.getFeatures(curState, action, tactic)
        weights = self.weights
        new_value = features * weights
        return new_value, features

    def MCTS(self, curState):
        # First, determine the tactics and set the timer
        startTime = time.time()
        QValues = util.Counter()
        Values = util.Counter()
        tactic = self.getTactics(curState, self.index)
        print(self.index, "  ", tactic)  # FIXME for debug purpose

        # Do the main loop of MCTS
        fringe = PriorityQueue()
        fringe.push(curState, 0)
        tempActions = Deque()
        bestActions = Deque()
        pathAndReward = util.Stack()
        bestReward = float("-inf")
        expandedStates = util.Counter()
        nextStates = util.Counter()
        curDepth = 0
        while fringe.isEmpty() is False:
            state = fringe.pop()
            topPos = state.getAgentPosition(self.index)
            if curDepth >= self.depth:
                # Backpropagation
                cumulativeReward = 0
                while pathAndReward.isEmpty() is False:
                    state, reward = pathAndReward.pop()
                    cumulativeReward = reward + cumulativeReward * self.discount
                    Values[state] = cumulativeReward
                if cumulativeReward > bestReward:
                    bestReward = cumulativeReward
                    bestActions = tempActions
                    # print(bestActions.list)
                    # util.pause()
                (priority, _, _) = fringe.peekPriority()
                depthDiff = curDepth + priority
                curDepth = -priority
                for _ in range(depthDiff):
                    tempActions.popBack()
            else:
                reward = self.getReward(state)
                pathAndReward.push((state, reward))
                if expandedStates[state] > 0:
                    # Not only calculate Q(s, a), should consider V(s) for some descendants
                    expandedStates[state] += 1
                    actionProb = util.Counter()
                    for action in nextStates[state]:
                        nextState = nextStates[state][action]
                        # If next state is expanded, use V(s)
                        if expandedStates[nextState] > 0:
                            actionProb[action] = Values[nextState]
                        # If next state is not expanded, use Q(s, a)
                        else:
                            actionProb[action] = QValues[topPos][action]
                    # Calculate probability according to Q(s, a) or V(s)
                    actionProb = softmax(actionProb)
                else:
                    # If the state has not been expanded, expand the state
                    expandedStates[state] += 1
                    legalActions = state.getLegalActions(self.index)
                    actionProb = util.Counter()
                    for action in legalActions:
                        # print(self.getQValue(topState, action, tactic), QValues[topPos][action])
                        if QValues[topPos] == 0:
                            QValues[topPos] = util.Counter()
                        QValues[topPos][action] = self.getQValue(state, action, tactic)[0]
                        actionProb[action] = QValues[topPos][action]
                        if nextStates[state] == 0:
                            nextStates[state] = util.Counter()
                        nextStates[state][action] = self.getNextState(state, action)
                    actionProb = softmax(actionProb)  # Calculate probability according to Q(s, a)
                # Choose action according to action probability
                flip = random.random()
                cumulative = 0
                chosenAction = "Error"  # Marking error
                if util.flipCoin(self.epsilon):
                    for prob in actionProb:
                        if cumulative <= flip <= cumulative + actionProb[prob]:
                            chosenAction = prob
                            break
                        else:
                            cumulative += actionProb[prob]
                else:
                    chosenAction = actionProb.argMax()
                tempActions.push((chosenAction, QValues[topPos][chosenAction]))
                nextState = nextStates[state][chosenAction]

                # Determine whether to do a back track
                if util.flipCoin(1 / exp(.4 * (curDepth + self.bias))):
                    fringe.push(curState, -curDepth)
                curDepth += 1
                fringe.push(nextState, -curDepth)
            endTime = time.time()
            if endTime - startTime > self.timeInterval:
                break
        self.actionsChosen = bestActions

    def chooseAction(self, curState):
        # If no pre-calculated actions, initiate MCTS algorithm
        global agentActions
        if self.actionsChosen.isEmpty():
            self.MCTS(curState)
        else:
            # calculate current features and determine whether to give up the old actions
            optimalAction, preQValue = self.actionsChosen.pop()
            tactic = self.getTactics(curState, self.index)
            print(self.index, "  ", tactic)  # FIXME for debug purpose
            legalActions = curState.getLegalActions(self.index)
            maxQValue = float("-inf")
            for action in legalActions:
                maxQValue = max(self.getQValue(curState, action, tactic)[0], maxQValue)
            diff = abs(maxQValue - preQValue)
            if diff > self.tolerance or optimalAction not in legalActions:
                self.MCTS(curState)
            else:
                self.pre_state = curState
                return optimalAction
        self.pre_state = curState
        optimalAction = self.actionsChosen.pop()[0]
        return optimalAction
