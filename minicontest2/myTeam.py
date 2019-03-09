from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, heapq
from game import Directions, Actions
import game
from util import nearestPoint
from numpy import exp, log2, sqrt, sqrt

def createTeam(firstIndex, secondIndex, isRed,
               first='QLearningAgent', second='QLearningAgent'):
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

class Deque:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def popBack(self):
        return self.list.pop(0)

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class PriorityQueue:
    def  __init__(self):
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

def softmax(counter):
    result = 0
    for element in counter:
        result += exp(counter[element])
    for element in counter:
        counter[element] = exp(counter[element]) / result
    return counter

class QLearningAgent(CaptureAgent):

    def registerInitialState(self, curState):
        CaptureAgent.registerInitialState(self, curState)

        # Training values
        self.weights = self.getWeights()
        self.epsilon = 0
        self.alpha = 0.3
        self.discount = 1

        # Map attributes
        self.is_dead_end = util.Counter()
        self.is_tunnel = util.Counter()
        self.is_crossing = util.Counter()
        self.is_open_area = util.Counter()
        self.walls = None
        self.width = None
        self.height = None

        # Attributes of the previous state
        self.pre_state = None
        self.pre_action = None
        self.pre_features = None
        self.preQValue = None
        self.curQValue = None

        # Attributes for pacman to make decisions
        self.carry_food = 0

        # Calculate in the initial 15 seconds
        self.preCalculate(curState)

    # Calculating the map's attributes in the initial 15 seconds
    def preCalculate(self, curState): #TODO replace it with a improved version
        walls = curState.getWalls()
        width = walls.width
        height = walls.height
        for i in range(width):
            for j in range(height):
                count = 0
                for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST,
                               Directions.WEST, Directions.STOP]:
                    dx, dy = Actions.directionToVector(action)
                    next_x, next_y = int(i + dx), int(j + dy)
                    if 0 <= next_x < width and 0 <= next_y < height \
                            and walls[next_x][next_y] is False:
                        count += 1
                if count <= 2:
                    self.is_dead_end[(i, j)] = 1
                elif count == 3:
                    self.is_tunnel[(i, j)] = 1
                elif count == 4:
                    self.is_crossing[(i, j)] = 1
                else:
                    self.is_open_area[(i, j)] = 1
        self.walls = walls
        self.width = width
        self.height = height

    # Pick among the next action with the highest Q(s, a)
    # And update the weights. Also updates preState with
    # curState and updates self.preQValue
    # Updates weights for features
    # Updates self.preQValue
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
        pre_food = pre_state.getBlueFood() if pre_state.isOnRedTeam(self.index)\
                   else pre_state.getRedFood()
        cur_food = state.getBlueFood() if state.isOnRedTeam(self.index)\
                   else state.getRedFood()
        food_bonus = 2 * (len(pre_food.asList()) - len(cur_food.asList()))
        if food_bonus > 0:
            self.carry_food += 1
        elif pre_agent_state.isPacman ^ cur_agent_state.isPacman:
            self.carry_food = 0
        food_bonus /= sqrt(self.carry_food + 1)
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
        stop_penalty, wandering_penalty = 0, 0
        if self.pre_action and len(self.pre_action)>1:
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

    # Set the weights of every feature
    def getWeights(self):
        weights = util.Counter()
        # FIXME replace this with initial weights
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
        weights['is_stop'] = -4
        weights['is_wandering'] = -5
        return weights

    # Find the next state after taking an action in curState
    def getNextState(self, curState, action):
        # TODO add ghost movement
        newState = curState.generateSuccessor(self.index, action)
        newPos = newState.getAgentState(self.index).getPosition()
        # Check if only half a grid point is moved
        if newPos != nearestPoint(newPos):
            return newState.generateSuccessor(self.index, action)
        else:
            return newState

    # Returns the food to defend
    def getOwnFood(self, gameState):
        if self.red:
            return gameState.getRedFood()
        else:
            return gameState.getBlueFood()

    # Finds the closest possible food
    def getClosestFood(self, curState):
        pos = curState.getAgentState(self.index).getPosition()
        food = self.getFood(curState).asList()
        # Initiate uniform cost graph search
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            x, y, dist = fringe.pop(0)
            if (x, y) in expanded:
                continue
            expanded.add((x, y))
            if (x, y) in food:
                return dist
            neighbours = Actions.getLegalNeighbors((x, y), self.walls)
            for newX, newY in neighbours:
                fringe.append((newX, newY, dist + 1))
        # No more food left
        return None

    # Extracts the features for a given state and action pair under a given policy
    def getFeatures(self, gameState, action):
        # Initiate
        features = util.Counter()
        next_state = self.getNextState(gameState, action)
        food_to_eat = self.getFood(gameState)
        food_to_defend = self.getFoodYouAreDefending(gameState)
        opponent_index = self.getOpponents(gameState)
        walls = next_state.getWalls()
        features['bias'] = 1.0
        new_agent_state = next_state.getAgentState(self.index)
        next_x, next_y = new_agent_state.getPosition()
        # if action is stop
        features['is_stop'] = action == Directions.STOP
        features['is_wandering'] = \
            self.pre_action[-1] == Actions.reverseDirection(action) \
                if self.pre_action else 0
        # if currently carry food
        features['carry_food'] = self.carry_food
        # Offensive / distance to closest dot
        min_distance = closestFood((int(next_x), int(next_y)), food_to_eat, walls)
        if min_distance is not None:
            features['min_distance_to_food'] = min_distance / \
                                               min(walls.width, walls.height) * 1.5

        # Offensive / distance to closest ghost
        distances = []
        opponent_pos = []
        if next_state.getAgentState(self.index).isPacman:
            features['is_pacman'] = 1
            for opponent in opponent_index:
                if gameState.getAgentState(opponent).isPacman is False:
                    opponent_pos.append(gameState.getAgentPosition(opponent))
                    distances.append(closestDistance((next_x, next_y), \
                                                     opponent_pos[-1], walls))
            if distances:
                features['closest_distance_to_ghost'] = min(distances) / \
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
                    features['return_distance'] = abs(next_x / (walls.width / 2) - 1) * \
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
                pacman_distance_to_food.append(closestFood(oppo_pos, \
                                                           food_to_eat, walls))
                # eat opponent in next move
                features['eat_pacman'] |= int(next_x) == int(oppo_pos[0]) and \
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
            features['closest_distance_to_pacman'] = min(distances) / \
                                                     min(walls.width, walls.height)
            features['average_distance_to_pacman'] = sum(distances) / \
                                                     min(walls.width, walls.height) / len(distances)
            features['opponent_closest_distance_to_food'] = \
                min(pacman_distance_to_food) / min(walls.width, walls.height)
        features.divideAll(10.0)
        return features


# ----------------------------------------------------------------------------------------------

    def getGhostFeatures(self, curState, action):
        return util.Counter()

    def getTactics(self, curState):
        # Calculate some of the features
        # TODO implement feature calculation
        survival = 0
        if survival < self.Tsurvival:
            return
        foodloss = 0
        retreat = 0
        edible = 0
        winrate = 0


        return 0

    def getQValue(self, curState, action, tactic):
        features = self.getFeatures(curState, action)
        weights = self.weights
        new_value = features * weights
        return new_value, features

    def MCTS(self, curState):
        QValues = util.Counter()
        Values = util.Counter()
        # First, determine the tactics and set the timer
        tactic = self.getTactics(curState)
        startTime = time.time()

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
                # Backpropogation
                cumulativeReward = 0
                while pathAndReward.isEmpty() is False:
                    state, reward = pathAndReward.pop()
                    cumulativeReward = reward + cumulativeReward * self.discount
                    Values[state] = cumulativeReward
                if cumulativeReward > bestReward:
                    bestReward = cumulativeReward
                    bestActions = tempActions
                (priority, _, _) = fringe.peekPriority()
                depthDiff = curDepth + priority
                curDepth = -priority
                for i in range(depthDiff):
                    tempActions.popBack()
                # print("Finished a search  ", curDepth)
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
                            print(chosenAction)
                            break
                        else:
                            cumulative += actionProb[prob]
                else:
                    chosenAction = actionProb.argMax()
                tempActions.push((chosenAction, QValues[topPos][chosenAction]))
                nextState = nextStates[state][chosenAction]

                # Determine whether to do a back track
                if util.flipCoin(1 / log2(curDepth + self.bias)):
                    fringe.push(curState, -curDepth)
                curDepth += 1
                fringe.push(nextState, -curDepth)
            endTime = time.time()
            if endTime - startTime > self.timeInterval:
                break
        self.actionsChosen = bestActions

    def chooseAction(self, curState):
        # these should be put in init
        self.actionsChosen = util.Queue()  # A pair of action and QValue (calculated wit features)
        self.tolerance = 3  # FIXME Set to actual tolerance
        self.depth = 12
        self.bias = 2
        self.timeInterval = 0.9  #FIXME time interval and depth should be reevaluated
        self.Tsurvival = 0
        self.Tretreat = 0
        self.Tfoodloss = 0
        self.Twinrate = 0

        # If no pre-calculated actions, initiate MCTS algorithm
        if self.actionsChosen.isEmpty():
            self.MCTS(curState)
        else:
            # calculate current features and determine whether to give up the old actions
            optimalAction, preQValue = self.actionsChosen.pop()
            tactic = self.getTactics(curState)
            legalActions = curState.getLegalActions()
            maxQValue = float("-inf")
            for action in legalActions:
                maxQValue = max(self.getQValue(curState, action, tactic)[0], maxQValue)
            diff = abs(maxQValue - preQValue)
            if diff > self.tolerance:
                self.MCTS(curState)
            else:
                self.preState = curState
                return optimalAction
        self.preState = curState
        return self.actionsChosen.pop()[0]
