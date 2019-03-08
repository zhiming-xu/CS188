#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
import game
import numpy as np
from util import nearestPoint

def createTeam(firstIndex, secondIndex, isRed,
               first='QLearningAgent', second='QLearningAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

def softmax(counter):
    sum = 0
    for iter in counter:
        sum += np.exp(counter[iter])
    for iter in counter:
        counter[iter] /= sum
    return counter

class QLearningAgent(CaptureAgent):

    def registerInitialState(self, curState):
        CaptureAgent.registerInitialState(self, curState)

        # Training values
        self.weights = self.getWeights()
        self.epsilon = 0.75
        self.alpha = 0.3
        self.discount = 1

        # Map attributes
        self.isDeadEnd = util.Counter()
        self.isTunnel = util.Counter()
        self.isCrossing = util.Counter()
        self.isOpenArea = util.Counter()
        self.walls = None
        self.width = None
        self.height = None

        # Attributes of the previous state
        self.preState = None
        self.preAction = None
        self.preFeatures = None
        self.preQValue = None
        self.curQValue = None

        # Attributes for pacman to make decisions
        self.carryingFood = 0

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
                    self.isDeadEnd[(i, j)] = 1
                elif count == 3:
                    self.isTunnel[(i, j)] = 1
                elif count == 4:
                    self.isCrossing[(i, j)] = 1
                else:
                    self.isOpenArea[(i, j)] = 1
        self.walls = walls
        self.width = width
        self.height = height

    # Pick among the next action with the highest Q(s, a)
    # And update the weights. Also updates preState with
    # curState and updates self.preQValue
    def chooseAction(self, curState):
        # Calculate optimal action
        legalActions = curState.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            bestAction = random.choice(legalActions)
            features = self.getFeatures(curState, bestAction)
            maxQValue = self.weights * features
            self.preFeatures = features
        else:
            bestAction, maxQValue = '', float('-inf')
            for action in legalActions:
                features = self.getFeatures(curState, action)
                reward = self.weights * features
                if reward > maxQValue:
                    maxQValue = reward
                    bestAction = action
                    self.preFeatures = features
        if self.preState:
            self.updateWeights(curState, maxQValue)
        self.preState = curState
        self.preQValue = maxQValue
        return bestAction

    # Updates weights for features
    # Updates self.preQValue
    def updateWeights(self, curState, curQValue):
        diff = self.getReward(curState) + self.discount * curQValue - self.preQValue
        for feature in self.preFeatures:
            self.weights[feature] += self.alpha * diff * self.preFeatures[feature]

    # Returns the reward the pacman gets from the previous action
    def getReward(self, curState):
        # Calculating related variables
        curPos = curState.getAgentState(self.index).getPosition()
        prePos = self.preState.getAgentState(self.index).getPosition()
        curFood = self.getFood(curState)
        preFood = self.getFood(self.preState)
        curDefendingFood = self.getOwnFood(curState)
        preDefendingFood = self.getOwnFood(self.preState)
        opponentIndex = self.getOpponents(curState)

        # The bonus for gaining score
        scoreBonus = 10 * (curState.getScore() - self.preState.getScore())
        if curState.isOnRedTeam(self.index) is False:
            scoreBonus *= -1

        # The bonus for eating a food pallet
        foodBonus = 2 * (len(preFood.asList()) - len(curFood.asList()))
        if foodBonus > 0:
            self.carryingFood += 1

        # The bonus for eating an opponent
        numEaten = 0
        for index in opponentIndex:
            preOpponentPos = self.preState.getAgentState(index).getPosition()
            curOpponentPos = curState.getAgentState(index).getPosition()
            if abs(preOpponentPos[0] - curOpponentPos[0]) + abs(preOpponentPos[1] - curOpponentPos[1]) > 3:
                numEaten += 1
        enemyEatenBonus = 3 * numEaten

        # The penalty for being eaten
        deathPenalty = 0
        if abs(curPos[0] - prePos[0]) + abs(curPos[1] - prePos[1]) > 3:
            deathPenalty = 4
            self.carryingFood = 0

        # The penalty for losing a defending pallet
        foodLossPenalty = 1 * (len(preDefendingFood.asList()) - len(curDefendingFood.asList()))

        reward = scoreBonus + foodBonus + enemyEatenBonus + deathPenalty + foodLossPenalty
        return reward

    # Set the weights of every feature
    def getWeights(self):
        weights = util.Counter()
        # FIXME replace this with initial weights
        weights['bias'] = 0.2
        weights['closestDistanceToGhost'] = 1.7
        weights['eatsFood'] = 3
        weights['minDistanceToFood'] = -1
        weights['numberOfGhostsOneStepAway'] = -2
        weights['numberOfGhostsTwoStepsAway'] = -1
        weights['returns'] = 10
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
    def getFeatures(self, curState, action):
        # Initiates a new counter
        features = util.Counter()

        # Calculate the new state
        newState = self.getNextState(curState, action)
        newPos = newState.getAgentState(self.index).getPosition()
        opponentIndex = self.getOpponents(curState)

        # Constant feature bias
        features['bias'] = 1

        # Offensive / number of ghosts 1 and 2 step(s) away
        distances = []
        if newState.getAgentState(self.index).isPacman:
            for opponent in opponentIndex:
                opponentPos = newState.getAgentState(opponent).getPosition()
                distances.append(self.getMazeDistance((newPos[0], newPos[1]), opponentPos))
            count1 = 0
            count2 = 0
            for d in distances:
                if d == 1:
                    count1 += 1
                if d <= 2:
                    count2 += 1
            features['numberOfGhostsOneStepAway'] = count1
            features['numberOfGhostsTwoStepsAway'] = count2

        # Offensive / a food is eaten in the last action (if there is no threat of ghosts)
        if newPos in self.getFood(curState).asList() and not features['numberOfGhostsTwoStepsAway']:
            features['eatsFood'] = 1

        # Offensive / returns to its own side, crossing the boarder
        if curState.getAgentState(self.index).isPacman and \
                not newState.getAgentState(self.index).isPacman:
            features['returns'] = self.carryingFood

        # Offensive / distance to closest dot (if there is no threat of ghosts)
        if self.getFood(newState) and not features['numberOfGhostsTwoStepsAway']:
            distance = self.getClosestFood(newState)
            if distance:
                features['minDistanceToFood'] = distance / max(self.width, self.height)

        '''
        # Offensive / terrain effects
        if newState.getAgentState(self.index).isPacman:
            if self.isDeadEnd[newPos]:
                features['isDeadEnd'] = 1
            elif self.isTunnel[newPos]:
                features['isTunnel'] = 1
            elif self.isCrossing[newPos]:
                features['isCrossing'] = 1
            else:
                features['isOpenArea'] = 1
        '''

        features.divideAll(10)
        return features


# ----------------------------------------------------------------------------------------------

    def getGhostFeatures(self, curState, action):
        return util.Counter()

    def getTactics(self, curState):
        # Calculate some of the features
        return 0

    def getQValue(self, curState, action, tactic):
        return 0

    def MCTS(self, curState):
        QValues = util.Counter(util.Counter())
        Values = util.Counter()
        # First, determine the tactics and set the timer
        tactic = self.getTactics(curState)
        startTime = time.time()

        # Do the main loop of MCTS
        fringe = util.Stack()
        fringe.push(curState)
        tempActions = util.Queue()
        bestActions = util.Queue()
        pathAndReward = util.Stack()
        expandedStates = util.Counter()
        nextStates = util.Counter(util.Counter())
        curDepth = 0
        while True:
            topState = fringe.pop()
            topPos = topState.getAgentPosition(self.index)
            if curDepth >= self.Depth:
                # Backpropogation
                cumulativeReward = 0
                while pathAndReward.isEmpty() is False:
                    state, reward = pathAndReward.pop()
                    cumulativeReward = reward + cumulativeReward * self.discount
                    Values[state] = cumulativeReward
            else:
                reward = self.getReward(topState)
                pathAndReward.push((topState, reward))
                if expandedStates[topState] > 0:
                    # Not only calculate Q(s, a), should consider V(s) for some descendants
                    expandedStates[topState] += 1
                    actionProb = util.Counter()
                    for action in nextStates[topState]:
                        nextState = nextStates[topState][action]
                        # If next state is expanded, use V(s)
                        if expandedStates[nextState] > 0:
                            actionProb[action] = Values[nextState]
                        # If next state is not expanded, use Q(s, a)
                        else:
                            actionProb[action] = QValues[nextState][action]
                    actionProb = softmax(actionProb)  # Calculate probability according to Q(s, a) or V(s)

                else:
                    # If the state has not been expanded, expand the state
                    expandedStates[topState] += 1
                    legalActions = topState.getLegalActions(self.index)
                    actionProb = util.Counter()
                    for action in legalActions:
                        QValues[topPos][action] = self.getQValue(topState, action, tactic)
                        actionProb[action] = QValues[topPos][action]
                        nextStates[topState][action] = self.getNextState(topState, action)
                    actionProb = softmax(actionProb)  # Calculate probability according to Q(s, a)

                # Choose action according to action probability
                flip = np.random.random()
                cumulative = 0
                chosenAction = "Error"  # Marking error
                for prob in actionProb:
                    if cumulative <= flip <= cumulative + actionProb[prob]:
                        chosenAction = prob
                    else:
                        cumulative += actionProb[prob]
                tempActions.push((chosenAction, QValues[topPos][chosenAction]))
                nextState = nextStates[topState][chosenAction]

                # Determine whether to do a back track
                if util.flipCoin(1 / np.log2(curDepth + self.bias)):
                    fringe.push(curState)
                fringe.push(nextState)
                curDepth += 1
            endTime = time.time()
            if endTime - startTime > self.timeInterval:
                break
        return bestActions

    def chooseAction2(self, curState):
        # these should be put in init
        self.actionsChosen = util.Queue()  # A pair of action and QValue (calculated wit features)
        self.tolerance = 3  # Set to tolerance
        self.Depth = 20
        self.bias = 2
        self.timeInterval = 900

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
                maxQValue = max(self.getQValue(curState, action, tactic), maxQValue)
            diff = abs(maxQValue - preQValue)
            if diff > self.tolerance:
                self.MCTS(curState)
            else:
                return optimalAction
        return self.actionsChosen.pop()[0]
