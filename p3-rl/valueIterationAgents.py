# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations != 0:
            self.iterations -= 1
            new_values = util.Counter() # store new value of a state
            update_flag = util.Counter() # store whether a state has been updated
            for state in self.mdp.getStates():
                best_action = self.computeActionFromValues(state)
                if best_action:
                    new_value = self.computeQValueFromValues(state, best_action)
                    new_values[state] = new_value
                    update_flag[state] = 1
            for state in self.mdp.getStates():
                if update_flag[state]:
                    self.values[state] = new_values[state]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = .0
        state_prob = self.mdp.getTransitionStatesAndProbs\
                              (state, action)
        for new_state, prob in state_prob:
            q_value += prob * (self.mdp.getReward(state, action, new_state)+\
                              self.discount * self.getValue(new_state))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        best_action, best_reward = '', -1e9
        for action in actions:
            state_prob = self.mdp.\
                         getTransitionStatesAndProbs\
                         (state, action)
            reward = 0
            for new_state, prob in state_prob:
                reward += prob * (self.mdp.getReward(state, action, new_state)+\
                                 self.discount * self.getValue(new_state))
            if reward > best_reward:
                best_reward = reward
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        full_states = self.mdp.getStates()
        length = len(full_states)
        cnt = 0
        while self.iterations != 0:
            self.iterations -= 1
            state = full_states[cnt%length]
            cnt += 1
            best_action = self.computeActionFromValues(state)
            if best_action:
                self.values[state] = self.computeQValueFromValues(state, best_action)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # first, compute all pred of all states
        full_states = self.mdp.getStates()
        adjacent_matrix = []
        state_to_index = util.Counter()
        cnt = 0
        for s_i in full_states:
            adjacent_list = set()
            for s_j in full_states:
                actions = self.mdp.getPossibleActions(s_j)
                for action in actions:
                    state_prob = self.mdp.getTransitionStatesAndProbs(s_j, action)
                    for new_state, prob in state_prob:
                        if new_state == s_i and prob > 0:
                            adjacent_list.add(s_j)
            adjacent_matrix.append(adjacent_list)
            state_to_index[s_i] = cnt
            cnt += 1
        # initialize a priority queue
        p_queue = util.PriorityQueue()
        new_values = util.Counter()
        # find diff of each s, store new value in new_values, push s, -diff
        for state in full_states:
            actions = self.mdp.getPossibleActions(state)
            if self.mdp.isTerminal(state):
                continue
            current_value = self.getValue(state)
            best_action = self.computeActionFromValues(state)
            if best_action:
                new_value = self.computeQValueFromValues(state, best_action)
                new_values[state] = new_value
                diff = abs(current_value - new_value)
                p_queue.push(state, -diff)
            else:
                new_values[state] = current_value
        # do iterations
        for _ in range(self.iterations):
            # if p_queue is empty, terminate
            if p_queue.isEmpty():
                break
            front = p_queue.pop()
            if not self.mdp.isTerminal(front):
                self.values[front] = new_values[front]
            # precess front's pred
            for pred in adjacent_matrix[state_to_index[front]]:
                current_value = self.getValue(pred)
                best_action = self.computeActionFromValues(pred)
                if best_action:
                    new_value = self.computeQValueFromValues(pred, best_action)
                    diff = abs(current_value - new_value)
                    new_values[pred] = new_value
                    if diff > self.theta:
                        p_queue.update(pred, -diff)
