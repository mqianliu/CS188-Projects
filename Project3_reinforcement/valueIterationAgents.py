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
        states = self.mdp.getStates()
        for i in range(0, self.iterations):
            tempUpdate = self.values.copy()
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                actions = self.mdp.getPossibleActions(state)
                if len(actions) == 0:
                    continue
                temp = []
                for action in actions:
                    sum_ = self.computeQValueFromValues(state, action)
                    temp.append(sum_)
                tempUpdate[state] = max(temp)
            for state in states:
                self.values[state] = tempUpdate[state]

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
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        sum_ = 0
        for transition in transitions:
            nextState, prob = transition
            nextValue = self.getValue(nextState)
            reward = self.mdp.getReward(state, action, nextState)
            sum_ += prob * (reward + self.discount * nextValue)
        return sum_

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
        if len(actions) == 0:
            return None
        max_ = -9999
        maxIndex = 0
        for i in range(len(actions)):
            val = self.computeQValueFromValues(state, actions[i])
            if val > max_:
                max_ = val
                maxIndex = i

        return actions[maxIndex]

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
        states = self.mdp.getStates()
        size = len(states)
        for i in range(0, self.iterations):
            tempUpdate = self.values.copy()
            state = states[i%size]
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            if len(actions) == 0:
                continue
            temp = []
            for action in actions:
                sum_ = self.computeQValueFromValues(state, action)
                temp.append(sum_)
            tempUpdate[state] = max(temp)

            self.values[state] = tempUpdate[state]



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

    def computeDiff(self, state):
        max_ = -9999
        actions = self.mdp.getPossibleActions(state)
        currVal = self.getValue(state)
        for action in actions:
            val = self.computeQValueFromValues(state, action)
            if val > max_:
                max_ = val
        diff = abs(currVal - max_)
        return diff

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessorsList = util.Counter()
        states = self.mdp.getStates()
        queue = util.PriorityQueue()
        for state in states:
            predecessorsList[state] = set()
        # Computer predecessors of all states
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            if len(actions) == 0:
                continue
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for transition in transitions:
                    nextState, prob = transition
                    if prob != 0:
                        predecessorsList[nextState].add(state)

        for state in states:
            # print(predecessorsList[state])
            if self.mdp.isTerminal(state):
                continue
            diff = self.computeDiff(state)
            queue.update(state, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                return
            s = queue.pop()

            # Value Update
            actions = self.mdp.getPossibleActions(s)
            if len(actions) == 0:
                continue
            temp = []
            for action in actions:
                sum_ = self.computeQValueFromValues(s, action)
                temp.append(sum_)
            self.values[s] = max(temp)

            for p in predecessorsList[s]:
                diff = self.computeDiff(p)
                if diff > self.theta:
                    queue.update(p, -diff)

