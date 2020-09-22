# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        # print(newGhostStates)
        foodList = newFood.asList()
        ghostList = []
        for i in newGhostStates:
            ghostList.append(i.getPosition())
        minD = 9999
        for f in foodList:
            temp = util.manhattanDistance(f, newPos)
            if temp < minD:
                minD = temp
        minG = 9999
        for g in ghostList:
            temp = util.manhattanDistance(g, newPos)
            if temp < minG:
                minG = temp

        if len(foodList) == 0:
            return 1000
        if minG <= 1:
            return -1000

        # return successorGameState.getScore()
        return 0.001*minG - 15*len(foodList) - 0.01*minD


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        legalMoves = gameState.getLegalActions()
        agentNum = gameState.getNumAgents()
        # Choose one of the best actions
        successors = []
        scores = []
        for act in legalMoves:
            successors.append(gameState.generateSuccessor(0, act))
        for successor in successors:
            scores.append(self.minimax(successor, agentNum, 1, 0))
        # print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        return legalMoves[bestIndices[0]]


    def minimax(self, gameState, agentNum, player, depth):
        if player == 0:
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if player == 0:
            return self.max(gameState, agentNum, player, depth)
        if player > 0:
            return self.min(gameState, agentNum, player, depth)

    def max(self, gameState, agentNum, index, depth):
        val = -9999
        actions = gameState.getLegalActions(index)
        successors = []
        for act in actions:
            successors.append(gameState.generateSuccessor(index, act))
        for successor in successors:
            val = max(val, self.minimax(successor, agentNum, (index+1)%agentNum, depth))
        return val

    def min(self, gameState, agentNum, index, depth):
        val = 9999
        actions = gameState.getLegalActions(index)
        successors = []
        for act in actions:
            successors.append(gameState.generateSuccessor(index, act))
        for successor in successors:
            val = min(val, self.minimax(successor, agentNum, (index+1)%agentNum, depth))
        return val

action = 0

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # legalMoves = gameState.getLegalActions()
        agentNum = gameState.getNumAgents()
        self.minimax(gameState, agentNum, 0, -1, -9999, 9999)

        return action

    def minimax(self, gameState, agentNum, player, depth, alpha, beta):
        if player == 0:
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if player == 0:
            return self.max_(gameState, agentNum, player, depth, alpha, beta)
        if player > 0:
            return self.min_(gameState, agentNum, player, depth, alpha, beta)

    def max_(self, gameState, agentNum, index, depth, alpha, beta):
        val = -9999
        actions = gameState.getLegalActions(index)
        successors = []
        maxVal = -99999
        maxAct = 0
        for act in actions:
            successors.append(gameState.generateSuccessor(index, act))
            for successor in successors:
                val = max(val, self.minimax(successor, agentNum, (index+1)%agentNum, depth, alpha, beta))
                alpha = max(alpha, val)
                if alpha > beta:
                    return val
            successors.clear()
            if val > maxVal:
                maxVal = val
                maxAct = act
        global action
        action = maxAct
        return val

    def min_(self, gameState, agentNum, index, depth, alpha, beta):
        val = 9999
        actions = gameState.getLegalActions(index)
        successors = []
        for act in actions:
            successors.append(gameState.generateSuccessor(index, act))
            for successor in successors:
                val = min(val, self.minimax(successor, agentNum, (index+1)%agentNum, depth, alpha, beta))
                beta = min(beta, val)
                if alpha > beta:
                    return val
            successors.clear()
        return val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        agentNum = gameState.getNumAgents()
        # Choose one of the best actions
        successors = []
        scores = []
        for act in legalMoves:
            successors.append(gameState.generateSuccessor(0, act))
        for successor in successors:
            scores.append(self.minimax(successor, agentNum, 1, 0))
        # print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        return legalMoves[bestIndices[0]]

    def minimax(self, gameState, agentNum, player, depth):
        if player == 0:
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if player == 0:
            return self.max(gameState, agentNum, player, depth)
        if player > 0:
            return self.avg(gameState, agentNum, player, depth)

    def max(self, gameState, agentNum, index, depth):
        val = -9999
        actions = gameState.getLegalActions(index)
        successors = []
        for act in actions:
            successors.append(gameState.generateSuccessor(index, act))
        for successor in successors:
            val = max(val, self.minimax(successor, agentNum, (index+1)%agentNum, depth))
        return val

    def avg(self, gameState, agentNum, index, depth):
        sum = 0
        actions = gameState.getLegalActions(index)
        successors = []
        for act in actions:
            successors.append(gameState.generateSuccessor(index, act))
        for successor in successors:
            sum += self.minimax(successor, agentNum, (index+1)%agentNum, depth)
        return sum / len(successors)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()

    '''
    minG = 9999
    for g in ghostList:
        temp = util.manhattanDistance(g, pos)
        if temp < minG:
            minG = temp
        if minG <= 1:
        return -1000
    '''
    minD = 9999
    for f in foodList:
        temp = util.manhattanDistance(f, pos)
        if temp < minD:
            minD = temp

    return  currentGameState.getScore() + (1/minD)

# Abbreviation
better = betterEvaluationFunction
