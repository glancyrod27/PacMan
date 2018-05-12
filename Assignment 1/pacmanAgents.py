# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import scoreEvaluation
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # stores states , depth and action of leaf nodes
        queue1 = []
        # stores all the parent and child relations traversed
        predecessor = []
        # stores all the states and their respeective actions traversed
        all_actions = []
        # initialize queue with root state
        queue1.append((state,0,""))
        none_flag = False
        while queue1:
            temp1 = queue1.pop(0)
            depth = temp1[1]
            current_state = temp1[0]
            legal = current_state.getLegalPacmanActions()
            for action in legal:
                successor = current_state.generatePacmanSuccessor(action)
                if (successor == None):
                    none_flag = True
                    break
                if (successor.isLose()):
                    continue 
                if (successor.isWin()):
                    # If reached goal backtrace and return best action
                    Parent = current_state
                    Child = None
                    while Parent != state:
                        for i in range(0,int(predecessor.__len__())):
                            if (predecessor[i][0] == Parent):
                                Child = predecessor[i][0]
                                Parent = predecessor[i][1]
                    # Finding action of best state            
                    bestAction = [pair[1] for pair in all_actions if pair[0] == Child]
                    return bestAction[0]   
                else:
                    queue1.append((successor,depth+1, action))
                    predecessor.append((successor,current_state))
                    all_actions.append((successor, action))     
            if(none_flag):
                break
        scores = [(scoreEvaluation(current_state), depth, current_state) for current_state, depth, action in queue1]
        # Finding best scores best on score evaluation
        bestScore = max(scores, key=lambda item: item[0])[0]
        bestScores = [(scoreEvaluation(current_state), depth, current_state) for current_state, depth, action in queue1 if scoreEvaluation(current_state) == bestScore]
        # Choosing best state based on shallowest depth among all the bestscores
        bestState = min(bestScores, key=lambda item: item[1])[2]
        # Backtracking till we get child of root node
        Parent = bestState
        Child = None
        while Parent != state:
            for i in range(0,int(predecessor.__len__())):
              if (predecessor[i][0] == Parent):
                    Child = predecessor[i][0]
                    Parent = predecessor[i][1]          
        # Finding action of best state            
        bestAction = [pair[1] for pair in all_actions if pair[0] == Child]
        return bestAction[0]


class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # stores states , depth and action of leaf nodes
        stack1 = []
        # stores all the parent and child relations traversed
        predecessor = []
        # stores all the states and their respeective actions traversed
        all_actions = []
        # initialize stack with root state
        stack1.append((state,0,""))
        none_flag = False
        while stack1:
            temp1 = stack1.pop()
            depth = temp1[1]
            current_state = temp1[0]
            legal = current_state.getLegalPacmanActions()
            for action in legal:
                successor = current_state.generatePacmanSuccessor(action)
                if (successor == None):
                    none_flag = True
                    break
                if (successor.isLose()):
                    continue 
                if (successor.isWin()):
                    # If reached goal backtrace and return best action
                    Parent = current_state
                    Child = None
                    while Parent != state:
                        for i in range(0,int(predecessor.__len__())):
                            if (predecessor[i][0] == Parent):
                                Child = predecessor[i][0]
                                Parent = predecessor[i][1]
                    # Finding action of best state            
                    bestAction = [pair[1] for pair in all_actions if pair[0] == Child]
                    return bestAction[0] 
                else:
                    stack1.append((successor,depth+1, action))
                    predecessor.append((successor,current_state))
                    all_actions.append((successor, action))     
            if(none_flag):
                break
        scores = [(scoreEvaluation(current_state), depth, current_state) for current_state, depth, action in stack1]
        # Finding best scores best on score evaluation
        bestScore = max(scores, key=lambda item: item[0])[0]
        bestScores = [(scoreEvaluation(current_state), depth, current_state) for current_state, depth, action in stack1 if scoreEvaluation(current_state) == bestScore]
        # Choosing best state based on shallowest depth among all the bestscores
        bestState = min(bestScores, key=lambda item: item[1])[2]
        # Backtracking till we get child of root node
        Parent = bestState
        Child = None
        while Parent != state:
            for i in range(0,int(predecessor.__len__())):
                if (predecessor[i][0] == Parent):
                    Child = predecessor[i][0]
                    Parent = predecessor[i][1]             
        # Finding action of best state            
        bestAction = [pair[1] for pair in all_actions if pair[0] == Child]
        return bestAction[0]
        

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # stores states , depth and action of leaf nodes
        p_queue1 = []
        # stores all the parent and child relations traversed
        predecessor = []
        # stores all the states and their respeective actions traversed
        all_actions = []
        root_scoreEvaluation = scoreEvaluation(state)
        # initialize priority queue with root state
        p_queue1.append((0,state,0,""))
        none_flag = False
        while p_queue1:
            # Sorting priority queue
            p_queue1.sort(key=lambda tup: tup[0])
            temp1 = p_queue1.pop(0)
            depth = temp1[2]
            current_state = temp1[1]
            legal = current_state.getLegalPacmanActions()
            for action in legal:
                successor = current_state.generatePacmanSuccessor(action)
                if (successor == None):
                    none_flag = True
                    break
                if (successor.isLose()):
                    continue    
                if (successor.isWin()):
                    # If reached goal backtrace and return best action
                    Parent = current_state
                    Child = None
                    while Parent != state:
                        for i in range(0,int(predecessor.__len__())):
                            if (predecessor[i][0] == Parent):
                                Child = predecessor[i][0]
                                Parent = predecessor[i][1]
                    # Finding action of best state            
                    bestAction = [pair[1] for pair in all_actions if pair[0] == Child]
                    return bestAction[0] 
                else:
                    cost = (depth+1) - (scoreEvaluation(successor) - root_scoreEvaluation)
                    p_queue1.append((cost,successor,depth+1, action))
                    predecessor.append((successor,current_state))
                    all_actions.append((successor, action))     
            if(none_flag):
                break
        scores = [(scoreEvaluation(current_state), depth, current_state) for cost,current_state, depth, action in p_queue1]
        # Finding best scores best on score evaluation
        bestScore = max(scores, key=lambda item: item[0])[0]
        bestScores = [(score, depth, current_state) for score, depth, current_state in scores if score == bestScore]
        # Choosing best state based on shallowest depth among all the bestscores
        bestState = min(bestScores, key=lambda item: item[1])[2]
        # Backtracking till we get child of root node
        Parent = bestState
        Child = None
        while Parent != state:
            for i in range(0,int(predecessor.__len__())):
                if (predecessor[i][0] == Parent):
                    Child = predecessor[i][0]
                    Parent = predecessor[i][1]
        # Finding action of best state              
        bestAction = [pair[1] for pair in all_actions if pair[0] == Child]
        return bestAction[0]
