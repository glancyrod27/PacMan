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
from heuristics import *
import random
import math

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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)


class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        # initializing actionlist of length 5
        self.actionList = []
        for i in range(0,5):
            self.actionList.append(Directions.STOP)
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        possible = state.getAllPossibleActions()
        action_to_take = Directions.STOP

        # Randomly selecting action sequence of length 5
        for i in range(0,5):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)]
        none_flag = False
        bestscore = 0

        while True:
            tempState = state

            for i in range(0, 5):
                if tempState.isWin() + tempState.isLose() == 0:
                    seqstate = tempState.generatePacmanSuccessor(self.actionList[i])
                    if seqstate == None:
                        none_flag = True
                        break
                    tempState = seqstate
                else:
                    break

            if none_flag == True:
                break

            # comparing for action sequence with highest score evaluation
            if (scoreEvaluation(tempState)>bestscore):
                bestscore = scoreEvaluation(tempState)
                action_to_take = self.actionList[0]

            # generating new sequence with 50% chance to be change
            for index,action in enumerate(self.actionList):
                probability = random.randint(0,100)
                if (probability>50):
                    self.actionList[index] = possible[random.randint(0,len(possible)-1)];

        # returning first action of sequence of highest score evaluation           
        return  action_to_take



class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return Directions.STOP

    # GetAction Function: Called with every frame
    def getAction(self, state):

        possible = state.getAllPossibleActions()
        population = []
        # creating population of size 8 having actionlist of length 5
        for i in range(0,8):
            chrom = []
            for i in range(0,5):
                chrom.append(possible[random.randint(0,len(possible)-1)])
            population.append(chrom)

        # building probability list to find parents randomly for mutation     
        probability = []
        for i in range(0,8):
            for j in range(0,i+1):
                    probability.append(j+1)

        none_flag = False
        while True:
            scores = []
            for member in population:
                tempState = state
                for i in range(0,5):
                    if tempState.isWin() + tempState.isLose() == 0:
                        seqstate = tempState.generatePacmanSuccessor(member[i])
                        if seqstate == None:
                            none_flag = True
                            break
                        tempState = seqstate
                    else:
                        break
                if none_flag == True:
                    break
                scores.append((scoreEvaluation(tempState),member))

            if none_flag == True:
                break

            # Finding ranks based on score by sorting scores    
            scores.sort(key=lambda x: x[0])
            sorted_scores = scores

            # Randomly choosing parents based on probability
            parent1 = scores[random.choice(probability)-1][1]
            parent2 = scores[random.choice(probability)-1][1]

            # for test less than 70% generating new children using crossover
            pr = random.randint(0,100)
            if (pr<=70):
                child1= self.crossOver(parent1,parent2)
                child2= self.crossOver(parent1,parent2)
                for index,member in enumerate(population):
                    if member == parent1:
                        population[index] = child1
                    if member == parent2:
                        population[index] = child2

            # Getting new population after mutation
            population=self.mutation(population,possible)

        # returning first action of sequence having highest score evaluation    
        action_to_take = sorted_scores.pop()
        return action_to_take[1][0]


    def crossOver(self,parent1,parent2):
        child = []
        for j in range(0,5):
            pr = random.randint(0,100)
            if(pr<=50):
                child.append(parent1[j])
            else:
                child.append(parent2[j])
        return child

    def mutation(self,population,possible):
        for member in population:
                pr= random.randint(0,100)
                if (pr <= 10):
                    member[random.randint(0,4)]=possible[random.randint(0,len(possible)-1)]
        return population



class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        #node class to store each node
        class Node():
            def __init__(self,action,parent):
                self.action = action
                self.children = []
                self.parent = parent
                self.reward = 0.0
                self.visits = 0

            def get_child(self,action):
                for child in self.children:
                    if child.action == action:
                        return child

            def set_child(self, action):
                child = Node(action,self)
                self.children.append(child)


            def update(node, reward):
                while node != None:
                    node.visits += 1
                    node.reward += reward
                    node = node.parent
                return

        
        None_flag = False
        self.rt = state
        Root_node = Node(None,None)
        Root_node.set_child(None)
        
        #traversing till we dont get none 
        while True:
            if None_flag == True: 
                break
            New_node = Root_node
            New_state = self.rt
            while True:
                max_uct = 0
                Start_over = False
                best_action = Directions.STOP
                actions = New_state.getLegalPacmanActions()
                for action in actions:
                    x = New_state.generatePacmanSuccessor(action)
                    if x == None:
                        None_Flag = True,
                        break
                    if (x.isWin() + x.isLose() == 0):
                        node1 = New_node.get_child(action)
                        if node1 == None:
                            New_node.set_child(action)
                            New_node.get_child(action).update(self.Rollout(x))
                            Start_over = True
                            break
                        else:
                            value = node1.reward / float(node1.visits) + math.sqrt(2 * math.log(node1.parent.visits) / float(node1.visits))
                            if (value>max_uct):
                                max_uct = value
                                best_action = action
                    else:
                        break

                
                if None_flag == True: 
                    break
                if Start_over == True: 
                    break
                New_state = New_state.generatePacmanSuccessor(best_action)
                if New_state == None:
                    None_flag = True
                    break
                New_node = New_node.get_child(best_action)


        temp = -1
        for node in Root_node.children:
            if node.visits>temp:
                action = node.action
        return action

    def Rollout(self,current1):
        current = current1
        for i in range(0, 5):
            if (current.isLose() + current.isWin() != 0):
                return normalizedScoreEvaluation(self.rt, current)
            else:
                action = random.choice(current.getAllPossibleActions())
                successors = current.generatePacmanSuccessor(action)
                if successors == None: break
                current = successors

        return normalizedScoreEvaluation(self.rt, current)
    
