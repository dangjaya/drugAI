import numpy as np
from helper import *
from Models import *

# import pytorch packages
import torch.nn.functional as F
from torchtext import data
import torch
import torch.nn as nn
from torch.autograd import Variable

cacheModels                 = {}
outputs                     = []
reward_formula              = "isValid * (10) * qed * isLipinski"
nodeSelection               = "lastNodeAfterTermination"
nbr_simulations_per_rollout = 1

class MonteCarloTreeSearchNode():

    # Reward formula : isValid * cumulative of log probabilities of atoms

    def __init__(self, model,
                 currentAtoms,
                 encoderContext,
                 source_mask,
                 eos_tok,
                 opt,
                 parent=None,
                 cutOff=0.005,
                 nodeLogProb=0,
                 smileVocab=None,
                 rootNode=None,
                 verbose=True,
                 device='cpu',
                 nodeType='leaf'):
        # mark this node if this is a root node
        self.rootNode = self if parent is None else rootNode
        self.model = model
        self.currentAtoms = currentAtoms
        self.encoderContext = encoderContext
        self.source_mask = source_mask
        self.eos_token = eos_tok
        self.maxLen = opt.max_len
        self.opt = opt
        self.nodeLogProb = nodeLogProb
        self.cutOff = cutOff
        self.parent = parent
        self._number_of_visits = 0
        self._reward = float('-inf')
        self._untried_actions = None
        self.children = []
        self.verbose = verbose
        self.smileVocab = smileVocab
        self.nodeType = nodeType
        self.device   = device

    def rollout_policy(self,
                       possible_moves):
        # run discrete sampling with prob dictated by the prob. distribution derived from the decoder model
        probs = np.array([i['prob'] for i in possible_moves])
        # to ensure that the total probs sums up to 1
        scaledProbs = probs / probs.sum(axis=0, keepdims=1)
        nextAtom = np.random.choice(a=possible_moves,
                                    size=1,
                                    p=scaledProbs)[0]
        return nextAtom

    def rollout(self):
        current_rollout_state = self
        currentAtomList = self.currentAtoms.copy()
        probs = self.nodeLogProb
        if self.verbose:
            print("Roll-out begin after the current list of tree-nodes ", currentAtomList)
        while not current_rollout_state.is_terminal_node(currentAtomList):
            rolloutProbs = self.obtainNextAtom(currentAtomList)
            # To filter out atom whose probability is really miniscule ( also used to prune the tree ) to speed up the process
            possible_atoms = [{'childNode': i, 'prob': p} for i, p in rolloutProbs if p >= self.cutOff]
            nextAtom = self.rollout_policy(possible_atoms)
            currentAtomList.append(nextAtom['childNode'])
            probs = probs * nextAtom['prob']
        # it reaches the terminal node (  max_len or EOS )
        # and reward is calculated
        smile = index2SMILE(currentAtomList, self.eos_token, self.opt.max_len, self.smileVocab)
        isValid = validSMILES(smile)
        isLipinski = checkLipinski(smile)
        qed = computeQED(smile)
        reward = eval(reward_formula)
        if reward > 0:
            outputs.append({'smile': smile,
                            'atomList': currentAtomList,
                            'reward': reward})
        if self.verbose:
            print("Roll-out generates molecule with SMILE of ", smile)
            print("Reward is ", reward, " isValid : ", isValid, " Lipinski ", isLipinski, " QED ", qed)
        return reward

    def backpropagate(self, reward):
        self._number_of_visits += 1.
        self._reward = reward if self._reward == float('-inf') else self._reward + reward
        if self.parent:
            self.parent.backpropagate(reward)

    def obtainNextAtom(self, atomList):
        # reading the previously-predicted next character for performance enhancement
        if ','.join(str(atomList)) in cacheModels:
            return cacheModels[','.join(str(atomList))]
        else:
            atomLength = len(atomList)
            trg_mask = nopeak_mask(atomLength).to(self.device)
            outputs = torch.LongTensor([atomList]).to(self.device)
            out = self.model.to(self.device).out(self.model.decoder.to(self.device)(outputs,
                                                                          self.encoderContext,
                                                                          self.source_mask,
                                                                          trg_mask))
            probs = F.softmax(out, dim=-1)[:, -1].tolist()
            prob_output = [(index, eachProb) for index, eachProb in enumerate(probs[0])]
            cacheModels[','.join(str(atomList))] = prob_output
            return prob_output

    def expand(self, verbose=True):
        # Usage :
        # to find out what available action(s) for this given node, we just need to get the list of
        # possible children ( subsequent atom ) by running this node against the decoder model
        # Argument
        # cutOff : the cutoff - point to prune the 'tree' in order to speed up the performance
        if self._untried_actions is None:
            probs = self.obtainNextAtom(self.currentAtoms)
            # To filter out atom whose probability is really miniscule ( also used to prune the tree )
            self._untried_actions = [{'childNode': i, 'logProb': p} for i, p in probs if p > self.cutOff]
        # the next step is to select the last atom generated from running it against the model
        for eachChild in self._untried_actions:
            addingAtoms = self.currentAtoms + [eachChild['childNode']]
            child_node = MonteCarloTreeSearchNode(model=self.model,
                                                  parent=self,
                                                  currentAtoms=addingAtoms,
                                                  nodeLogProb=self.nodeLogProb * eachChild['logProb'],
                                                  encoderContext=self.encoderContext,
                                                  source_mask=self.source_mask,
                                                  eos_tok=self.eos_token,
                                                  opt=self.opt,
                                                  cutOff=self.cutOff,
                                                  smileVocab=self.smileVocab,
                                                  nodeType='leaf',
                                                  rootNode=self.rootNode,
                                                  device  = self.device,
                                                  verbose=self.verbose)
            self.children.append(child_node)
        # set the current node as non-leaf node
        self.nodeType = 'non-leaf'
        if self.verbose:
            print("Expanding into ")
            for eachChild in self.children:
                print(eachChild.currentAtoms)

    def ucb(self,
            c_param=2,
            simulations=0):
        # Compute UCB - Upper Confidence Bound to allow the system to perform both exploitation-and-exploration
        # compute mean reward
        exploitation = self._reward / self._number_of_visits if self._number_of_visits > 0 else float('inf')
        # compute exploration - term by looking into total simulations and the explorations that this node has done
        exploration = c_param * np.sqrt(
            np.log(simulations) / self._number_of_visits) if self._number_of_visits > 0 else float('inf')
        return exploitation + exploration

    def traverse(self, currentNode, global_simulations=0):
        if currentNode.nodeType == 'leaf':
            return currentNode
        elif len(currentNode.children) > 0:
            # Only traverse the best node
            bestNode, _, UCBs = self.bestChild(nodes=currentNode.children,
                                               simulations=global_simulations)
            if self.verbose:
                print("The best node is ", bestNode.currentAtoms)
                print("w/ UCB of ", UCBs)
            leafNode = currentNode.traverse(bestNode, global_simulations)
        return leafNode

    def bestChild(self,
                  nodes,
                  simulations):
        nodeUCBs = [eachLeaf.ucb(c_param=2, simulations=simulations) for eachLeaf in nodes]
        childNo = np.argmax(nodeUCBs)
        return nodes[childNo], childNo, np.max(nodeUCBs)

    def has_been_visited(self):
        return True if self._number_of_visits > 0 else False


    def is_terminal_node(self, atoms):
        ones = (torch.LongTensor([atoms]) == self.eos_token).nonzero()
        eosNotFound = len([vec[0] for vec in ones]) == 0
        return not eosNotFound or len(atoms) == self.maxLen


    def rollOutPropagate(self,
                         node=None,
                         nbr_simulations=100,
                         global_simulations=0):
        simulationNo = 0
        while simulationNo < nbr_simulations:
            if self.verbose:
                if simulationNo % 10 == 0:
                    print("Roll-out %d out of %d " % (simulationNo, nbr_simulations))
            reward = node.rollout()
            node.backpropagate(reward)
            simulationNo += 1
            global_simulations += 1


    def simulate(self):
        currentNode = self
        currentSMILE = ''
        nbr_rounds = 0
        global_sims = 0

        while (not currentNode.is_terminal_node(currentNode.currentAtoms) or not validSMILES(currentSMILE) or len(
                currentSMILE) < 2):
            if self.verbose:
                print(
                    "<<<<--------------- Round %d with %d total sims ----------------->>>>" % (nbr_rounds, global_sims))
            smi = index2SMILE(currentNode.currentAtoms,
                              self.eos_token,
                              self.maxLen,
                              self.smileVocab)
            if self.verbose:
                print("Start with ", smi)

            # Compile a list of leaf nodes ( they don't have to be on the same-level ) that are traversed from the root node
            # or currentNode for performance enhancement
            leafNode = self.traverse(self.rootNode, nbr_rounds)
            if self.verbose:
                print("Leaf node selected ", leafNode.currentAtoms, " reward : ", leafNode._reward)

            currentNode = leafNode
            currentSMILE = index2SMILE(currentNode.currentAtoms,
                                       self.eos_token,
                                       self.maxLen,
                                       self.smileVocab)
            # Check to see if the current node has 1) children 2) all of them have been explored / visited
            if currentNode.has_been_visited():
                currentNode.expand()
                if self.verbose:
                    print("Expanded current node with %d children" % (len(currentNode.children)))
                continue

            if self.verbose:
                print("Roll out ", leafNode.currentAtoms)
            self.rollOutPropagate(node=leafNode,
                                  nbr_simulations=nbr_simulations_per_rollout)
            nbr_rounds += nbr_simulations_per_rollout

        ## Since the stopping criteria has been met
        ## select the node based on selection strategy before pass it back to the caller
        if nodeSelection == "lastNodeAfterTermination":
            return currentNode.currentAtoms
        elif nodeSelection == "bestReward":
            # Returning the best molecule that has the highest reward
            if len(outputs) > 0:
                bestMol = sorted(outputs, key=lambda x: x['reward'])
                return bestMol[0]['atomList']
            else:
                return None  # currentNode.currentAtoms
        else:
            return currentNode.currentAtoms


def encoderOutput(sequence,
                  model,
                  opt,
                  device='cpu'):
    # outputs of encoder remain the same
    src_mask = (sequence != opt.src_pad).to(device)
    encoder_output = model.to(device).encoder(sequence[:, :opt.maxProtLen], src_mask[:, :opt.maxProtLen])
    return encoder_output, src_mask