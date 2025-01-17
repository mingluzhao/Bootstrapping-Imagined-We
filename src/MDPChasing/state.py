import numpy as np
import bisect


class GetAgentsPositionsFromState:
    def __init__(self, agentIds, posIndex):
        self.agentIds = agentIds
        self.posIndex = posIndex

    def __call__(self, state):
        state = np.asarray(state)
        agentsStates = state[self.agentIds]
        agentsPositions = np.asarray([state[self.posIndex] for state in agentsStates])
        return agentsPositions


def getStateOrActionFirstPersonPerspective(stateOrAction, goalId, multiAgentsIds, selfId, blocksId = []):
    selfIndexInMultiAgents = list(multiAgentsIds).index(selfId)
    agentsIds = list(multiAgentsIds).copy()
    agentsIds.insert(0, agentsIds.pop(selfIndexInMultiAgents)) # [4, 5, 6], selfID = 1 --> [5, 4, 6]
    IdsRelative = agentsIds.copy()
    for Id in list(np.array([goalId]).flatten()):
        bisect.insort(IdsRelative, Id)
    Ids = list(IdsRelative) + list(blocksId)
    sortedIds = np.array(Ids)
    stateOrActionRelative = np.array(stateOrAction)[sortedIds]
    return stateOrActionRelative


def getStateOrActionThirdPersonPerspective(stateOrAction, goalId, multiAgentsIds, blocksId = []): #TODO: equals to having no change?
    IdsRelative = list(np.array([goalId]).flatten()) + list(multiAgentsIds)
    Ids = list(IdsRelative) + list(blocksId)
    sortedIds = np.array(np.sort(Ids))
    stateOrActionRelative = np.array(stateOrAction)[sortedIds]
    return stateOrActionRelative
