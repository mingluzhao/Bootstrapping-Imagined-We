import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, loadFromPickle
from src.sampleTrajectoryTools.evaluation import ComputeStatistics
import itertools as it
from src.sampleTrajectoryTools.trajectoriesSaveLoad import saveToPickle

class CalculatePercentageActionChange:
    def __init__(self, interestedDim, agentID):
        self.interestedDim = interestedDim
        self.agentID = agentID

    def __call__(self, trajectory):
        timeStepPercentChange = []
        for timestepInfo in trajectory:
            actionAtInterestedDim = np.round(timestepInfo[1][self.agentID][self.interestedDim], 3)
            actionPerturbedAtInterestedDim = np.round(timestepInfo[2][self.agentID][self.interestedDim], 3)
            diff = actionPerturbedAtInterestedDim - actionAtInterestedDim
            if diff!= 0:
                percentChange = abs((actionPerturbedAtInterestedDim - actionAtInterestedDim)/actionAtInterestedDim *100) if actionAtInterestedDim!= 0 else 100
            else:
                percentChange = 0
            timeStepPercentChange.append(percentChange)

        # meanPercentChange = np.mean(trajectoriesPercentChangeMean)
        # sePercentChange = np.std(trajectoriesPercentChangeMean)/np.sqrt(len(trajectoriesPercentChangeMean)-1)
        meanTimeStepPercentChange = np.mean(timeStepPercentChange)
        return meanTimeStepPercentChange


def main():
    # manipulated variables
    # manipulatedVariables = OrderedDict()
    # manipulatedVariables['numWolves'] = [3]
    # manipulatedVariables['numSheep'] = [1, 2, 4]
    # manipulatedVariables['sheepConcern'] = ['selfSheep']
    # manipulatedVariables['wolfType'] = ['sharedAgencyByIndividualRewardWolf']
    # manipulatedVariables['perturbAgentID'] = [0, 1, 2]
    # manipulatedVariables['perturbQuantile'] = [0.3, 0.4, 0.5, 0.6, 0.7]
    # manipulatedVariables['perturbDim'] = [0, 1]

    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [1, 2, 4]
    manipulatedVariables['sheepConcern'] = ['selfSheep']
    manipulatedVariables['wolfType'] = ['sharedAgencyByIndividualRewardWolf']
    manipulatedVariables['perturbAgentID'] = [0, 1, 2]
    manipulatedVariables['perturbQuantile'] = [0.3, 0.4, 0.5, 0.6, 0.7]
    manipulatedVariables['perturbDim'] = [0, 1]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluatePerturbation', 'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
   
    maxRunningSteps = 101
    numWolves = 3
    trajectoryFixedParameters = {'numWolves': numWolves, 'maxRunningStepsToSample': maxRunningSteps}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
     
    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    calculatePercentageActionChange = CalculatePercentageActionChange(interestedDim=0, agentID=1)
    measureActionChange = lambda df: lambda trajectory: calculatePercentageActionChange(trajectory)

    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureActionChange)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)

    resultPath = os.path.join(dirName, '..', 'evalResults')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    resultLoc = os.path.join(resultPath, 'evalPerturbResult.pkl')
    saveToPickle(statisticsDf, resultLoc)

    print('saved to evalPerturbResult.pkl')


    # fig = plt.figure()
    # fig.set_dpi(120)
    # numColumns = len(manipulatedVariables['sheepConcern'])
    # numRows = 1
    # plotCounter = 1
    #
    # wolfTypeTable = {
    #     'individualReward': 'Individual Reward',
    #     'sharedReward': 'Shared Reward',
    #     'sharedAgencyBySharedRewardWolf': 'Shared Agency By Shared Reward Wolf',
    #     'sharedAgencyByIndividualRewardWolf': 'Shared Agency'}
    #
    # for key, group in statisticsDf.groupby(['sheepConcern']):
    #     group.index = group.index.droplevel(['sheepConcern'])
    #     axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    #
    #     for wolfType, grp in group.groupby('wolfType'):
    #         grp.index = grp.index.droplevel('wolfType')
    #         grp.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', label = wolfTypeTable[wolfType], ylim = (0, 18), marker = 'o', rot = 0 )
    #         axForDraw.set_xlabel('Number of Biting Sheep')
    #         if plotCounter % numColumns == 1:
    #             axForDraw.set_ylabel('Number of Wolves = ' + str(key))
    #     plt.xticks(manipulatedVariables['numSheep'])
    #     plotCounter = plotCounter + 1
    # fig.text(x=0.03, y=0.5, s='Mean Episode Bite', ha='center', va='center', rotation=90)
    #
    # plt.suptitle('Compare Shared Agency with Individual/Sharing Rewards')
    # resultPath = os.path.join(DIRNAME, '..', 'evalResult')
    # if not os.path.exists(resultPath):
    #     os.makedirs(resultPath)
    # plt.savefig(os.path.join(resultPath, 'RewardMADDPG3WolfSharedAgencyVSIndividualRewardVSSharedReward'))
    # plt.show()

if __name__ == '__main__':
    main()
