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


def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSheep'] = [1, 2, 4]
    manipulatedVariables['wolfType'] = ['individualReward', 'sharedReward', 'sharedAgencyByIndividualRewardWolf']
    manipulatedVariables['sheepConcern'] = ['selfSheep']
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateHierarchyPlanningEnvMADDPG', 'trajectories')
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
    
    measureIntentionArcheivement = lambda df: lambda trajectory: np.sum(np.array(trajectory)[:, 3])
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf) 

    fig = plt.figure()
    fig.set_dpi(120)
    numColumns = len(manipulatedVariables['sheepConcern'])
    numRows = 1
    plotCounter = 1

    wolfTypeTable = {
        'individualReward': 'Individual Reward',
        'sharedReward': 'Shared Reward',
        'sharedAgencyBySharedRewardWolf': 'Shared Agency By Shared Reward Wolf',
        'sharedAgencyByIndividualRewardWolf': 'Shared Agency'}

    for key, group in statisticsDf.groupby(['sheepConcern']):
        group.index = group.index.droplevel(['sheepConcern'])
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)

        for wolfType, grp in group.groupby('wolfType'):
            grp.index = grp.index.droplevel('wolfType')
            grp.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', label = wolfTypeTable[wolfType], ylim = (0, 18), marker = 'o', rot = 0 )
            axForDraw.set_xlabel('Number of Biting Sheep')
            if plotCounter % numColumns == 1:
                axForDraw.set_ylabel('Number of Wolves = ' + str(key))
        plt.xticks(manipulatedVariables['numSheep'])
        plotCounter = plotCounter + 1
    fig.text(x=0.03, y=0.5, s='Mean Episode Bite', ha='center', va='center', rotation=90)

    plt.suptitle('Compare Shared Agency with Individual/Sharing Rewards')
    resultPath = os.path.join(DIRNAME, '..', 'evalResult')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    plt.savefig(os.path.join(resultPath, 'RewardMADDPG3WolfSharedAgencyVSIndividualRewardVSSharedReward'))
    plt.show()

if __name__ == '__main__':
    main()
