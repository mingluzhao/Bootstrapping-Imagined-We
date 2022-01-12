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
            actionAtInterestedDim = np.round(timestepInfo[1][self.agentID][self.interestedDim], 2)
            actionPerturbedAtInterestedDim = np.round(timestepInfo[2][self.agentID][self.interestedDim], 2)
            diff = actionPerturbedAtInterestedDim - actionAtInterestedDim
            percentChange = abs(diff/actionAtInterestedDim) *100 if actionAtInterestedDim!= 0 else 100
            timeStepPercentChange.append(percentChange)
        meanTimeStepPercentChange = np.mean(timeStepPercentChange)
        return meanTimeStepPercentChange


def evalPerturbationPercent(df):
    interestedDim = df.index.get_level_values('interestedDim')[0]
    interestedAgentID = df.index.get_level_values('interestedAgentID')[0]
    varsNotInTraj = ['interestedDim', 'interestedAgentID']


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluatePerturbation', 'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'numWolves': 3, 'maxRunningStepsToSample': 101}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)

    trajVariables = readParametersFromDf(df)
    [trajVariables.pop(var) for var in varsNotInTraj]
    allTrajectories = loadTrajectories(trajVariables)

    calculatePercentageActionChange = CalculatePercentageActionChange(interestedDim, interestedAgentID)
    allMeasurements = np.array([calculatePercentageActionChange(trajectory) for trajectory in allTrajectories])
    measurementMean = np.mean(allMeasurements, axis=0)
    measurementSe = np.std(allMeasurements, axis=0) / np.sqrt(len(allTrajectories) - 1)

    return pd.Series({'mean': measurementMean, 'se': measurementSe})





def main():
    load = 1

    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [1, 2, 4]
    manipulatedVariables['sheepConcern'] = ['selfSheep']
    manipulatedVariables['wolfType'] = ['sharedAgencyByIndividualRewardWolf']
    manipulatedVariables['perturbAgentID'] = [0, 1, 2]
    manipulatedVariables['perturbQuantile'] = [0.3, 0.4, 0.5, 0.6, 0.7]
    manipulatedVariables['perturbDim'] = [0, 1]
    manipulatedVariables['interestedDim'] = [0, 1]
    manipulatedVariables['interestedAgentID'] = [0, 1, 2]

    resultPath = os.path.join(dirName, '..', '..', 'evalResults')
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    resultLoc = os.path.join(resultPath, 'evalPerturbResult.pkl')

    if not load:
        levelNames = list(manipulatedVariables.keys())
        levelValues = list(manipulatedVariables.values())
        modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
        toSplitFrame = pd.DataFrame(index=modelIndex)

        statisticsDf = toSplitFrame.groupby(levelNames).apply(evalPerturbationPercent)
        print(statisticsDf)
        saveToPickle(statisticsDf, resultLoc)
    else:
        statisticsDf = loadFromPickle(resultLoc)
        # print(statisticsDf)

    toSplitFrame = statisticsDf.groupby(['perturbAgentID', 'perturbQuantile', 'interestedAgentID']).apply(np.mean)
    resultDF = toSplitFrame[toSplitFrame.index.get_level_values('perturbAgentID') != toSplitFrame.index.get_level_values('interestedAgentID')]
    print(resultDF)

    figure = plt.figure(figsize=(10, 10))
    plotCounter = 1

    numRows = len(manipulatedVariables['numSheep'])#
    numColumns = len(manipulatedVariables['perturbAgentID'])

    # for key, outmostSubDf in resultDF.groupby('numSheep'):#
    #     outmostSubDf.index = outmostSubDf.index.droplevel('numSheep')#
    #
    for keyCol, outterSubDf in resultDF.groupby('perturbAgentID'):
        outterSubDf.index = outterSubDf.index.droplevel('perturbAgentID')
        axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
        for keyRow, innerSubDf in outterSubDf.groupby('interestedAgentID'):
            innerSubDf.index = innerSubDf.index.droplevel('interestedAgentID')
            innerSubDf.plot.line(ax = axForDraw, y='mean', label = keyRow)
            if plotCounter <= numColumns:
                axForDraw.title.set_text('Perturbed Agent = ' + str(keyCol) )
            axForDraw.set_xlabel('Perturbation Quantile')
        # print(innerSubDf)
        plotCounter += 1
        # plt.xlim([0.3, 0.7])
        # if key == 1:
        #     plt.ylim([10, 25])
        # elif key == 2:
        #     plt.ylim([30, 140])
        # else:
        #     plt.ylim([80, 165])
        plt.xticks(manipulatedVariables['perturbQuantile'])#[0.3, 0.4, 0.5, 0.6, 0.7]
        plt.legend(title='Effect on Agent', title_fontsize = 8, prop={'size': 8})

    # figure.text(x=0.03, y=0.5, s='Mean Episode Kill', ha='center', va='center', rotation=90)
    # plt.suptitle('MADDPG Evaluate predatorSelfishness/ preySpeed/ actionCost')
    # plt.savefig(os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_killNum_allcond_regroup'))
    plt.show()
    # plt.close()











# computational biology


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
