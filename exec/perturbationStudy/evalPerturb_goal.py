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
    def __init__(self, agentID):
        self.agentID = agentID
        self.getPercentChange = lambda actionAtDim, actionPerturbedAtDim: abs((actionPerturbedAtDim - actionAtDim)/actionAtDim) * 100 if actionAtDim!= 0 else 100*(actionAtDim!=actionPerturbedAtDim)
        self.rounding = 3
        self.getActions = lambda timestepInfo: timestepInfo[1]
        self.getActionsPerturbed = lambda timestepInfo: timestepInfo[2]
        # self.getActions = lambda timestepInfo: np.array(timestepInfo[3]) - np.array(timestepInfo[0])
        # self.getActionsPerturbed = lambda timestepInfo: np.array(timestepInfo[4]) - np.array(timestepInfo[0])

    def __call__(self, trajectory):
        timeStepPercentChange = []
        for timestepInfo in trajectory:
            actionAtX = np.round(self.getActions(timestepInfo)[self.agentID][0], self.rounding)
            actionPerturbedAtX = np.round(self.getActionsPerturbed(timestepInfo)[self.agentID][0], self.rounding)
            percentChangeX = self.getPercentChange(actionAtX, actionPerturbedAtX)

            actionAtY = np.round(self.getActions(timestepInfo)[self.agentID][1], self.rounding)
            actionPerturbedAtY = np.round(self.getActionsPerturbed(timestepInfo)[self.agentID][1], self.rounding)
            percentChangeY = self.getPercentChange(actionAtY, actionPerturbedAtY)

            timeStepPercentChange.append(np.mean([percentChangeX, percentChangeY]))
        meanTimeStepPercentChange = np.mean(timeStepPercentChange)

        return meanTimeStepPercentChange


def evalPerturbationPercent(df):
    # numWolves = df.index.get_level_values('numWolves')[0]
    # numSheep = df.index.get_level_values('numSheep')[0]
    # # wolfType = 'sharedAgencyByIndividualRewardWolf'
    # # sheepConcern = 'selfSheep'
    # wolfType = df.index.get_level_values('wolfType')[0]
    # sheepConcern = df.index.get_level_values('sheepConcern')[0]
    # perturbAgentID = df.index.get_level_values('perturbAgentID')[0]
    # perturbQuantile = df.index.get_level_values('perturbQuantile')[0]
    # perturbDim = df.index.get_level_values('perturbDim')[0]

    interestedAgentID = df.index.get_level_values('interestedAgentID')[0]
    varsNotInTraj = ['interestedAgentID']


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateGoalPerturbation', 'trajectories')
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

    calculatePercentageActionChange = CalculatePercentageActionChange(interestedAgentID)
    allMeasurements = np.array([calculatePercentageActionChange(trajectory) for trajectory in allTrajectories])
    measurementMean = np.mean(allMeasurements, axis=0)
    measurementSe = np.std(allMeasurements, axis=0) / np.sqrt(len(allTrajectories) - 1)

    return pd.Series({'mean': measurementMean, 'se': measurementSe})





def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [2] # [1, 2, 4]
    manipulatedVariables['wolfType'] = ['individualReward', 'sharedReward']
    manipulatedVariables['sheepConcern'] = ['selfSheep']
    manipulatedVariables['perturbedWolfID'] = [0]#[0, 1, 2]
    manipulatedVariables['interestedAgentID'] = [0, 1, 2]



    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    statisticsDf = toSplitFrame.groupby(levelNames).apply(evalPerturbationPercent)
    print(statisticsDf)
    # #
    # resultPath = os.path.join(dirName, '..', 'evalResults')
    # if not os.path.exists(resultPath):
    #     os.makedirs(resultPath)
    #
    # resultLoc = os.path.join(resultPath, 'evalPerturbResult.pkl')
    # # saveToPickle(statisticsDf, resultLoc)
    #
    # statisticsDf = loadFromPickle(resultLoc)
    #
    # toSplitFrame = statisticsDf.groupby(['numWolves', 'numSheep', 'sheepConcern', 'wolfType', 'perturbAgentID', 'perturbQuantile',
    #      'interestedAgentID']).apply(np.mean)
    # resultDF = toSplitFrame[toSplitFrame.index.get_level_values('perturbAgentID') != toSplitFrame.index.get_level_values('interestedAgentID')]



    # figure = plt.figure(figsize=(10, 10))
    # plotCounter = 1
    #
    # numRows = len(manipulatedVariables['numSheep'])#
    # numColumns = len(manipulatedVariables['perturbAgentID'])
    #
    # for key, outmostSubDf in resultDF.groupby('numSheep'):#
    #     outmostSubDf.index = outmostSubDf.index.droplevel('numSheep')#
    #     for keyCol, outterSubDf in outmostSubDf.groupby('perturbAgentID'):
    #         outterSubDf.index = outterSubDf.index.droplevel('perturbAgentID')
    #         axForDraw = figure.add_subplot(numRows, numColumns, plotCounter)
    #         for keyRow, innerSubDf in outterSubDf.groupby('interestedAgentID'):
    #             innerSubDf.index = innerSubDf.index.droplevel('interestedAgentID')
    #             innerSubDf.plot.line(ax = axForDraw, y='mean', label = keyRow)
    #             if plotCounter <= numColumns:
    #                 axForDraw.title.set_text('Perturbed Agent = ' + str(keyCol) )
    #             if plotCounter% numColumns == 1:
    #                 axForDraw.set_ylabel('Number of Sheep = ' + str(key))
    #             axForDraw.set_xlabel('Perturbation Quantile')
    #         # print(innerSubDf)
    #         plotCounter += 1
    #         # plt.xlim([0.3, 0.7])
    #         if key == 1:
    #             plt.ylim([10, 25])
    #         elif key == 2:
    #             plt.ylim([30, 140])
    #         else:
    #             plt.ylim([80, 165])
    #         plt.xticks([0, 1, 2, 3, 4], manipulatedVariables['perturbQuantile'])#[0.3, 0.4, 0.5, 0.6, 0.7]
    #         plt.legend(title='Effect on Agent', title_fontsize = 8, prop={'size': 8})
    #
    # # figure.text(x=0.03, y=0.5, s='Mean Episode Kill', ha='center', va='center', rotation=90)
    # # plt.suptitle('MADDPG Evaluate predatorSelfishness/ preySpeed/ actionCost')
    # # plt.savefig(os.path.join(resultPath, 'evalRewardWithKillProbAndDistSensitiveNoBiteRewKillInfo_killNum_allcond_regroup'))
    # plt.show()
    # # plt.close()

if __name__ == '__main__':
    main()
