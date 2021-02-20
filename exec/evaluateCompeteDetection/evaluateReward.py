import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import scipy.stats 
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp
import math 


from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, \
IsTerminal, InterpolateOneFrame, TransitWithTerminalCheckOfInterpolation
from src.MDPChasing.reward import RewardFunctionByTerminal
from src.MDPChasing.trajectory import ForwardOneStep, SampleTrajectory
from src.MDPChasing.policy import RandomPolicy, HeatSeekingDiscreteStochasticPolicy
from src.MDPChasing.state import getStateOrActionFirstPersonPerspective, getStateOrActionThirdPersonPerspective
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution
from src.mathTools.soft import SoftMax
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, ApproximateValue, restoreVariables
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.intention import CreateIntentionSpaceGivenSelfId, CalIntentionValueGivenState, AdjustIntentionPriorGivenValueOfState, UpdateIntention
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsPolicyLikelihood, InferOneStep
from src.generateAction.imaginedWeSampleAction import PolicyForUncommittedAgent, PolicyForCommittedAgent, GetActionFromJointActionDistribution, \
        HierarchyPolicyForCommittedAgent, SampleIndividualActionGivenIntention, \
        SampleActionOnChangableIntention, SampleActionOnFixedIntention, SampleActionMultiagent
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.sampleTrajectoryTools.evaluation import ComputeStatistics


def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [2] # temp just 2
    manipulatedVariables['numSheep'] = [1] # temp just 1
    manipulatedVariables['inferenceSoft'] = [1.0]
    manipulatedVariables['wolfPolicySoft'] = [2.5]
    manipulatedVariables['otherCompeteRate'] = [0.0, 0.5, 1.0] # 0 never compete, 1 always compete
    manipulatedVariables['competeDetectionRate'] = [0.0, 0.5] # 0 never detect compete, 1 only detect compete
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateCompeteDetection',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
   
    NNNumSimulations = 250
    maxRunningSteps = 40
    sheepPolicySoft = 1.0
    competePolicy = 'heatseeking'
    trajectoryFixedParameters = {'competePolicy': competePolicy, 'sheepPolicySoft': sheepPolicySoft, 
            'maxRunningSteps': maxRunningSteps, 'NNNumSimulations': NNNumSimulations}
    heatseekingPrecesion = 50
    trajectoryFixedParameters.update({'heatseekingPrecesion': heatseekingPrecesion})
    print(trajectoryFixedParameters)
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    
    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    maxSteps = 50
    measureFunction = lambda df: lambda trajectory: int(np.linalg.norm(trajectory[-1][0][0] - trajectory[-1][0][1]) <= 50) - 0.00 * min(len(trajectory), maxSteps)
    
    def measureSelfReward(trajectory):
        stateLastTimeStep = trajectory[-1][0]
        goalState, selfState, otherState = stateLastTimeStep
        selfReward = int(np.linalg.norm(goalState - selfState) <= 50)
        otherReward = int(np.linalg.norm(goalState - otherState) <= 50)

        intentionLastTimeStep = trajectory[-1][4]
        selfIntention, otherIntention = intentionLastTimeStep

        selfShare = False
        if (0, (1, 2)) in list(selfIntention):
            if selfIntention[(0, (1, 2))] >= 0.8:
                selfShare = True
        
        otherShare = False 
        if (0, (1, 2)) in list(otherIntention):
            if otherIntention[(0, (1, 2))] >= 0.8:
                otherShare = True
        
        if selfReward == 1 and selfShare:
            selfReward = 0.5
        if selfReward == 0 and otherShare:
            selfReward = 0.5

        selfCost = -0.02 * len(trajectory)
        selfFinalGot = selfReward + selfCost
        return selfFinalGot

    measureFunction = lambda df: lambda trajectory: measureSelfReward(trajectory)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf) 
    
    fig = plt.figure()
    numColumns = len(manipulatedVariables['inferenceSoft'])
    numRows = len(manipulatedVariables['wolfPolicySoft'])
    plotCounter = 1

    for key, group in statisticsDf.groupby(['numWolves', 'numSheep', 'wolfPolicySoft', 'inferenceSoft']):
        group.index = group.index.droplevel(['numWolves', 'numSheep', 'wolfPolicySoft', 'inferenceSoft'])
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_ylabel('Accumulated Reward')
        #axForDraw.set_ylabel(str(numWolves))
        
        #if plotCounter <= numColumns:
        #    axForDraw.set_title(str(key[3]) + 'inferenceSoft')
        competeDetectionLabels = ['Coorperation Only', 'Coorperation Or Competetion Detection', 'Competetion Only']
        for competeDetection, grp in group.groupby('competeDetectionRate'):
            grp.index = grp.index.droplevel('competeDetectionRate')
            grp.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', label = competeDetectionLabels[int(competeDetection * 2)], ylim = (-0.3, 0.4), marker = 'o', rot = 0 )
        axForDraw.xaxis.set_label_text('Other Wolf Competing Rate') 
        plotCounter = plotCounter + 1

    #plt.suptitle(str(competePolicy))
    plt.show()

if __name__ == '__main__':
    main()
