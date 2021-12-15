import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pygame as pg
from pygame.color import THECOLORS

from src.visualization.drawDemo import DrawBackground, DrawCircleOutsideEnvMADDPG, DrawStateEnvMADDPG, ChaseTrialWithTraj
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, LoadTrajectories, loadFromPickle


def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateGoalPerturbationHighLevel', 'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    maxRunningSteps = 101
    trajectoryFixedParameters = {'maxRunningStepsToSample': maxRunningSteps}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    numWolves = 3
    numSheep = 2
    wolfType = 'sharedAgencyByIndividualRewardWolf' # sharedReward, sharedAgencyByIndividualRewardWolf
    perturbedWolfID = 0
    perturbedWolfGoalID = 0
    perturbed = 1
    # trajectoryParameters = {'numWolves': numWolves, 'numSheep': numSheep, 'wolfType': wolfType,
    #                         'perturbedWolfID': perturbedWolfID,
    #                         'sheepConcern': 'selfSheep'}

    if perturbed:
        trajectoryParameters = {'numWolves': numWolves, 'numSheep': numSheep, 'wolfType': wolfType, 'perturbedWolfID': perturbedWolfID,
                                'perturbedWolfGoalID': perturbedWolfGoalID}
    else:
        trajectoryParameters = {'numWolves': numWolves, 'numSheep': numSheep, 'wolfType': wolfType}

    trajectories = loadTrajectories(trajectoryParameters)
    # generate demo image
    screenWidth = 700
    screenHeight = 700
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 700]
    yBoundary = [0, 700]
    lineColor = THECOLORS['white']
    lineWidth = 4
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    
    FPS = 10
    numBlocks = 2
    perturbedWolfColor = [255,160,122]
    wolfColor = [255, 255, 255]
    sheepColor = [0, 250, 0]
    blockColor = [200, 200, 200]
    targetSheepColor = [0, 100, 0]

    wolvesColor = [wolfColor] * numWolves
    wolvesColor[perturbedWolfID] = perturbedWolfColor if perturbed else wolfColor
    sheepColorList = [sheepColor] * numSheep
    sheepColorList[perturbedWolfGoalID] = targetSheepColor  if perturbed else [sheepColor] * numSheep

    circleColorSpace = wolvesColor + sheepColorList + [blockColor] * numBlocks
    # viewRatio = 1.5
    # sheepSize = int(0.05 * screenWidth / (2 * viewRatio))
    # wolfSize = int(0.075 * screenWidth / (3 * viewRatio))
    # blockSize = int(0.2 * screenWidth / (3 * viewRatio))

    sheepSize = int(0.05 * screenWidth/2)
    wolfSize = int(0.075 * screenWidth/2)
    blockSize = int(0.2 * screenWidth/2)
    circleSizeSpace = [wolfSize] * numWolves + [sheepSize] * numSheep + [blockSize] * numBlocks
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numWolves + numSheep + numBlocks))

    saveImage = False
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str('forDemo')
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    imaginedWeIdsForInferenceSubject = list(range(numWolves))
    
    updateColorSpaceByPosterior = None
    outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
    outsideCircleColor = np.array([[255, 0, 0]] * numWolves)
    outsideCircleSize = int(wolfSize * 1.5)
    viewRatio = 1
    drawCircleOutside = DrawCircleOutsideEnvMADDPG(screen, viewRatio, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize)
    drawState = DrawStateEnvMADDPG(FPS, screen, viewRatio, circleColorSpace, circleSizeSpace, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
    
   # MDP Env
    interpolateState = None
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = None
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep, posteriorIndexInTimeStep)
    
    maxWolfPositions = np.array([max([max([max(abs(timeStep[0][wolfId][0]), abs(timeStep[0][wolfId][1]))
        for wolfId in range(numWolves)]) 
        for timeStep in trajectory])
        for trajectory in trajectories])
    flags = maxWolfPositions < 1.3 * viewRatio
    index = flags.nonzero()[0]
    # print(trajectories[0])
    # state, action, actionPerturbed, nextState, nextStatePerturbed, reward
    # [chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[[0, 2, 3]]]]
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[list(range(5))]]

if __name__ == '__main__':
    main()
