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

white = [255, 255, 255]
green = [0, 250, 0]
grey = [200, 200, 200]
blue = [40, 80, 200]
red = [255, 0, 0]


class UpdateColorSpace:
    def __init__(self, perturbedWolfID, wolvesID, colorSpace):
        self.perturbedWolfID = perturbedWolfID
        self.wolvesID = wolvesID
        self.wolvesWithInference = wolvesID.copy()
        self.wolvesWithInference.remove(self.perturbedWolfID)
        self.colorSpace = colorSpace

    def __call__(self, posterior):
        numWolvesWithInference = len(self.wolvesWithInference)
        originalColor = self.colorSpace[self.wolvesID[1]]
        intentionSpace = posterior[0].keys()

        colorRepresentProbability = [np.array(originalColor) * 2 * (1 - max(list(posterior[individualId].values())))+
             np.sum([np.array(self.colorSpace[intention[0]]) * max(0, 2 * (posterior[individualId][intention] - 1 / len(intentionSpace)))
                  for intention in list(intentionSpace)], axis=0)
             for individualId in list(range(numWolvesWithInference))
             ]

        updatedColorSpace = np.array(self.colorSpace.copy())
        updatedColorSpace[self.wolvesWithInference] = colorRepresentProbability
        return updatedColorSpace


class SoftPolicy:
    def __init__(self, softParameter):
        self.softParameter = softParameter

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        softenUnnormalizedProbabilities = np.array([np.power(probability, self.softParameter) for probability in list(actionDist.values())])
        softenNormalizedProbabilities = list(softenUnnormalizedProbabilities / np.sum(softenUnnormalizedProbabilities))
        softenActionDist = dict(zip(actions, softenNormalizedProbabilities))
        return softenActionDist

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateGoalPerturbationHighLevel', 'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    maxRunningSteps = 101
    trajectoryFixedParameters = {'maxRunningStepsToSample': maxRunningSteps}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    numWolves = 3
    numSheep = 2
    wolfType = 'sharedAgencyByIndividualRewardWolf' # sharedReward, sharedAgencyByIndividualRewardWolf
    perturbedWolfID = 0
    perturbedWolfGoalID = 0
    perturbed = 1

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

    wolfColor = white
    perturbedWolfColor = blue
    sheepColor = green
    perturbedSheepColor = blue
    blockColor = grey

    wolvesColor = [wolfColor] * numWolves
    wolvesColor[perturbedWolfID] = perturbedWolfColor if perturbed else wolfColor
    sheepColorList = [sheepColor] * numSheep
    sheepColorList[perturbedWolfGoalID] = perturbedSheepColor if perturbed else sheepColor
    circleColorSpace = wolvesColor + sheepColorList + [blockColor] * numBlocks

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
    softParameter = 0.05
    softFunction = SoftPolicy(softParameter)

    updateColorSpace = UpdateColorSpace(perturbedWolfID, list(range(numWolves)), circleColorSpace)
    updateColorSpaceByPosterior = lambda posterior: updateColorSpace([softFunction(individualPosterior) for individualPosterior in posterior])

    outsideCircleAgentIds = list(range(numWolves))
    outsideCircleColor = np.array([red] * numWolves)
    outsideCircleSize = int(wolfSize * 1.5)
    viewRatio = 1
    drawCircleOutside = DrawCircleOutsideEnvMADDPG(screen, viewRatio, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize)
    drawState = DrawStateEnvMADDPG(FPS, screen, viewRatio, circleColorSpace, circleSizeSpace, agentIdsToDraw, positionIndex,
                saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)

    # MDP Env
    interpolateState = None
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = 4
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep,
                                    posteriorIndexInTimeStep)

    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[list(range(5))]]


if __name__ == '__main__':
    main()