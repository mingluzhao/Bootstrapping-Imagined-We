import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

from collections import OrderedDict
import itertools as it
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.MDPChasing.envMADDPG import *
from src.algorithms.MADDPG.myMADDPG import *
from src.visualization.visualizeEnvMADDPG import *
from src.MDPChasing.trajectory import ForwardOneStep, SampleTrajectory
from src.mathTools.distribution import sampleFromDistribution, BuildGaussianFixCov, sampleFromContinuousSpace
from src.neuralNetwork.policyValueResNet import restoreVariables
from src.generateAction.imaginedWeSampleAction import SampleActionOnFixedIntention
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, saveToPickle

# Perturbed group of wolves has one of them always chasing for sheep 0 (they can only observe sheep 0 here)


class SampleTrajectoryWithPerturbation:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep

    def __call__(self, sampleAction, samplePerturbedAction):
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()
        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None, None, None, 0))
                break

            state, actionPerturbed, nextStatePerturbed, rewardPerturbed = self.forwardOneStep(state, samplePerturbedAction)
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)

            trajectory.append((state, action, actionPerturbed, nextState, nextStatePerturbed, reward))
            state = nextState

        return trajectory


class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)

        numWolves = parameters['numWolves']
        numSheep = parameters['numSheep']
        numBlocks = 2
        wolfSelfish = 1.0 if parameters['wolfType'] == 'individualReward' else 0.0
        perturbedWolfID = parameters['perturbedWolfID']

        ## MDP Env
        numAgents = numWolves + numSheep
        numEntities = numAgents + numBlocks
        wolvesID = list(range(numWolves))
        sheepsID = list(range(numWolves, numWolves + numSheep))
        blocksID = list(range(numAgents, numEntities))

        sheepSize = 0.05
        wolfSize = 0.075
        blockSize = 0.2
        entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheep + [blockSize] * numBlocks

        costActionRatio = 0.0
        sheepSpeedMultiplier = 1.0
        sheepMaxSpeed = 1.3 * sheepSpeedMultiplier
        wolfMaxSpeed = 1.0
        blockMaxSpeed = None

        entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheep + [blockMaxSpeed] * numBlocks
        entitiesMovableList = [True] * numAgents + [False] * numBlocks
        massList = [1.0] * numEntities

        collisionReward = 1 # for evaluation, count # of bites
        isCollision = IsCollision(getPosFromAgentState)
        rewardAllWolves = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, wolfSelfish)
        rewardWolf = lambda state, action, nextState: np.sum(rewardAllWolves(state, action, nextState))

        reshapeActionInTransit = lambda action: action
        getCollisionForce = GetCollisionForce()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
        transit = TransitMultiAgentChasing(numEntities, reshapeActionInTransit, applyActionForce, applyEnvironForce, integrateState)

        forwardOneStep = ForwardOneStep(transit, rewardWolf)

        reset = ResetMultiAgentChasing(numAgents, numBlocks)
        isTerminal = lambda state: False
        maxRunningStepsToSample = 101
        sampleTrajectoryWithPerturbation = SampleTrajectoryWithPerturbation(maxRunningStepsToSample, isTerminal, reset, forwardOneStep)

        ## MDP Policy
        worldDim = 2
        actionDim = worldDim * 2 + 1

        layerWidth = [128, 128]
        maxTimeStep = 75
        maxEpisode = 60000
        dirName = os.path.dirname(__file__)

        # ------------ sheep recover variables ------------------------
        numSheepToObserve = 1
        sheepModelListOfDiffWolfReward = []
        sheepTypeList = [0.0, 1.0]

        for sheepType in sheepTypeList:
            wolvesIDForSheepObserve = list(range(numWolves))
            sheepsIDForSheepObserve = list(range(numWolves, numSheepToObserve + numWolves))
            blocksIDForSheepObserve = list(range(numSheepToObserve + numWolves, numSheepToObserve + numWolves + numBlocks))
            observeOneAgentForSheep = lambda agentID: Observe(agentID, wolvesIDForSheepObserve, sheepsIDForSheepObserve, blocksIDForSheepObserve, getPosFromAgentState, getVelFromAgentState)
            observeSheep = lambda state: [observeOneAgentForSheep(agentID)(state) for agentID in range(numWolves + numSheepToObserve)]
           
            obsIDsForSheep = wolvesIDForSheepObserve + sheepsIDForSheepObserve + blocksIDForSheepObserve
            initObsForSheepParams = observeSheep(reset()[obsIDsForSheep])
            obsShapeSheep = [initObsForSheepParams[obsID].shape[0] for obsID in range(len(initObsForSheepParams))]
            
            buildSheepModels = BuildMADDPGModels(actionDim, numWolves + numSheepToObserve, obsShapeSheep)
            sheepModelsList = [buildSheepModels(layerWidth, agentID) for agentID in range(numWolves, numWolves + numSheepToObserve)]

            sheepFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
        numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, sheepType)
            sheepModelPaths = [os.path.join(dirName, '..', '..', 'data', 'preTrainModel', sheepFileName + str(i)) for i in range(numWolves, numWolves + numSheepToObserve)]
            [restoreVariables(model, path) for model, path in zip(sheepModelsList, sheepModelPaths)]
            sheepModelListOfDiffWolfReward = sheepModelListOfDiffWolfReward + sheepModelsList 
        
        # actOneStep = ActOneStep(actByPolicyTrainNoisy) #TODO
        actOneStep = ActOneStep(actByPolicyTrainNoNoisy)

        numAllSheepModels = len(sheepModelListOfDiffWolfReward)

        # ------------ wolves recover variables ------------------------

        # ------------ Recover one perturbed wolf for comparison -------
        numSheepForPerturbedWolf = 1
        wolvesIDForPerturbedWolf = wolvesID
        sheepsIDForPerturbedWolf = list(range(numWolves, numWolves + numSheepForPerturbedWolf))
        blocksIDForPerturbedWolf = list(range(numWolves + numSheep, numEntities)) # skip the unattended sheep id

        observeOneAgentForPerturbedWolf = lambda agentID: Observe(agentID, wolvesIDForPerturbedWolf, sheepsIDForPerturbedWolf,
                blocksIDForPerturbedWolf, getPosFromAgentState, getVelFromAgentState)
        observePerturbedWolf = lambda state: [observeOneAgentForPerturbedWolf(agentID)(state) for agentID in range(numWolves + numSheepForPerturbedWolf)]

        initObsForPerturbedWolfParams = observePerturbedWolf(reset())
        obsShapePerturbedWolf = [initObsForPerturbedWolfParams[obsID].shape[0] for obsID in range(len(initObsForPerturbedWolfParams))]
        buildPerturbedWolfModels = BuildMADDPGModels(actionDim, numWolves + numSheepForPerturbedWolf, obsShapePerturbedWolf)
        layerWidthForWolf = [128, 128]
        perturbedWolfModel = buildPerturbedWolfModels(layerWidthForWolf, perturbedWolfID)

        perturbedWolfFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
            numWolves, numSheepForPerturbedWolf, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, wolfSelfish)
        perturbedWolfModelPath = os.path.join(dirName, '..', '..', 'data', 'preTrainModel', perturbedWolfFileName + str(perturbedWolfID))
        restoreVariables(perturbedWolfModel, perturbedWolfModelPath)

        # ------------ Recover other wolves trained with multiple goals -------

        wolvesIDForWolfObserve = wolvesID
        sheepsIDForWolfObserve = sheepsID
        blocksIDForWolfObserve = blocksID
        observeOneAgentForWolf = lambda agentID: Observe(agentID, wolvesIDForWolfObserve, sheepsIDForWolfObserve, 
                blocksIDForWolfObserve, getPosFromAgentState, getVelFromAgentState)
        observeWolf = lambda state: [observeOneAgentForWolf(agentID)(state) for agentID in range(numWolves + numSheep)]

        obsIDsForWolf = wolvesIDForWolfObserve + sheepsIDForWolfObserve + blocksIDForWolfObserve
        initObsForWolfParams = observeWolf(reset()[obsIDsForWolf])
        obsShapeWolf = [initObsForWolfParams[obsID].shape[0] for obsID in range(len(initObsForWolfParams))]
        buildWolfModels = BuildMADDPGModels(actionDim, numWolves + numSheep, obsShapeWolf)
        layerWidthForWolf = [128, 128]
        wolfModelsList = [buildWolfModels(layerWidthForWolf, agentID) for agentID in range(numWolves)]

        wolfFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
            numWolves, numSheep, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio, wolfSelfish)
        wolfModelPaths = [os.path.join(dirName, '..', '..', 'data', 'preTrainModel', wolfFileName + str(i)) for i in range(numWolves)]
        [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]

        # ------------ compose  policy ---------------------
        actionDimReshaped = 2
        cov = [0.00000000001 ** 2 for _ in range(actionDimReshaped)]
        buildGaussian = BuildGaussianFixCov(cov)
        reshapeAction = ReshapeAction()

        # unperturbed policy
        composeWolfPolicy = lambda wolfModel: lambda state: sampleFromContinuousSpace(buildGaussian(
            tuple(reshapeAction(actOneStep(wolfModel, observeWolf(state))))))
        wolvesSampleActions = [composeWolfPolicy(wolfModel) for wolfModel in wolfModelsList]

        # perturbed policy
        composePerturbedWolfPolicy = lambda perturbedModel: lambda state: sampleFromContinuousSpace(buildGaussian(
            tuple(reshapeAction(actOneStep(perturbedModel, observePerturbedWolf(state))))))
        wolvesSampleActionsPerturbed = wolvesSampleActions.copy()
        wolvesSampleActionsPerturbed[perturbedWolfID] = composePerturbedWolfPolicy(perturbedWolfModel)

       
        trajectories = []
        for trajectoryId in range(self.numTrajectories):
            sheepModelsForPolicy = [sheepModelListOfDiffWolfReward[np.random.choice(numAllSheepModels)] for sheepId in sheepsID]
            composeSheepPolicy = lambda sheepModel : lambda state: {tuple(reshapeAction(actOneStep(sheepModel, observeSheep(state)))): 1}
            sheepChooseActionMethod = sampleFromDistribution
            sheepSampleActions = [SampleActionOnFixedIntention(selfId, wolvesID, composeSheepPolicy(sheepModel), sheepChooseActionMethod, blocksID)
                    for selfId, sheepModel in zip(sheepsID, sheepModelsForPolicy)]

            allIndividualSampleActions = wolvesSampleActions + sheepSampleActions
            sampleAction = lambda state: [sampleIndividualAction(state) for sampleIndividualAction in allIndividualSampleActions]
            allIndividualSampleActionsPerturbed = wolvesSampleActionsPerturbed + sheepSampleActions
            sampleActionPerturbed = lambda state: [sampleIndividualAction(state) for sampleIndividualAction in allIndividualSampleActionsPerturbed]

            trajectory = sampleTrajectoryWithPerturbation(sampleAction, sampleActionPerturbed)
            trajectories.append(trajectory)

        trajectoryFixedParameters = {'maxRunningStepsToSample': maxRunningStepsToSample}
        self.saveTrajectoryByParameters(trajectories, trajectoryFixedParameters, parameters)




def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [2] # [1, 2, 4]
    manipulatedVariables['wolfType'] = ['individualReward', 'sharedReward']
    manipulatedVariables['sheepConcern'] = ['selfSheep']
    manipulatedVariables['perturbedWolfID'] = [0] # [0, 1, 2]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateGoalPerturbation', 'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = lambda trajectoryFixedParameters: GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, trajectoryFixedParameters, parameters: saveToPickle(trajectories, getTrajectorySavePath(trajectoryFixedParameters)(parameters))
   
    numTrajectories = 50# 200
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]


if __name__ == '__main__':
    main()
