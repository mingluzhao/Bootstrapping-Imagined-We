import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

from collections import OrderedDict
import pandas as pd
import itertools as it
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.MDPChasing.envMADDPG import *
from src.algorithms.MADDPG.myMADDPG import *
from src.visualization.visualizeEnvMADDPG import *
from src.MDPChasing.trajectory import ForwardOneStep, SampleTrajectory
from src.MDPChasing.policy import RandomPolicy
from src.MDPChasing.state import getStateOrActionFirstPersonPerspective, getStateOrActionThirdPersonPerspective
from src.mathTools.distribution import sampleFromDistribution, SoftDistribution, BuildGaussianFixCov, \
    sampleFromContinuousSpace
from src.neuralNetwork.policyValueResNet import restoreVariables
from src.inference.percept import SampleNoisyAction, PerceptImaginedWeAction
from src.inference.intention import UpdateIntention
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsContinuousPolicyLikelihood, \
    InferOneStep
from src.generateAction.imaginedWeSampleAction import PolicyForUncommittedAgent, PolicyForCommittedAgent, \
    GetActionFromJointActionDistribution, \
    SampleIndividualActionGivenIntention, SampleActionOnChangableIntention, SampleActionOnFixedIntention, \
    SampleActionMultiagent
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, \
    GetObjectsValuesOfAttributes
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, saveToPickle


# Change sheep policy to be to act by no noisy actions

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


class ComposeCentralControlPolicyByGaussianOnDeterministicAction:
    def __init__(self, reshapeAction, observe, actOneStepOneModel, buildGaussian):
        self.reshapeAction = reshapeAction
        self.observe = observe
        self.actOneStepOneModel = actOneStepOneModel
        self.buildGaussian = buildGaussian

    def __call__(self, individualModels, numAgentsInWe):
        centralControlPolicy = lambda state: [self.buildGaussian(tuple(self.reshapeAction(
            self.actOneStepOneModel(individualModels[agentId], self.observe(state))))) for agentId in
            range(numAgentsInWe)]
        return centralControlPolicy



class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)

        valuePriorEndTime = -100
        deviationFor2DAction = 1.0
        rationalityBetaInInference = 1.0

        numWolves = parameters['numWolves']
        numSheep = parameters['numSheep']
        wolfType = parameters['wolfType']
        sheepConcern = parameters['sheepConcern']
        wolfSelfish = 0.0 if wolfType == 'sharedAgencyBySharedRewardWolf' else 1.0
        perturbedWolfID = parameters['perturbedWolfID']

        ## MDP Env
        numBlocks = 2
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

        collisionReward = 1  # for evaluation, count # of bites
        isCollision = IsCollision(getPosFromAgentState)
        rewardAllWolves = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, wolfSelfish)
        rewardWolf = lambda state, action, nextState: np.sum(rewardAllWolves(state, action, nextState))

        reshapeActionInTransit = lambda action: action
        getCollisionForce = GetCollisionForce()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce,
                                              getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList,
                                        getVelFromAgentState, getPosFromAgentState)
        transit = TransitMultiAgentChasing(numEntities, reshapeActionInTransit, applyActionForce, applyEnvironForce,
                                           integrateState)

        forwardOneStep = ForwardOneStep(transit, rewardWolf)

        reset = ResetMultiAgentChasing(numAgents, numBlocks)
        isTerminal = lambda state: False
        maxRunningStepsToSample = 101
        sampleTrajectory = SampleTrajectoryWithPerturbation(maxRunningStepsToSample, isTerminal, reset, forwardOneStep)

        ## MDP Policy
        worldDim = 2
        actionDim = worldDim * 2 + 1

        layerWidth = [128, 128]
        maxTimeStep = 75
        maxEpisode = 60000
        dirName = os.path.dirname(__file__)

        # ------------ sheep recover variables ------------------------
        sheepConcernSelfOnly = 1 if sheepConcern == 'selfSheep' else 0
        numSheepToObserve = 1
        sheepModelListOfDiffWolfReward = []
        sheepTypeList = [0.0, 1.0]

        for sheepType in sheepTypeList:
            wolvesIDForSheepObserve = list(range(numWolves))
            sheepsIDForSheepObserve = list(range(numWolves, numSheepToObserve + numWolves))
            blocksIDForSheepObserve = list(
                range(numSheepToObserve + numWolves, numSheepToObserve + numWolves + numBlocks))
            observeOneAgentForSheep = lambda agentID: Observe(agentID, wolvesIDForSheepObserve, sheepsIDForSheepObserve,
                                                              blocksIDForSheepObserve, getPosFromAgentState,
                                                              getVelFromAgentState)
            observeSheep = lambda state: [observeOneAgentForSheep(agentID)(state) for agentID in
                                          range(numWolves + numSheepToObserve)]

            obsIDsForSheep = wolvesIDForSheepObserve + sheepsIDForSheepObserve + blocksIDForSheepObserve
            initObsForSheepParams = observeSheep(reset()[obsIDsForSheep])
            obsShapeSheep = [initObsForSheepParams[obsID].shape[0] for obsID in range(len(initObsForSheepParams))]

            buildSheepModels = BuildMADDPGModels(actionDim, numWolves + numSheepToObserve, obsShapeSheep)
            sheepModelsList = [buildSheepModels(layerWidth, agentID) for agentID in
                               range(numWolves, numWolves + numSheepToObserve)]

            sheepFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
                numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
                sheepType)
            sheepModelPaths = [os.path.join(dirName, '..', '..', 'data', 'preTrainModel', sheepFileName + str(i)) for i
                               in range(numWolves, numWolves + numSheepToObserve)]
            [restoreVariables(model, path) for model, path in zip(sheepModelsList, sheepModelPaths)]
            sheepModelListOfDiffWolfReward = sheepModelListOfDiffWolfReward + sheepModelsList

        actOneStep = ActOneStep(actByPolicyTrainNoNoisy)
        numAllSheepModels = len(sheepModelListOfDiffWolfReward)

        # ------------ recover variables for "we" ------------------------
        numAgentsInWe = numWolves
        numSheepInWe = 1
        numBlocksForWe = numBlocks
        wolvesIDForWolfObserve = list(range(numAgentsInWe))
        sheepsIDForWolfObserve = list(range(numAgentsInWe, numSheepInWe + numAgentsInWe))
        blocksIDForWolfObserve = list(
            range(numSheepInWe + numAgentsInWe, numSheepInWe + numAgentsInWe + numBlocksForWe))

        observeOneAgentForWolf = lambda agentID: Observe(agentID, wolvesIDForWolfObserve, sheepsIDForWolfObserve,
                                                         blocksIDForWolfObserve, getPosFromAgentState,
                                                         getVelFromAgentState)
        observeWolf = lambda state: [observeOneAgentForWolf(agentID)(state) for agentID in
                                     range(numAgentsInWe + numSheepInWe)]

        obsIDsForWolf = wolvesIDForWolfObserve + sheepsIDForWolfObserve + blocksIDForWolfObserve
        initObsForWolfParams = observeWolf(reset()[obsIDsForWolf])
        obsShapeWolf = [initObsForWolfParams[obsID].shape[0] for obsID in range(len(initObsForWolfParams))]
        buildWolfModels = BuildMADDPGModels(actionDim, numAgentsInWe + numSheepInWe, obsShapeWolf)
        layerWidthForWolf = [128, 128]
        wolfModelsList = [buildWolfModels(layerWidthForWolf, agentID) for agentID in range(numAgentsInWe)]

        wolfFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost{}individ{}_agent".format(
            numWolves, numSheepInWe, numBlocks, maxEpisode, maxTimeStep, sheepSpeedMultiplier, costActionRatio,
            wolfSelfish)
        wolfModelPaths = [os.path.join(dirName, '..', '..', 'data', 'preTrainModel', wolfFileName + str(i)) for i in
                          range(numAgentsInWe)]
        [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]

        # ------------ compose wolves policy no perturbation ------------------------
        actionDimReshaped = 2
        cov = [deviationFor2DAction ** 2 for _ in range(actionDimReshaped)] # 1
        buildGaussian = BuildGaussianFixCov(cov)
        actOneStep = ActOneStep(actByPolicyTrainNoNoisy)
        reshapeAction = ReshapeAction()
        composeCentralControlPolicy = lambda observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction(
            reshapeAction, observe, actOneStep, buildGaussian)
        wolvesCentralControlPolicy = [composeCentralControlPolicy(observeWolf)(wolfModelsList, numAgentsInWe)]  # input state, return a list of gaussian distributions with cov 1

        softPolicyInInference = lambda distribution: distribution
        getStateThirdPersonPerspective = lambda state, goalId, weIds: getStateOrActionThirdPersonPerspective(state,
                                                                                                             goalId,
                                                                                                             weIds,
                                                                                                             blocksID)  # nochange
        policyForCommittedAgentsInInference = PolicyForCommittedAgent(wolvesCentralControlPolicy, softPolicyInInference,
                                                                      getStateThirdPersonPerspective) # same as wolvesCentralControlPolicy(state)
        concernedAgentsIds = wolvesID
        calCommittedAgentsPolicyLikelihood = CalCommittedAgentsContinuousPolicyLikelihood(concernedAgentsIds,
                                                                                          policyForCommittedAgentsInInference,
                                                                                          rationalityBetaInInference)

        randomActionSpace = [(5, 0), (3.5, 3.5), (0, 5), (-3.5, 3.5), (-5, 0), (-3.5, -3.5), (0, -5), (3.5, -3.5),
                             (0, 0)]
        randomPolicy = RandomPolicy(randomActionSpace)
        getStateFirstPersonPerspective = lambda state, goalId, weIds, selfId: getStateOrActionFirstPersonPerspective(
            state, goalId, weIds, selfId, blocksID)
        policyForUncommittedAgentsInInference = PolicyForUncommittedAgent(wolvesID, randomPolicy, softPolicyInInference,
                                                                          getStateFirstPersonPerspective)  # random policy, returns action distribution
        calUncommittedAgentsPolicyLikelihood = CalUncommittedAgentsPolicyLikelihood(wolvesID, concernedAgentsIds,
                                                                                    policyForUncommittedAgentsInInference)  # returns 1

        # Joint Likelihood
        calJointLikelihood = lambda intention, state, perceivedAction: calCommittedAgentsPolicyLikelihood(intention,
                                                                                                          state,
                                                                                                          perceivedAction) * \
                                                                       calUncommittedAgentsPolicyLikelihood(intention,
                                                                                                            state,
                                                                                                            perceivedAction)  # __* 1

        # ------------ wolves intention ------------------------
        intentionSpacesForAllWolves = [tuple(it.product(sheepsID, [tuple(wolvesID)])) for wolfId in
                                       wolvesID]  # <class 'tuple'>: ((3, (0, 1, 2)), (4, (0, 1, 2)), (5, (0, 1, 2)), (6, (0, 1, 2)))
        print('intentionSpacesForAllWolves', intentionSpacesForAllWolves)
        wolvesIntentionPriors = [
            {tuple(intention): 1 / len(allPossibleIntentionsOneWolf) for intention in allPossibleIntentionsOneWolf} for
            allPossibleIntentionsOneWolf in intentionSpacesForAllWolves]
        perceptSelfAction = SampleNoisyAction(deviationFor2DAction)
        perceptOtherAction = SampleNoisyAction(deviationFor2DAction)
        perceptAction = PerceptImaginedWeAction(wolvesID, perceptSelfAction,
                                                perceptOtherAction)  # input self, others action

        # Infer and update Intention
        variablesForAllWolves = [[intentionSpace] for intentionSpace in intentionSpacesForAllWolves]
        jointHypothesisSpaces = [pd.MultiIndex.from_product(variables, names=['intention']) for variables in
                                 variablesForAllWolves]
        concernedHypothesisVariable = ['intention']
        priorDecayRate = 1
        softPrior = SoftDistribution(priorDecayRate)  # no change
        inferIntentionOneStepList = [InferOneStep(jointHypothesisSpace, concernedHypothesisVariable,
                                                  calJointLikelihood, softPrior) for jointHypothesisSpace in
                                     jointHypothesisSpaces]

        if numSheep == 1:
            inferIntentionOneStepList = [lambda prior, state, action: prior] * 3

        adjustIntentionPriorGivenValueOfState = lambda state: 1
        chooseIntention = sampleFromDistribution
        updateIntentions = [UpdateIntention(intentionPrior, valuePriorEndTime, adjustIntentionPriorGivenValueOfState,
                                            perceptAction, inferIntentionOneStep, chooseIntention)
                            for intentionPrior, inferIntentionOneStep in
                            zip(wolvesIntentionPriors, inferIntentionOneStepList)]

        # reset intention and adjust intention prior attributes tools for multiple trajectory
        intentionResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
        intentionResetAttributeValues = [
            dict(zip(intentionResetAttributes, [0, None, None, intentionPrior, [intentionPrior]]))
            for intentionPrior in wolvesIntentionPriors]
        resetIntentions = ResetObjects(intentionResetAttributeValues, updateIntentions)
        returnAttributes = ['formerIntentionPriors']
        getIntentionDistributions = GetObjectsValuesOfAttributes(returnAttributes, updateIntentions)
        attributesToRecord = ['lastAction']
        recordActionForUpdateIntention = RecordValuesForObjects(attributesToRecord, updateIntentions)

        # Wovels Generate Action #TODO
        covForPlanning = [0.00000001 for _ in range(actionDimReshaped)]
        # covForPlanning = [0.03 ** 2 for _ in range(actionDimReshaped)]
        buildGaussianForPlanning = BuildGaussianFixCov(covForPlanning)
        composeCentralControlPolicyForPlanning = lambda \
                observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction(reshapeAction,
                                                                                    observe, actOneStep,
                                                                                    buildGaussianForPlanning)
        wolvesCentralControlPoliciesForPlanning = [
            composeCentralControlPolicyForPlanning(observeWolf)(wolfModelsList, numAgentsInWe)]

        centralControlPolicyListBasedOnNumAgentsInWeForPlanning = wolvesCentralControlPoliciesForPlanning  # 0 for two agents in We, 1 for three agents...
        softPolicyInPlanning = lambda distribution: distribution
        policyForCommittedAgentInPlanning = PolicyForCommittedAgent(
            centralControlPolicyListBasedOnNumAgentsInWeForPlanning, softPolicyInPlanning,
            getStateThirdPersonPerspective)

        policyForUncommittedAgentInPlanning = PolicyForUncommittedAgent(wolvesID, randomPolicy, softPolicyInPlanning,
                                                                        getStateFirstPersonPerspective)

        def wolfChooseActionMethod(individualContinuousDistributions):
            centralControlAction = tuple(
                [tuple(sampleFromContinuousSpace(distribution)) for distribution in individualContinuousDistributions])
            return centralControlAction

        getSelfActionIDInThirdPersonPerspective = lambda weIds, selfId: list(weIds).index(selfId)
        chooseCommittedAction = GetActionFromJointActionDistribution(wolfChooseActionMethod,
                                                                     getSelfActionIDInThirdPersonPerspective)
        chooseUncommittedAction = sampleFromDistribution
        wolvesSampleIndividualActionGivenIntentionList = [
            SampleIndividualActionGivenIntention(selfId, policyForCommittedAgentInPlanning,
                                                 policyForUncommittedAgentInPlanning, chooseCommittedAction,
                                                 chooseUncommittedAction)
            for selfId in wolvesID]

        # ------------------- recover one wolf model that only concerns sheep 0 -------------------
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


        # ------------------- Sample and Save Trajectory -------------------

        wolvesSampleActions = [
            SampleActionOnChangableIntention(updateIntention, wolvesSampleIndividualActionGivenIntention)
            for updateIntention, wolvesSampleIndividualActionGivenIntention in
            zip(updateIntentions, wolvesSampleIndividualActionGivenIntentionList)]

        perturbedWolfSampleActions = lambda state: tuple(reshapeAction(actOneStep(perturbedWolfModel, observePerturbedWolf(state))))
        wolvesSampleActionsPerturbed = wolvesSampleActions.copy()
        wolvesSampleActionsPerturbed[perturbedWolfID] = perturbedWolfSampleActions


        trajectoriesWithIntentionDists = []
        for trajectoryId in range(self.numTrajectories):
            sheepModelsForPolicy = [sheepModelListOfDiffWolfReward[np.random.choice(numAllSheepModels)] for sheepId in
                                    sheepsID]
            if sheepConcernSelfOnly:
                composeSheepPolicy = lambda sheepModel: lambda state: {
                    tuple(reshapeAction(actOneStep(sheepModel, observeSheep(state)))): 1}
                sheepChooseActionMethod = sampleFromDistribution
                sheepSampleActions = [SampleActionOnFixedIntention(selfId, wolvesID, composeSheepPolicy(sheepModel),
                                                                   sheepChooseActionMethod, blocksID)
                                      for selfId, sheepModel in zip(sheepsID, sheepModelsForPolicy)]
            else:
                composeSheepPolicy = lambda sheepModel: lambda state: tuple(
                    reshapeAction(actOneStep(sheepModel, observeSheep(state))))
                sheepSampleActions = [composeSheepPolicy(sheepModel) for sheepModel in sheepModelsForPolicy]

            allIndividualSampleActions = wolvesSampleActions + sheepSampleActions
            sampleActionMultiAgent = SampleActionMultiagent(allIndividualSampleActions, recordActionForUpdateIntention)

            allIndividualSampleActionsPerturbed = wolvesSampleActionsPerturbed + sheepSampleActions
            sampleActionMultiAgentPerturbed = SampleActionMultiagent(allIndividualSampleActionsPerturbed, recordActionForUpdateIntention)

            trajectory = sampleTrajectory(sampleActionMultiAgent, sampleActionMultiAgentPerturbed)
            intentionDistributions = getIntentionDistributions()
            trajectoryWithIntentionDists = [tuple(list(SASRPair) + list(intentionDist)) for SASRPair, intentionDist in
                                            zip(trajectory, intentionDistributions)]
            trajectoriesWithIntentionDists.append(tuple(trajectoryWithIntentionDists))
            resetIntentions()
        trajectoryFixedParameters = {'maxRunningStepsToSample': maxRunningStepsToSample}
        self.saveTrajectoryByParameters(trajectoriesWithIntentionDists, trajectoryFixedParameters, parameters)


def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [2] # [1, 2, 4]
    manipulatedVariables['sheepConcern'] = ['selfSheep']
    manipulatedVariables['wolfType'] = ['sharedAgencyByIndividualRewardWolf']
    manipulatedVariables['perturbedWolfID'] = [1, 2]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateGoalPerturbationSharedAgency',
                                       'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = lambda trajectoryFixedParameters: GetSavePath(trajectoryDirectory, trajectoryExtension,
                                                                          trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, trajectoryFixedParameters, parameters: saveToPickle(trajectories,
                                                                                                          getTrajectorySavePath(
                                                                                                              trajectoryFixedParameters)(
                                                                                                              parameters))
    numTrajectories = 50#100
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]


if __name__ == '__main__':
    main()
