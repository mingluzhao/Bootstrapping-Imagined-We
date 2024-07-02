# Bootstrapping Imagined We
Source code for Tang, Ning, Stephanie Stacy, Minglu Zhao, Gabriel Marquez, and Tao Gao. "Bootstrapping an Imagined We for Cooperation." In CogSci. 2020.

### Required Packages

* [gym 0.10.5](http://gym.openai.com/docs/)
* [tensorflow 1.13.1](https://www.tensorflow.org/install/pip)
* Numpy 1.16.4
* Pandas 0.24.2


### To generate results

1. Generate models
    * Specify conditions at [runMADDPGchasing.py](https://github.com/mingluzhao/Bootstrapping-Imagined-We/blob/master/exec/evaluateHierarchyPlanningEnvMADDPG/runMADDPGchasing.py)
    * Run the file and get trained models at [preTrainModel](https://github.com/mingluzhao/Bootstrapping-Imagined-We/tree/master/data/preTrainModel) 
2. Generate trajectories
    * Run [sampleTrajectoryNoSharedAgency.py](https://github.com/mingluzhao/Bootstrapping-Imagined-We/blob/master/exec/evaluateHierarchyPlanningEnvMADDPG/sampleTrajectoryNoSharedAgency.py) and [sampleTrajectorySharedAgency.py](https://github.com/mingluzhao/Bootstrapping-Imagined-We/blob/master/exec/evaluateHierarchyPlanningEnvMADDPG/sampleTrajectorySharedAngecy.py)
    * Trajectories are saved to [trajectories](https://github.com/mingluzhao/Bootstrapping-Imagined-We/tree/master/data/evaluateHierarchyPlanningEnvMADDPG/trajectories)
    
3. Generate evaluation graph
    * Run [evaluateWolvesType.py](https://github.com/mingluzhao/Bootstrapping-Imagined-We/blob/master/exec/evaluateHierarchyPlanningEnvMADDPG/evaluateWolvesType.py) 
    * Evaluation graph is saved to [evalResult](https://github.com/mingluzhao/Bootstrapping-Imagined-We/tree/master/exec/evalResult)
    
4. (optional) Generate demos if needed
    * Specify conditions at [generateDemo.py](https://github.com/mingluzhao/Bootstrapping-Imagined-We/blob/master/exec/evaluateHierarchyPlanningEnvMADDPG/generateDemo.py) and run the code
