## MAB_Dashbot

## Prerequisites
Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have python3  installed  
* You have installed requirements  `pip install -r requirements.txt`
* You have jupyter notebook installed

## Running
The script `expe.py` allows to reproduce experiments and will create result files for each case in folder `experiment/results`. Experiment cases are based on parameters of `expe.py` explained in the following :

* idx : int - Id of the experiment
* algos : string - Name of method(s) from {Random, Egreedy, MAB, Explanation} (Egreedy is for Semantic 1 and MAB for Semantic 2). In the case where several methods are used, the parameter takes the following form : 'Algo1/Algo2'.
* explication_ratio : string - Ratio of using each algorithm in alogs. In case where one algorithm is mentioned in algos, the explication_ratio must be 1. If several algorithms are mentioned, the explication_ratio takes the following form : 'x1/x2' using Algo1 x1 times and Algo2 x2 times.
* epsilon : float - Probablity of Exploration/Exploitation
* exploration_type : string - Type of exploration. For 'Egreedy' aglo , exploration_type is from {'Standard','NewPanel','Hybrid'} ('Standard' for Far_Panel, 'NewPanel' for New_Panel, 'Hybrid' for Alternating). For 'MAB' algo,  exploration_type is from {'Softmax','Standard','UCB','Thompson'} ('Standard' for e-greedy-E).
* exploration_distances : float - Distance of exploration in case where algo is Egreedy.
* nb_attr : int - Number of attributes.
* targets : string - The target panels we seek.
* inclusion : boolean - if True we use the inclusion.
* history : boolean - if True we check history.

The file `graphs.ipynb` allows to generate the figures based on result files in folder `experiment/results`. It is divided into three parts : 
* The first is for the evolution of time and #steps as function of #attributes.
* The second is for the evolution of F1-score as function of clicks and #steps.
* The third is for the evolution of time and #steps as function of #panels in target.

## Repository organization
MAB_Dashbot repository is organised as follows:
* /implem : directory of source code and cleaned (or preprocessed) dataset.
* /experiment/figures : directory of all figures used in the paper.
* /experiment/results : directory of experiment files used to generate figures of the paper.
