from implem.dashbot import *
from implem.modules.panel import *
from experiment.dashbot_experiment import *
from experiment.eval import *

import numpy as np
import pandas as pd
import time
import itertools
import sys
import os
import random

import multiprocessing

targets = [ 
{'groupBy': ['rating'], 'aggregates': {'*': ['count'], 'age': ['avg']}},
{'groupBy': ['gender'], 'aggregates': {'g_Comedy': ['sum']}}
]

n_expe = list(range(10)) #Number of runs for each method

epsilon = 0.1 # Epsilon of exploration/exploitation
exploration_distances = 0.2 #distance of exploration
ns_attributes = [4,6,8] #Number of attributes in the dataset

#Explanation-1, Explanation-2, Explanation-10
explication_ratios = ['9/1', '49/1', '99/1'] # with_Explanation
exploration_types = ['Standard'] # Type of semantic 1
algos = ['Egreedy/Explanation']
algos = list( itertools.product(*[algos,explication_ratios]) )
cases = list( itertools.product(*[n_expe,algos,exploration_types,ns_attributes]) )

#ep-greedy Far_Panel, New_Panel, Alternating_Pnale
explication_ratios = ['1'] # Without_Explanation
exploration_types = ['Standard','NewPanel','Hybrid'] # Type of semantic 1
algos = ['Egreedy']
algos = list( itertools.product(*[algos,explication_ratios]) )
cases += list( itertools.product(*[n_expe,algos,exploration_types,ns_attributes]) )

#Explanation-100
explication_ratios = ['1']
exploration_types = ['Standard'] # Type of semantic 1
algos = ['Explanation']
algos = list( itertools.product(*[algos,explication_ratios]) )
cases += list( itertools.product(*[n_expe,algos,exploration_types,ns_attributes]) )

#Semantic 2 : Thompson, e-greedy-E, UCB, Softmax
explication_ratios = ['1'] # Without_Explanation
exploration_types = ['UCB','Standard','Softmax','Thompson'] # Type of semantic 2
algos = ['MAB']
algos = list( itertools.product(*[algos,explication_ratios]) )
cases += list( itertools.product(*[n_expe,algos,exploration_types,ns_attributes]) )

def worker(case):

    idx = case[0]
    algos = case[1][0]
    explication_ratio = case[1][1]
    exploration_type = case[2]
    nb_attr = case[3]

    experiment = Experiment(idx, algos, explication_ratio, epsilon, exploration_type, exploration_distances ,\
     nb_attr, targets, inclusion=True, history=True, verbose=False)

    experiment.initialize_dashboard_generation()
    
    while len(experiment.found_panels) < len(targets):
        experiment.eval.start_iteration_time = time.time()
        experiment.suggest_panel()
        experiment.eval.end_iteration_time = time.time()
        experiment.evaluate_iteration()
    experiment.eval.end_experiment()

    print(f'---------------------------------- END : {algos} - {explication_ratio} - {exploration_type} - {nb_attr}')

pool = multiprocessing.Pool(multiprocessing.cpu_count())    
res = pool.map(worker, cases)
pool.close()
