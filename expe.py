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
{'groupBy': ['rating'], 'aggregates': {'*': ['count'], 'age': ['min','max','avg']}},
{'groupBy': ['rating'], 'aggregates': {'*': ['count'], 'g_Comedy': ['min','max','avg']}},
{'groupBy': ['rating'], 'aggregates': {'*': ['count'], 't': ['min','max','avg']}},

{'groupBy': ['age'], 'aggregates': {'*': ['count'], 'rating': ['min','max','avg']}},
{'groupBy': ['age'], 'aggregates': {'*': ['count'], 'g_Comedy': ['min','max','avg']}},
{'groupBy': ['age'], 'aggregates': {'*': ['count'], 't': ['min','max','avg']}},

{'groupBy': ['g_Comedy'], 'aggregates': {'*': ['count'], 'rating': ['min','max','avg']}},
{'groupBy': ['g_Comedy'], 'aggregates': {'*': ['count'], 'age': ['min','max','avg']}},
{'groupBy': ['g_Comedy'], 'aggregates': {'*': ['count'], 't': ['min','max','avg']}},

{'groupBy': ['t'], 'aggregates': {'*': ['count'], 'rating': ['min','max','avg']}},
{'groupBy': ['t'], 'aggregates': {'*': ['count'], 'age': ['min','max','avg']}},
{'groupBy': ['t'], 'aggregates': {'*': ['count'], 'g_Comedy': ['min','max','avg']}},
]

n_expe = list(range(1))

ns_attributes = [4,6,8] #Number of attributes in the dataset
explication_ratios = [ '9/1', '49/1', '99/1'] # Without_Expe/Explication
exploration_types = ['UCB','Softmax','Standard','Thompson'] # Type of e_greedy

epsilon = 0.1 # Epsilon of exploration/exploitation
exploration_distances = 0.2 #distance of exploration

#algos = ['Explanation']

algos = list( itertools.product(*[algos,explication_ratios]) )
algos = [('MAB','1')]

cases = list( itertools.product(*[n_expe,algos,exploration_types,ns_attributes]) )

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





