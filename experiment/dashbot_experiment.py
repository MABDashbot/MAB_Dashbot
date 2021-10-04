from implem.dashbot import *
from implem.modules.panel import *
from experiment.eval import *

import random

import os
import numpy as np

import sys

class Experiment(DashBot):

    def __init__(self, i_expe, algos, explication_ratio, epsilon, exploration_type, exploration_distance, n_attributes, target,\
                inclusion=True, history=True, verbose=False):
        
        """
        i_expe : Int - Id of the experiment.
        algos : String - Name of the algo from [Random, Egreedy, MAB, Explanation]. If Algo2 existsn it's 'Explanation' (Algo1/Algo2).
        explication_ratio : String - Ratio of using each algo in Algos. e.g : '4/1' using Algo1 4 times and Algo2 1 time.
        epsilon : Float - probablity of Exploration/Exploitation
        exploration_type : String - Type of exploration. For Egreedy : ['Standard','NewPanel','Hybrid']. For MAB : ['Softmax','Standard','UCB'].
        exploration_distance : Float - Distance of Exploration in case where algo is Egreedy.
        n_attributes : Int - Number of attributes.
        target : String - The panel we seek.
        inclusion : Boolean - if True we use the inclusion.
        history : Boolean - if True we check history.
        verbose : Boolean.
        """

        DashBot.__init__(self, algos, explication_ratio, epsilon, exploration_type, exploration_distance, n_attributes, target, \
        inclusion, history, verbose)

        self.i_expe = i_expe
        self.numeric_ratio = None

        self.iter_algos = [] #Algo Scenario

        for algo, explor_ratio in zip(self.algos, self.explanation_ratios):
            self.iter_algos += [algo] * explor_ratio

        # data prerocessing
        self.preprocess_data()
        self.select_attributes()
        
        # initialize dashboard and target_dashboard
        self.instanciate_target_dashboard()

        # initialize result_file
        self.initialize_result_file()
        
        # eval
        self.eval = Eval(self.result_file_path)

    def select_attributes(self):
        selected_attributes = []
        
        # select attributes on target dashboard
        for i in range(len(self.target_dashboard)):
            selected_attributes += self.target_dashboard[i]['groupBy']
            selected_attributes += list(self.target_dashboard[i]['aggregates'].keys())
        
        selected_attributes = list(set(selected_attributes))

        if '*' in selected_attributes:
            selected_attributes.remove('*')

        list_ = []
        #if self.n_attributes < 20:
        #    list_ = ['t','user_id']

        #choose equal number of numerical and non numerical attributes
        if self.numeric_ratio:
            n_num = self.n_attributes * self.numeric_ratio
            n_target_num = len([at for at in self.attributes if at.name in selected_attributes and at.is_numeric]) # Nb Numeric Attributes

            n_nom = self.n_attributes - n_num
            n_nom = n_nom - len(selected_attributes) + n_target_num
            n_num = n_num - n_target_num

            selected_attributes += random.sample([at.name for at in self.attributes if (at.name not in selected_attributes) and (at.is_numeric) and (at.name not in list_)], k=int(n_num))
            selected_attributes += random.sample([at.name for at in self.attributes if (at.name not in selected_attributes) and (not at.is_numeric)], k=int(n_nom))

            # if ratio*n_attributes is not int, select one randomly with acurate densities of probability
            num_proba = n_num - int(n_num)
            if num_proba != 0:
                num_or_nom = np.random.choice(['num', 'nom'], size=1, p=[num_proba, 1 - num_proba])
                if num_or_nom == 'num':
                    selected_attributes.append(random.choice([at.name for at in self.attributes if at.name not in selected_attributes and at.is_numeric]))
                else:
                    selected_attributes.append(random.choice([at.name for at in self.attributes if at.name not in selected_attributes and not at.is_numeric]))
        else:
            n_others = self.n_attributes - len(selected_attributes)
            
            if n_others < 0 :
                raise ValueError(f'Number of attributes ({self.n_attributes}) is supperior to the number of attributes in the panels ({len(selected_attributes)})')
            
            selected_attributes += random.sample( [at.name for at in self.attributes if (at.name not in selected_attributes) and (at.name not in list_)], k=n_others)
        # restrict attributes of instance experiment with selected attributes

        self.dataset = self.dataset[selected_attributes]

        self.ranking.preprocessed['groupBy'] = [at for at in self.ranking.preprocessed['groupBy'] if at.name in selected_attributes]
        self.ranking.preprocessed['aggregation'] = [at for at in self.ranking.preprocessed['aggregation'] if at.name in selected_attributes]
        self.ranking.preprocessed['aggregation'] += [self.star]

        self.attributes = [at for at in self.attributes if at.name in selected_attributes]
        self.attributes_and_star = self.attributes + [self.star]
        self.numeric_at = [at for at in self.attributes if at.is_numeric]

        if self.verbose == True:
            print(f"selection of {self.n_attributes} attributes: {[at.name for at in self.attributes]}")
        
        if (self.verbose == True) & (self.numeric_ratio is not None):
            print(f"including {int(100*len(self.numeric_at)/self.n_attributes)}% numeric attributes (asked:{int(self.numeric_ratio * 100)}%)\n")
        
        self.selected_attributes = selected_attributes

        self.ranking.calculate_general_ranking('groupBy', self.diversity)
        self.ranking.calculate_general_ranking('aggregation', self.diversity)

    def instanciate_target_dashboard(self):
        self.target = Dash(self.attributes_and_star)

        for index, panel in enumerate(self.target_dashboard):
            target_panel = Panel(self.attributes_and_star)

            for attribute in panel['groupBy']:
                target_panel.vector[attribute, 'groupBy'] = 2
            for attribute, functions in panel['aggregates'].items():
                for f in functions:
                    target_panel.vector[attribute, f] += 1
            
            self.target.add_panel(target_panel, index)
        
        self.found_panels = set()      # where index of found target panels will be stored

    def initialize_result_file(self):
        self.directory_path = f"experiment/results/"
        self.result_file_path = f"{self.directory_path}/{self.algos}_{self.explanation_ratios}_{self.exploration_type}_{self.n_attributes}_{self.i_expe}.csv"
        
        result_file_header = "#iteration,algo,#panel,#suggestion,distances,iteration_time,total_time,target_found,#generations,panel,si,ni"
        
        with open(self.result_file_path, 'w') as result_file:
            result_file.write(result_file_header)
    
    def evaluate_iteration(self):
        self.eval.evaluate_time()
        
        self.eval.evaluate_distances(self.panel.vector.to_numpy(), self.target.dict, self.found_panels)

        if self.inclusion:
            self.eval.target_found = self.is_panel_included_by_panel_in_target()

        if self.eval.target_found is not None:
            self.validate_panel()

        self.eval.write_iteration_in_result_storage(self.panel_number[0], self.panel_number[1], self.algo, self.generation_counter, self.panel.vector.to_numpy(),self.si,self.ni)

    def is_panel_included_by_panel_in_target(self):
        target_found = None

        panel_list = self.panel.vector.to_numpy()

        for index, target_panel_list in self.target.dict.items():
            product = np.sqrt(target_panel_list*panel_list)

            if np.array_equal(product,panel_list):
                target_found = index

                self.panel_number[1] += 1

                self.dashbot_history.append(target_panel_list)
                break

        return target_found

    def validate_panel(self):
        super().validate_panel()
        
        if self.algo=='MAB':
            self.si[self.MAB_explanation] += 1
        
        if self.verbose:
            print(f"\nPanel #{self.eval.target_found} found in {self.panel_number[1] + 1} iterations")
        
        self.found_panels.add(self.eval.target_found)

        del self.target.dict[self.eval.target_found]
        self.target.dataframe = self.target.dataframe.drop(index=[self.eval.target_found])

        for at in self.attributes:
            at.bad_groupBy = 0
            at.bad_func = list()

        self.ranking.calculate_general_ranking('groupBy', self.diversity)
        self.ranking.calculate_general_ranking('aggregation', self.diversity)

        self.new_panel_level = self.n_attributes-1
        self.last_combi = []
        self.last_quality = []

    def suggest_panel(self):
        if self.eval.target_found is not None:
            self.deal_with_good_panel()
        else:
            self.deal_with_bad_panel()
            
        self.show_to_U()

    def deal_with_good_panel(self):

        self.generation_counter = 0
        self.panel_number[0] += 1
        self.panel_number[1] = 0
        self.eval.target_found = False

        if self.algos[0] == 'Random':
            self.algo = 'Random'
            self.find_random_panel()
        else:
            self.algo = 'NewPanel'
            self.generate_new_panel()

    def deal_with_bad_panel(self):
        self.algo = self.iter_algos[self.panel_number[1]%len(self.iter_algos)]
        
        self.generation_counter = 0
        self.panel_number[1] += 1

        if self.algo == 'Random':
            self.find_random_panel()

        elif self.algo == 'Egreedy':
            self.perform_Egreedy()
        
        elif self.algo == 'MAB':
            self.perform_MAB()
            
        elif self.algo == 'Explanation':
            explanation = self.eval.evaluate_explanation(self.attributes_and_star, self.panel, self.target, self.found_panels, self.explanation_type)
            self.apply_explanation(explanation)
