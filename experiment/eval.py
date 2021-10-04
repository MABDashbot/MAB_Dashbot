from implem.dashbot import *
from implem.modules.panel import *

import time

class Eval:
    def __init__(self, result_file_path):
        self.result_file_path = result_file_path
        self.file_storage = ""
        self.iteration_number = 0
        self.total_time = 0
        self.start_iteration_time = None
        self.end_iteration_time = None
        self.target_found = False

    def evaluate_time(self):
        self.iteration_time = self.end_iteration_time - self.start_iteration_time
        self.total_time = self.total_time + self.iteration_time

    def evaluate_distances(self, suggested_panel_list, target_dashboard_dict, found_panels):
        self.distances = dict()

        for index, target_panel in target_dashboard_dict.items():
            if index in found_panels:
                self.distances[index] = None
            else:
                dist = self.distance(suggested_panel_list, target_panel)
                self.distances[index] = dist

    def distance(self, panel_1, panel_2):
        """
        Metric for distance between 2 panels
        option 1 : 1 for each different bit in vector representation
        option 2 : 1 for each different group By / 1 for each different aggregation (i.e. 1/4 for each different aggregate)
        """

        # #  OPTION 1  #
        # differences = list(map(lambda x,y: (x-y), panel_1, panel_2))
        # distance = len([diff for diff in differences if diff !=0])
        
        #  OPTION 2  #
        distance = ( ( (panel_1-panel_2)/2 )**2 ).sum()

        return distance
    
    def write_iteration_in_result_storage(self, panel_number, suggestion_number, algo, generation_counter, panel_list, si, ni):
        distances = sorted(list(self.distances.items()), key = lambda item: item[0])
        distances = [str(item[1]) for item in distances]
        distances = ('/').join(distances)
        
        self.file_storage += f'\n{self.iteration_number},{algo},{panel_number},{suggestion_number},{distances},{self.iteration_time},{self.total_time},{str(self.target_found)},{generation_counter},{panel_list},{si},{ni}'
        self.iteration_number += 1

    def end_experiment(self, message=""):
        self.file_storage += message
        
        with open(self.result_file_path, 'a') as result_file:
            result_file.write(self.file_storage)

    def evaluate_explanation(self, attributes_and_star, suggested_panel, target, found_panels, explanation_type):
        
        # find closest panel in target dashboard
        closest_panel = Panel(attributes_and_star)
        target_to_find = dict([(index, dist) for (index, dist) in self.distances.items() if index not in found_panels])

        closest_panel_index = min(target_to_find, key = target_to_find.get) 
        closest_panel.vector = target.dataframe.loc[closest_panel_index,:]
        closest_panel.vector_to_attributes()

        explanation = {'groupBy_to_add': [], 'aggregation_to_add': [], 'aggregation_to_remove': [] }

        #Groupbys in the target and not in the panel to add
        #explanation['groupBy_to_add'] = [at for at in closest_panel.groupBy if at not in suggested_panel.groupBy]

        #Groupbys in the panel and not in the target to remove
        explanation['groupBy_to_remove'] = [at for at in suggested_panel.groupBy if at not in closest_panel.groupBy]

        #Agg in the target and not in the panel to add
        #explanation['aggregation_to_add'] = [at for at in closest_panel.aggregates if at not in suggested_panel.aggregates]

        #Agg in the panel and not in the target to remove
        #explanation['aggregation_to_remove'] = [at for at in suggested_panel.aggregates if at not in closest_panel.aggregates]

        good_agg = []
        [good_agg.append((at,func)) for (at,values) in closest_panel.aggregates.items() for func in values ]

        func_change = []
        [func_change.append((at,func)) for (at,values) in suggested_panel.aggregates.items() for func in values ]

        func_change = list( (set(func_change) | set(good_agg)) - (set(func_change) & set(good_agg)) ) 

        explanation['functions_to_change'] = func_change
        
        return explanation
