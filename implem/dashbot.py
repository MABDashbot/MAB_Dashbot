import pandas as pd
import sys
import itertools
import random
import numpy as np

from .dataset.datasets import *
from .modules.attribute import *
from .modules.panel import *
from .modules.ranking import *
from .modules.data_preprocessor import *

semantics = ['Random', 'Egreedy', 'MAB', 'Explanation']
qualities = ['good','less_good','bad','very_bad']

class DashBot():

    def __init__(self, algos, explication_ratio, epsilon, exploration_type, exploration_distance, n_attributes, target,inclusion,\
                history, verbose):

        self.dataset_name = 'MovieLens_numeric'
        self.mode = 'experiment'
        self.default_diversity = 'True'

        self.algos = algos.split('/')
        self.explanation_ratios = [int(x) for x in  explication_ratio.split('/')]

        #Check Errors

        if len(self.algos) != len(self.explanation_ratios):
            raise ValueError(f'{len(self.algos)} Algos and {len(self.explanation_ratios)} Ratios given. These must be equal.')

        for alg in self.algos:
            if alg not in semantics:
                raise ValueError(f'{alg} not in list of algorithms. List is : {semantics}')

        if (epsilon < 0) | (epsilon > 1):
            raise ValueError(f'Epsilon must be between 0 and 1')

        if (exploration_distance < 0) | (exploration_distance > 1):
            raise ValueError(f'Exploration_distance must be between 0 and 1')

        if (self.algos[0] == 'Egreedy') & (exploration_type not in ['Standard','NewPanel','Hybrid']):
            raise ValueError(f"For Egreedy algorithm, the exploration types must be : ['Standard','NewPanel','Hybrid']")

        if (self.algos[0] == 'MAB') & (exploration_type not in ['Softmax','Standard','UCB','Thompson']):
            raise ValueError(f"For MAB algorithm, the exploration types must be : ['Softmax','Standard','UCB','Thompson']")

        self.check_history = history
        self.inclusion = inclusion
        self.exploration_type = exploration_type
        self.epsilon = epsilon
        self.exploration_distance = exploration_distance
        self.n_attributes = n_attributes

        self.explanation_type = 'one'
        self.attribute_threshold = 7
        self.discretize_K = True

        self.target_dashboard = target
        self.verbose = verbose

        # set some properties
        self.diversity = {'asked': self.default_diversity, 'achieved': False}

        self.star = Attribute()
        self.star.name = '*'

        self.new_panel_level = self.n_attributes-1
        self.last_combi = []
        self.last_quality = []

        self.MAB_init = 0
        self.MAB_explanation = 0
        self.si = np.ones(7)
        self.ni = np.ones(7)

    def preprocess_data(self):

        self.data_preprocessor = DataPreprocessor(self.dataset_name, self.attribute_threshold, self.discretize_K)
        self.attributes = self.data_preprocessor.attributes
        self.attributes_and_star = self.attributes + [self.star]

        self.dataset = self.data_preprocessor.dataset
        self.ranking = AttributesRanking(self.attributes, self.star)

    def initialize_dashboard_generation(self):
        """
        create panel, panel_number, dashboard, dashbot_history, generation_counter
        """
        self.panel_number = [0,0]
        self.dashboard = Dash(self.attributes_and_star)

        if self.check_history:
            self.dashbot_history = []

        self.generation_counter = 0

        self.panel = Panel(self.attributes_and_star)

    def validate_panel(self):
        """
        update dashboard, diversity, attributes_on_dashboard
        prepare next suggestion
        """
        # put last suggestion on dashboard 
        self.dashboard.add_panel(self.panel, tuple(self.panel_number))
        # update objects needed for diversity
        if self.diversity['achieved'] == False :
            for attribute in self.panel.groupBy + list(self.panel.aggregates.keys()):
                attribute.on_dashboard = True
                self.dashboard.attributes_on.add(attribute)
                if len(self.dashboard.attributes_on) == len(self.attributes):
                    self.diversity['achieved'] = True
 
    ################### RANDOM #####################

    def generate_random_panel(self):

        self.generation_counter += 1
        self.panel.reset()

        aggregation_attributes = [at for at in self.attributes if at.is_numeric]# + [self.star]
        functions = ['min', 'max', 'sum', 'avg']

        # Choose groupby_attributes
        groupBy_at = [at for at in self.attributes if at.name not in [i[0] for i in self.bad_groupBy_names]]

        for at in groupBy_at:
            bit = random.choice([0,2])
            if bit > 0 :
                self.panel.groupBy.append(at)

        if len(self.panel.groupBy) == 0:
            self.panel.groupBy.append(random.choice(groupBy_at))

        # Choose aggregation attributes
        grpb = [i.name for i in self.panel.groupBy]
        aggs = [ at for at in aggregation_attributes if at.name not in grpb] #Remove attributes with a GroupBy

        name2attr = {at.name:at for at in aggs}

        aggs = list(name2attr.keys()) 
        aggs = list( itertools.product(*[aggs,functions]) )+[('*','count')] #Get all pairs (Attribute-Agg_Function)

        aggs = list( set(aggs)-set(self.bad_agg_names) ) #Select (Attribute-Agg_Function) that are not forbiden
        name2attr['*'] = self.star

        for (name,func) in aggs:
            bit = random.choice([0,1])

            if bit > 0 : 
                if name2attr[ name ] in self.panel.aggregates:
                    self.panel.aggregates[ name2attr[ name ] ] = self.panel.aggregates[ name2attr[ name ] ]+[ func ]
                else:
                    self.panel.aggregates[ name2attr[ name ] ] = [ func ]

        if len(self.panel.aggregates) == 0:
            agg_bit = random.choice(aggs)
            self.panel.aggregates[ name2attr[ agg_bit[0] ] ] = [ agg_bit[1] ]

        self.panel.attributes_to_vector()

    def find_random_panel(self):
        loop = True

        num_attributes = [at for at in self.attributes if at.is_numeric] #Get numeric attributes

        self.bad_groupBy_names = [(at.name, 'groupBy') for at in self.attributes if at.bad_groupBy > 0] #Get attributes that are Bad

        ########## Here Add after that the case where Star is forbiden
        
        self.bad_agg_names = [] #Get aggregation functions of attirbutes that are bad
        [self.bad_agg_names.extend( [(at.name,func) for func in at.bad_func] ) for at in num_attributes if len(at.bad_func) > 0]

        while loop:
            # generate random panel
            self.generate_random_panel()

            # check if already in history
            if self.check_history:
                loop = self.check_in_history(self.panel.vector.to_numpy())

            if self.generation_counter == 10000000:
                self.eval.end_experiment("\nSTUCK IN RECURSIVE FUNCTION <find_panel_satisfying_criteria> !!!")
                sys.exit("RANDOM KILLED")
                return None

    ################### Egreedy #####################

    def perform_Egreedy(self):
        loop = True
        hybrid_count = 0
        num_stack = 1

        num_attributes = [at for at in self.attributes if at.is_numeric] #Get numeric attributes

        self.bad_groupBy_names = [(at.name, 'groupBy') for at in self.attributes if at.bad_groupBy > 0] #Get attributes that are Bad

        self.bad_agg_names = [] #Get aggregation functions of attirbutes that are bad
        [self.bad_agg_names.extend( [(at.name,func) for func in at.bad_func] ) for at in num_attributes if len(at.bad_func) > 0]
        #self.bad_agg_names.append(('*','count'))

        while loop:
            # change bits
            random_number = np.random.rand()

            if random_number > self.epsilon:
                next_panel = self.refine_by_egreedy(True) #Exploitation
            else: 
                if (self.exploration_type == 'Standard') or (self.exploration_type == 'Hybrid' and hybrid_count < 20):
                    hybrid_count += 1
                    next_panel = self.refine_by_egreedy(False) #Exploration Egreedy-1 or Egreedy-3
                else:
                    hybrid_count = 0
                    self.generate_new_panel() #Exploration Egreedy-2 or Egreedy-3
                    next_panel = self.panel

            self.generation_counter += 1
            
            next_panel.vector_to_attributes()

            #Check if Panel is correct, otherwise fix it
            next_panel = self.fix_panel(next_panel)

            # check if in history
            recommendable = self.check_if_recommendable(next_panel)
            
            if recommendable:
                loop = False
                
            self.panel = next_panel

            if self.generation_counter % 2000 == 0:
                print("STUCK")
                print(self.exploration_type)
                #print(self.algos)
                #print(self.n_attributes)
                #print(self.explication_ratio)
                print()

                self.panel.list_to_vector(self.dashbot_history[-1*num_stack])
                self.panel.vector_to_attributes()

                num_stack += 1

                #self.panel = None
                #return self.panel
        
    def refine_by_egreedy(self, exploit):
        # switching probabilities vector
        next_panel = self.panel.copy_from_vector()
        attributes = next_panel.attributes

        bad = self.bad_groupBy_names + self.bad_agg_names #Bad groupBys and Aggregations

        bad_bits = [next_panel.columns.get_locs(i)[0] for i in bad] #Bad bits

        next_panel.vector[ bad_bits ] = 0 #Force Bad bits to 0

        switching_probabilities = np.ones(len(next_panel.vector))
        switching_probabilities[bad_bits] = 0 #Force bad bits to proba 0

        switching_probabilities = switching_probabilities/switching_probabilities.sum() #Probabilities of switch
        
        # exploit
        if exploit:
            number_of_bits = 1
        else:
            number_of_bits = int(self.exploration_distance * len(switching_probabilities))

        bits_to_change = np.random.choice( range(len(switching_probabilities)) , size=number_of_bits, p=switching_probabilities)
        bits_to_change = next_panel.columns[bits_to_change]

        for bit in bits_to_change:
            old_value = next_panel.vector[bit]
            attribute, function = bit

            if old_value == 0:
                if function == 'groupBy':
                    next_panel.vector[bit] = 2
                else:
                    next_panel.vector[bit] = 1
            else:
                next_panel.vector[bit] = 0

        return next_panel
    
    def fix_panel(self, panel_in):

        aggregation_attributes = [at for at in self.attributes if at.is_numeric]# + [self.star]
        functions = ['min', 'max', 'sum', 'avg']

        panel = panel_in.copy_from_vector()

        loop = True

        while loop :
            loop = False

            # No GROUP BY attribute
            if len(panel.groupBy) == 0:
                loop = True
                groupbys = [ at for at in self.attributes if at.name not in [i[0] for i in self.bad_groupBy_names] ] #Select Groupbys that are not forbiden
                panel.groupBy.append(random.choice(groupbys)) #Choose one randomly
            
            # No aggregates
            if len(panel.aggregates.keys()) == 0:
                loop = True

                grpb = [i.name for i in panel.groupBy]
                aggs = [ at for at in aggregation_attributes if at.name not in grpb] #Remove attributes with a GroupBy

                name2attr = {at.name:at for at in aggs}

                aggs = list(name2attr.keys()) 
                aggs = list( itertools.product(*[aggs,functions]) )+[('*','count')] #Get all pairs (Attribute-Agg_Function)

                aggs = list( set(aggs)-set(self.bad_agg_names) ) #Select (Attribute-Agg_Function) that are not forbiden
                name2attr['*'] = self.star

                aggs = random.choice(aggs)
                panel.aggregates[ name2attr[aggs[0]] ] = [aggs[1]]
            
            # same attribute with GroupBy and Aggregation
            same_attributes = list(set(panel.groupBy) & set(panel.aggregates.keys()))

            if len(same_attributes) > 0:
                loop = True

                last_dim = ''
                dims = np.array(['groupBy', 'aggregation'])

                #Remove either Groupby or Aggregations for each shared attribute
                for idx, at in enumerate(same_attributes):
                    if idx == 0: #First dimension to remove
                        if len(panel.groupBy) > len(panel.aggregates) : #More Groupbys than Aggregations
                            last_dim = 'groupBy'
                        elif len(panel.groupBy) < len(panel.aggregates): #More Aggregations than Groupbys
                            last_dim = 'aggregation'
                        else:
                            last_dim = random.choice(dims) #Choose Randomly One Dimension (Groupbys == Aggregations)
                    else:
                        last_dim = dims[dims!=last_dim][0] #Choose the other alternative (based on last time) 

                    if last_dim == 'groupBy':
                        panel.groupBy.remove(at)
                    else:
                        del panel.aggregates[at]
                
                same_attributes = list(set(panel.groupBy) & set(panel.aggregates.keys()))

            panel.attributes_to_vector()
            panel.vector_to_attributes()

        return panel

    ################### MAB #####################
    def perform_MAB(self):

        output_panel = self.panel.copy_from_vector()

        if self.MAB_init < 7:
            #initialization phase
            output_panel = self.get_panel_after_explanation(output_panel, self.MAB_init)
            self.MAB_explanation = self.MAB_init
            self.MAB_init += 1
        else:
            #MAB phase
            self.MAB_explanation = self.strategy_mab()
            output_panel = self.get_panel_after_explanation(output_panel, self.MAB_explanation)

        num_attributes = [at for at in self.attributes if at.is_numeric] #Get numeric attributes
        self.bad_groupBy_names = [(at.name, 'groupBy') for at in self.attributes if at.bad_groupBy > 0] #Get attributes that are Bad

        self.bad_agg_names = [] #Get aggregation functions of attirbutes that are bad
        [self.bad_agg_names.extend( [(at.name,func) for func in at.bad_func] ) for at in num_attributes if len(at.bad_func) > 0]
        #self.bad_agg_names.append(('*','count'))

        output_panel = self.fix_panel(output_panel)
        self.panel = output_panel

    def get_panel_after_explanation(self, panel, index_list):

        self.ni[index_list] += 1

        if index_list == 0:
            #Bad Group By
            bad_groupbys = [at for at in self.attributes if at.bad_groupBy >0]

            groupbys = panel.groupBy
            aggs = list( panel.aggregates.keys() )

            good_grby = [at for at in self.attributes if (at not in bad_groupbys) and (at not in groupbys)]

            if len(good_grby) == 0:
                good_grby = aggs
            
            good_grby_2 = list( set(good_grby) - set(aggs) )

            if len(good_grby_2) == 0:
                #If other groupbys does not exist, choose one from the previous groupbys
                panel.groupBy = [ random.choice(groupbys) ]
            else:
                panel.groupBy = [ random.choice(good_grby_2) ]

        elif index_list == 1:
            #Bad Aggregates
            groupbys = panel.groupBy
            aggs = list( panel.aggregates.keys() ) #+[at for at in self.attributes if at in groupbys]

            bad_func_agg = []
            [bad_func_agg.extend( [ (at,func) for func in at.bad_func ] ) for at in self.attributes if (len(at.bad_func) > 0)]
            
            good_agg = [at for at in self.attributes if (at not in aggs) and (at.is_numeric==True)]
            
            if self.star not in aggs:
                good_agg += [self.star]
            
            if len(good_agg) == 0:
                good_agg = groupbys
            
            if len( set(good_agg)-set(groupbys) ) > 0:
                good_agg = list( set(good_agg)-set(groupbys) )

                random.shuffle(good_agg)
                loop = True
                k = 0

                while ( k < len(good_agg) ) and ( loop==True ): 
                    agg_choice = good_agg[k]

                    if agg_choice in [self.star]:
                        agg = [(self.star,'count')]
                    else:
                        agg = list( itertools.product([agg_choice],['min','max','sum','avg']) )
                    
                    agg = list(set(agg) - set(bad_func_agg))
                    k += 1
                    if len(agg) != 0:
                        loop = False

                if loop == True:
                    raise ValueError('No Aggregation attribute available for MAB')
            
            else:
                #No good_agg or All good_agg are groupbys.
                agg = random.choice( aggs )
                
                if agg in [self.star]:
                    agg = [(self.star,'count')]
                else:
                    agg = list( itertools.product([agg],['min','max','sum','avg']) )

            func = [j for i,j in agg]
            agg_choice = [i for i,j in agg]
            panel.aggregates = {agg_choice[0]:func}

        elif index_list == 2:
            #Change min
            aggs = [at for at,values in panel.aggregates.items() if 'min' in values]
            
            if len(aggs) > 0 :
                at = random.choice(aggs)
                panel.aggregates[at].remove('min')
            else:
                good_agg = [at for at in self.attributes if ('min' not in at.bad_func) and (at not in panel.groupBy) and (at.is_numeric==True)]
                at = random.choice(good_agg)
                if at in panel.aggregates:
                    panel.aggregates[at].append('min')
                else:
                    panel.aggregates[at] = ['min']
        
        elif index_list == 3:
            #Change max
            aggs = [at for at,values in panel.aggregates.items() if 'max' in values]
            
            if len(aggs) > 0 :
                at = random.choice(aggs)
                panel.aggregates[at].remove('max')
            else:
                good_agg = [at for at in self.attributes if ('max' not in at.bad_func) and (at not in panel.groupBy) and (at.is_numeric==True)]
                at = random.choice(good_agg)
                if at in panel.aggregates:
                    panel.aggregates[at].append('max')
                else:
                    panel.aggregates[at] = ['max']
        
        elif index_list == 4:
            #Change count
            if self.star in panel.aggregates.keys():
                del panel.aggregates[self.star]
            else:
                panel.aggregates[self.star] = ['count']
        
        elif index_list == 5:
            #Change sum
            aggs = [at for at,values in panel.aggregates.items() if 'sum' in values]
            
            if len(aggs) > 0 :
                at = random.choice(aggs)
                panel.aggregates[at].remove('sum')
            else:
                good_agg = [at for at in self.attributes if ('sum' not in at.bad_func) and (at not in panel.groupBy) and (at.is_numeric==True)]
                at = random.choice(good_agg)
                if at in panel.aggregates:
                    panel.aggregates[at].append('sum')
                else:
                    panel.aggregates[at] = ['sum']
        
        else:
            #Change avg
            aggs = [at for at,values in panel.aggregates.items() if 'avg' in values]
            
            if len(aggs) > 0 :
                at = random.choice(aggs)
                panel.aggregates[at].remove('avg')
            else:
                good_agg = [at for at in self.attributes if ('avg' not in at.bad_func) and (at not in panel.groupBy) and (at.is_numeric==True)]
                at = random.choice(good_agg)
                if at in panel.aggregates:
                    panel.aggregates[at].append('avg')
                else:
                    panel.aggregates[at] = ['avg']

        panel.attributes_to_vector()
        return panel

    def strategy_mab(self):
        ui = self.si / self.ni

        if self.exploration_type == 'Standard':
            # Egreedy du MAB
            random_number = random.random()

            if random_number > self.epsilon:

                if self.MAB_init < 7:
                    return np.argmax(ui)
                else:
                    maxi = ui.max()
                    indexes = np.where(ui==maxi)[0]
                    
                    return np.argmax(indexes)
            else:
                return random.choice( list(range(7)) )
            
        elif self.exploration_type == 'UCB':
            # UCB
            ui_prime = np.sqrt(2*np.log2(self.eval.iteration_number)/self.ni)+ui
            
            if self.MAB_init < 7:
                return np.argmax(ui_prime)
            else:
                maxi = ui_prime.max()
                indexes = np.where(ui_prime==maxi)[0]

                return random.choice(indexes)

        elif self.exploration_type == 'Softmax':
            # Softmax
            tau = 0.05

            ui_prime = ui/tau
            ui_prime = np.exp(ui_prime)
            ui_prime = ui_prime/ui_prime.sum()

            if self.MAB_init < 7:
                return np.argmax(ui_prime)
            else:
                maxi = ui_prime.max()
                indexes = np.where(ui_prime==maxi)[0]

                return random.choice(indexes)
        
        elif self.exploration_type == 'Thompson':
            betas = [random.betavariate(self.si[i]+1, self.ni[i]-self.si[i]+1) for i in range(len(ui))]

            if self.MAB_init < 7:
                return np.argmax(betas)
            else:
                maxi = betas.max()
                indexes = np.where(betas==maxi)[0]

                return random.choice(indexes)

    ################### New Panel #####################
    def generate_new_panel(self):

        self.panel.reset()
        loop = True

        while loop:

            if (self.new_panel_level < self.n_attributes) and (self.new_panel_level > 1):
                new_panel = self.find_both_attributes_combination(self.new_panel_level)

                if new_panel is None :
                    self.new_panel_level -= 1
                
            elif self.new_panel_level == 1:
                # One Groupby Attribute and One Aggregation Attribute
                new_panel = self.find_both_attributes()
                if new_panel is None :
                    self.new_panel_level -= 1
            else:
                new_panel = self.apply_strategy_if_all_pairwise_panels_have_been_shown()

            if new_panel is not None:
                loop = False

        new_panel.attributes_to_vector()
        self.panel = new_panel

    def find_both_attributes(self, forbidden_groupBy=list(), forbidden_agg=list()):
        
        #if len(self.panel) == 0:
        #    self.panel.aggregates[self.star] = ['count']
        #    self.panel.attributes_to_vector()

        for groupBy_quality, agg_quality in self.ranking.pairwise_qualities_2:
            
            out_panel = self.panel.copy_from_vector()

            if groupBy_quality != None:
                self.ranking.calculate_ranking('groupBy', out_panel, self.diversity, forbidden_groupBy)
                ranking_groupBy = self.ranking.local['groupBy'][groupBy_quality][:]

                if len(ranking_groupBy) == 0:
                    continue

            if agg_quality != None:

                loop_gb_attr = True
                len_gb_attr = 0

                while (loop_gb_attr==True) and (len_gb_attr < len(ranking_groupBy)):

                    gb_attr = ranking_groupBy[len_gb_attr]
                    len_gb_attr += 1
                    
                    out_panel.groupBy.append(gb_attr)

                    self.ranking.calculate_ranking('aggregation', out_panel, self.diversity, forbidden_agg)
                    ranking_aggregation = self.ranking.local['aggregation'][agg_quality][:]
                    
                    reco_panel = False

                    #Case where the groupby attribute has no aggreagtion attributes
                    if len(ranking_aggregation) == 0:
                        out_panel.groupBy.pop(-1)
                        continue
                    
                    for agg in ranking_aggregation:
                        func = self.choose_functions(agg, True)

                        if len(func) == 0:
                            print('Case where the aggregation has no functions')
                            continue
                        
                        for f in func:
                            out_panel.aggregates[agg] = list(f)
                            self.generation_counter += 1

                            out_panel.attributes_to_vector()
                            recommendable = self.check_if_recommendable(out_panel)

                            #If Panel recommendable Stop, otherwise loop again
                            if recommendable == True:
                                break
                            else:
                                del out_panel.aggregates[agg]
                    
                        if recommendable == True:
                            reco_panel = True
                            break
                        else:
                            if agg in out_panel.aggregates:
                                del out_panel.aggregates[agg]

                    if reco_panel == True:
                        loop_gb_attr = False
                    else:
                        out_panel.groupBy.pop(-1)

                if (loop_gb_attr == True) and (len_gb_attr == len(ranking_groupBy)):
                    #Case where No groupbys - Some Groupbys have no Aggregations - No Recommandable Panel for this Groupbys
                    continue

                break
        
        out_panel.attributes_to_vector()

        if out_panel.vector.tolist() == self.panel.vector.tolist():
            out_panel = None

        return out_panel

    def choose_functions(self, agg_att, all_comb=False):
        if agg_att.name == '*':
            functions = ['count']
        else:
            functions = ['avg', 'sum', 'max', 'min'] # ordered list for choice of aggregate if visu = 'word cloud'
            
            for func in agg_att.bad_func:
                functions.remove(func)
        
        if all_comb == False:
            return functions 
        
        num_func = len(functions)

        combinaison = []
        [ combinaison.append(i) for j in range(num_func,0,-1) for i in list(itertools.combinations(functions,j)) ]

        return combinaison

    def apply_strategy_if_all_pairwise_panels_have_been_shown(self):
        loop = True

        while loop:
            self.panel.reset()

            num_attributes = [at for at in self.attributes if at.is_numeric] #Get numeric attributes

            self.bad_groupBy_names = [(at.name, 'groupBy') for at in self.attributes if at.bad_groupBy > 0] #Get attributes that are Bad

            self.bad_agg_names = [] #Get aggregation functions of attirbutes that are bad
            [self.bad_agg_names.extend( [(at.name,func) for func in at.bad_func] ) for at in num_attributes if len(at.bad_func) > 0]
            #self.bad_agg_names.append(('*','count'))

            self.panel = self.perform_Egreedy()
            self.panel.vector_to_attributes()

            next_panel = self.fix_panel(self.panel)

            self.ranking.calculate_ranking('groupBy', self.panel, self.diversity)
            self.ranking.calculate_ranking('aggregation', self.panel, self.diversity)

            #new_panel = self.find_both_attributes()

            if new_panel is not None:
                loop = False

        return new_panel

    def find_both_attributes_combination(self, num_combi, forbidden_groupBy=list(), forbidden_agg=list()):
        
        #if len(self.panel) == 0:
        #    self.panel.aggregates[self.star] = ['count']
        #    self.panel.attributes_to_vector()

        if num_combi > 12:
            num_combi = 12

        all_combinaison = list(range(1,num_combi+1))
        all_combinaison = list(itertools.product(all_combinaison,repeat=2))
        all_combinaison.remove((1,1))

        all_combinaison = [i for i in all_combinaison if (sum(i) < self.n_attributes+1) and (num_combi in set(i)) ]
        all_combinaison = [i for i in all_combinaison if i not in self.last_combi[:-1]]

        for cambi in all_combinaison:

            if cambi not in self.last_combi:
                self.last_combi.append(cambi)

            out_panel = self.panel.copy_from_vector()
            
            nb_attr_grp = cambi[0]
            nb_attr_agg = cambi[1]

            grp_cambi = list(itertools.product(qualities,repeat=nb_attr_grp))
            agg_cambi = list(itertools.product(qualities,repeat=nb_attr_agg))

            grp_cambi = [pair for pair in grp_cambi if len(set(pair))==len(pair) ]
            agg_cambi = [pair for pair in agg_cambi if len(set(pair))==len(pair) ]

            quality_cambi = list(itertools.product(grp_cambi,agg_cambi))
            quality_cambi = [i for i in quality_cambi if i not in self.last_quality[:-1]]
            
            #quality_cambi = list(set(quality_cambi)-set(self.last_quality[:-1]))

            for groupBy_quality, agg_quality in quality_cambi:
                prev_qual = (groupBy_quality,agg_quality)

                if prev_qual not in self.last_quality: 
                    self.last_quality.append(prev_qual)
                               
                if groupBy_quality != None:
                    ranking_groupBy = []
                    self.ranking.calculate_ranking('groupBy', out_panel, self.diversity, forbidden_groupBy)

                    for qual in groupBy_quality:
                        ranking_groupBy += self.ranking.local['groupBy'][qual][:]
                    
                    if len(ranking_groupBy) < nb_attr_grp:
                        continue

                if agg_quality != None:

                    loop_gb_attr = True
                    len_gb_attr = 0

                    ranking_groupBy = list( itertools.combinations(ranking_groupBy,nb_attr_grp) )

                    while (loop_gb_attr==True) and (len_gb_attr < len(ranking_groupBy)):

                        gb_attr = ranking_groupBy[len_gb_attr]
                        len_gb_attr += 1
                        
                        for gb_ in gb_attr:
                            out_panel.groupBy.append(gb_)
                        
                        ranking_aggregation = []

                        self.ranking.calculate_ranking('aggregation', out_panel, self.diversity, forbidden_agg)

                        for qual in agg_quality:
                            ranking_aggregation += self.ranking.local['aggregation'][qual][:]
                        
                        ranking_aggregation = [at for at in ranking_aggregation if at not in out_panel.groupBy]

                        #print([at.name for at in gb_attr],' - ',[at.name for at in ranking_aggregation])
                        reco_panel = False

                        #Case where the groupby attribute has no aggreagtion attributes
                        if len(ranking_aggregation) < nb_attr_agg:
                            for gb_ in gb_attr:
                                out_panel.groupBy.remove(gb_)
                            #out_panel.groupBy = list()
                            continue
                        
                        ranking_aggregation = list( itertools.combinations(ranking_aggregation,nb_attr_agg) )

                        ranking_aggregation = [agg for agg in ranking_aggregation if len(set(agg))==len(agg)]

                        for aggs in ranking_aggregation:  
                            no_func_number = 0

                            for agg in aggs :
                                func = self.choose_functions(agg)

                                if len(func) == 0:
                                    no_func_number += 1
                                    print('Case where the aggregation has no functions')
                                    continue
                                
                                out_panel.aggregates[agg] = func
                            
                            if no_func_number != 0 :
                                continue

                            self.generation_counter += 1

                            out_panel.attributes_to_vector()

                            recommendable = self.check_if_recommendable(out_panel)

                            #If Panel recommendable Stop, otherwise loop again
                            if recommendable == True:
                                reco_panel = True
                                break
                            else:
                                out_panel.aggregates = dict()
                                #out_panel.aggregates[self.star] = ['count']
                                out_panel.attributes_to_vector()
                        
                        if reco_panel == True:
                            loop_gb_attr = False
                        else:
                            for gb_ in gb_attr:
                                out_panel.groupBy.remove(gb_)
                            #out_panel.groupBy = list()

                    if (loop_gb_attr == True) and (len_gb_attr == len(ranking_groupBy)):
                        #Case where No groupbys - Some Groupbys have no Aggregations - No Recommandable Panel for this Groupbys 
                        continue
                    
                    break
            
            out_panel.attributes_to_vector()

            if out_panel.vector.tolist() != self.panel.vector.tolist():
                break

        if out_panel.vector.tolist() == self.panel.vector.tolist():
            out_panel = None
        
        return out_panel
                  
    ################### Explanation #####################
    def apply_explanation(self, explanation):
        
        choice = ''
        output = self.panel.copy_from_vector()

        #groupBy_to_add = explanation['groupBy_to_add']
        #aggregation_to_add = explanation['aggregation_to_add']
        #aggregation_to_remove = explanation['aggregation_to_remove']
        
        groupBy_to_remove = explanation['groupBy_to_remove']
        functions_to_change = explanation['functions_to_change']

        if len(functions_to_change) == 0 and len(groupBy_to_remove)==0:
            return None

        if len(groupBy_to_remove) > 0 and len(functions_to_change) > 0:
            choice = random.choice(['groupby','aggregation_function'])
        elif len(groupBy_to_remove) == 0:
            choice = 'aggregation_function'
        else:
            choice = 'groupby'

        if choice == 'groupby':
            output = self.remove_Groupby(output, groupBy_to_remove)
        else:
            output = self.change_function(output,functions_to_change)

        num_attributes = [at for at in self.attributes if at.is_numeric] #Get numeric attributes
        
        self.bad_groupBy_names = [(at.name, 'groupBy') for at in self.attributes if at.bad_groupBy > 0] #Get attributes that are Bad

        ########## Here Add after that the case where Star is forbiden<
        
        self.bad_agg_names = [] #Get aggregation functions of attirbutes that are bad
        [self.bad_agg_names.extend( [(at.name,func) for func in at.bad_func] ) for at in num_attributes if len(at.bad_func) > 0]

        output.attributes_to_vector()
        output = self.fix_panel(output)
        output.attributes_to_vector()

        self.generation_counter += 1
            
        self.panel = output
        
        self.ranking.calculate_general_ranking('groupBy', self.diversity)
        self.ranking.calculate_general_ranking('aggregation', self.diversity)

        self.new_panel_level = self.n_attributes-1
        self.last_combi = []
        self.last_quality = []
    
    def remove_Groupby(self, panel, groupBy_to_remove):
        groupby = random.choice(groupBy_to_remove)
        panel.groupBy.remove(groupby)

        idx = self.attributes.index(groupby)
        self.attributes[idx].bad_groupBy += 1

        return panel

    def change_function(self, panel, functions_to_change):
        agg_att = random.choice(functions_to_change)
        at, func = agg_att

        if at in panel.aggregates:
            agg_funcs = panel.aggregates[at]
        else:
            agg_funcs = []
        
        if at in self.attributes:
            idx = self.attributes.index(at)
        else:
            idx = -1 #Case where the attribute to remove is star

        if func in agg_funcs:
            #It is in the panel but it is not in the target, So we remove it
            agg_funcs.remove(func)
            panel.aggregates[at] = agg_funcs

            if idx > -1 :
                self.attributes[idx].bad_func.append(func)
        else:
            #It is in the target but not in the panel, So we add it
            agg_funcs.append(func)
            panel.aggregates[at] = agg_funcs

            if idx > -1 :
                if func in self.attributes[idx].bad_func:
                    self.attributes[idx].bad_func.remove(func)

        return panel

    ################### History #####################

    def check_if_recommendable(self, panel):
        recommendable = not self.check_in_history(panel.vector.to_numpy())
        return recommendable

    def check_in_history(self, panel_vector):
        
        if len(self.dashbot_history) == 0:
            return False

        if self.inclusion == False:
            found = np.any( np.all(panel_vector == self.dashbot_history, axis=1) ) 
        else:
            found = self.is_included_panel_in_history(panel_vector)

        return found
    
    def is_included_panel_in_history(self, panel_vector):
        found = np.any( np.all( np.sqrt(self.dashbot_history*panel_vector) == self.dashbot_history, axis=1 ) )
        return found

    def show_to_U(self):
        self.dashbot_history.append(self.panel.vector.to_numpy())
