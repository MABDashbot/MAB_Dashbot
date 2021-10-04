import pandas as pd
import numpy as np
import math
import sys

from .attribute import *

class Panel():
    """
    A panel can either be described by:
    1) a vector : self.vector
       and a list of all Attribute objects
    2) a list of Attribute objects for GROUP BY : self.groupBy
       and a dict (with key = Attribute object, value = list of function) for aggregates : self.aggregates
    """

    def __init__(self, attributes_and_star):
        self.attributes = attributes_and_star[:-1]
        self.star = attributes_and_star[-1]

        # self.groupBy : list of Attribute objects
        self.groupBy = list()

        # self.aggregates : dict with key = Attribute object, value = list of functions
        self.aggregates = dict()

        # self.columns : MultiIndex of tuples (attribute_name, function)
        all_functions = ['groupBy', 'min', 'max', 'sum', 'avg']
        numeric_functions = ['min', 'max', 'sum', 'avg']
        attributes_names = [at.name for at in self.attributes]

        index = pd.MultiIndex.from_product( (attributes_names, all_functions), names=["attribute", "function"])
        non_numeric_names = [at.name for at in self.attributes if not at.is_numeric]
        to_remove = [ (at,f) for f in numeric_functions for at in non_numeric_names ]

        self.columns = index.copy().drop(to_remove)
        self.columns = self.columns.insert(0, ('*','count'))

        # self.vector : Series with index = self.columns
        self.vector = pd.Series( np.zeros( shape = self.columns.size, dtype = 'int'), index = self.columns )

    def __len__(self):
        return len(self.groupBy) + len(self.aggregates)

    def show(self):
        return print(pd.DataFrame([self.vector]))

    def list_to_vector(self, panel_list):
        self.vector = pd.Series(panel_list.copy(), index=self.columns )
        self.vector_to_attributes()

    def attributes_to_vector(self):
        # put values in self.vector from self.groupBy and self.aggregates        
        self.vector = pd.Series(np.zeros( shape = self.columns.size, dtype = 'int'), index=self.columns )

        for attribute in self.groupBy:
            self.vector[attribute.name, 'groupBy'] = 2 # TODO : gérer le cas = 1
        for attribute, functions in self.aggregates.items():
            for f in functions:
                self.vector[attribute.name, f] += 1

    def vector_to_attributes(self):
        self.groupBy = list()
        self.aggregates = dict()

        name2attr = {att.name:att for att in self.attributes}

        for attribute_name, function in self.vector[lambda n : n > 0].index:
            if (attribute_name, function) == ('*', 'count'):
                self.aggregates[self.star] = ['count']
            else:
                if function == 'groupBy':
                    self.groupBy.append(name2attr[attribute_name])
                else:
                    self.aggregates.setdefault(name2attr[attribute_name], []).append(function)
        
    def reset(self):
        self.vector = pd.Series(np.zeros( shape = self.columns.size, dtype = 'int'), index=self.columns )
        self.groupBy = list()
        self.aggregates = dict()

    def copy_from_vector(self):
        new_panel = Panel(self.attributes + [self.star])
        new_panel.vector = self.vector.copy()
        new_panel.vector_to_attributes()
        return new_panel

class Dash(Panel):
    """
    Objects containing several panels.
    e.g. : dashboard (both modes) and target dashboard (experiment mode)
    """

    def __init__(self, attributes_and_star):
        super().__init__(attributes_and_star)

        self.index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'panel', u'suggestion'])
        self.dataframe = pd.DataFrame( columns=self.columns, index=self.index )
        self.dict = dict()
        self.attributes_on = set()

    def __len__(self):
        return self.dataframe.shape[0]

    def show(self):
        return print(self.dataframe)

    def add_panel(self, panel, index):

        if type(index) == int:
            self.dataframe.loc[index] = panel.vector.copy()
        elif type(index) == tuple:
            self.dataframe.loc[index,:] = panel.vector.copy()
        
        self.dataframe = self.dataframe.astype(int)
        self.dict[index] = panel.vector.to_numpy()
