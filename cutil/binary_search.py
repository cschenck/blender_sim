#!/usr/bin/env python

#system imports
import os
import math
import random
import copy
import pickle

FLOAT_LOG = 1


NUMBER_TYPES = [float, int, FLOAT_LOG]
    
class BinarySearch():
    def __init__(self, args, error_func, stop_ratio=0.01, cache_file=None):
        self.error_func = error_func
        self.stop_ratio = stop_ratio
        self.cache = {}
        self.cache_file = cache_file
        
        if self.cache_file is not None and os.path.exists(self.cache_file):
            self.cache = pickle.load(open(self.cache_file, "rb"))
        
        self.param_ranges = {}
        self.param_types = {}
        self.param_log_shifts = {}
        for n, t, r in args:
            self.param_ranges[n] = r
            self.param_types[n] = t
            if t is FLOAT_LOG:
                self.param_ranges[n] = []
                if r[0] <= 0.0:
                    self.param_log_shifts[n] = -r[0] + 1
                else:
                    self.param_log_shifts[n] = 0.0
                for i in range(len(r)):
                    self.param_ranges[n].append(math.log(r[i] + self.param_log_shifts[n]))
                    
    def __error_func_log(self, **args):
        for key in args.keys():
            if self.param_types[key] is FLOAT_LOG:
                args[key] = math.exp(args[key]) - self.param_log_shifts[key]
                
        #cache lookups so we don't call the error function on the same args
        key = frozenset(args.items())
        if not key in self.cache:
            self.cache[key] = self.error_func(**args)
            self.__save_cache()
        return self.cache[key]
        
    def __save_cache(self):
        if self.cache_file is not None:
            pickle.dump(self.cache, open(self.cache_file, "wb"))
            
    
    def init_centered_param(self, param):
        if self.param_types[param] is float or self.param_types[param] is FLOAT_LOG:
            return (self.param_ranges[param][0] + self.param_ranges[param][1])/2.0
        elif self.param_types[param] is int:
            return int(round((self.param_ranges[param][0] + self.param_ranges[param][1])/2.0))
        else:
            return self.param_ranges[param][0]
        
    def init_random_param(self, param):
        if self.param_types[param] is float or self.param_types[param] is FLOAT_LOG:
            return random.uniform(self.param_ranges[param][0] , self.param_ranges[param][1])
        elif self.param_types[param] is int:
            return random.randint(self.param_ranges[param][0], self.param_ranges[param][1])
        else:
            return aself.param_ranges[param][random.randint(0, len(self.param_ranges[param]) - 1)]
            
    def is_param_stopped(self, param):
        if self.param_types[param] in NUMBER_TYPES:
            if ((not self.max_param[param] == self.min_param[param]) and
                (self.max_param[param] - self.min_param[param])/(self.param_ranges[param][1] - self.param_ranges[param][0]) >= self.stop_ratio):
                return False
            else:
                return True
        else:
            return True
            
    def stopping_criteria_reached(self):
        for param in self.param_ranges.keys():
            if not self.is_param_stopped(param):
                return False
        return True
        
    def generate_all_steps(self, pivot):
        ret = []
        for param in pivot.keys():
            if self.param_types[param] in NUMBER_TYPES:
                up = copy.copy(pivot)
                up[param] = pivot[param] + (self.max_param[param] - pivot[param])/2.0
                down = copy.copy(pivot)
                down[param] = self.min_param[param] + (pivot[param] - self.min_param[param])/2.0
                if self.param_types[param] is int:
                    up[param] = int(round(up[param]))
                    down[param] = int(round(down[param]))
                ret.append(up)
                ret.append(down)
        return ret
        
    def search(self, init_params):
        
        #initialize the min and max values to be at the range extremes
        self.min_param = {}
        self.max_param = {}
        for param in self.param_ranges.keys():
            if self.param_types[param] in NUMBER_TYPES:
                self.min_param[param] = self.param_ranges[param][0]
                self.max_param[param] = self.param_ranges[param][1]
        
        pivot = init_params
        pivot_score = self.__error_func_log(**init_params)
        while not self.stopping_criteria_reached():
        
            #first generate all possible steps and then get the best one
            best_step = None
            best_step_score = 0.0
            for step in self.generate_all_steps(pivot):
                score = self.__error_func_log(**step)
                if best_step is None or score < best_step_score:
                    best_step_score = score
                    best_step = step
                    
            #next update the pivot
            if pivot_score <= best_step_score: #if the pivot is better than any of the steps
                best_step = pivot
                #pull the min and max in around the pivot
                for param in self.min_param.keys():
                    self.min_param[param] = self.min_param[param] + (pivot[param] - self.min_param[param])/2.0
                    self.max_param[param] = pivot[param] + (self.max_param[param] - pivot[param])/2.0
                    if self.param_types[param] is int:
                        self.min_param[param] = int(round(self.min_param[param]))
                        self.max_param[param] = int(round(self.max_param[param]))
            else: #let's move the min and max based on the best step
                for param in self.min_param.keys():
                    if pivot[param] < best_step[param]: #if this parameter stepped up
                        self.min_param[param] = pivot[param]
                    elif pivot[param] > best_step[param]: #if this parameter stepped down
                        self.max_param[param] = pivot[param]
                    else: #if this paramter didn't move, then don't do anything, leave it be
                        pass
                        
            pivot = best_step
            pivot_score = self.__error_func_log(**pivot)
        return (pivot, pivot_score)
                
                
    def __recursive_nominal_variable_search(self, vals, remaining_params, init_search):
        ret = []
        if len(vals) <= 0:
            for key in vals.keys():
                init_search[key] = vals[key]
            return [self.search(init_search)]
        else:
            param = remaining_params[0]
            remaining_params = remaining_params[1:]
            for val in self.param_ranges[param]:
                vals[param] = val
                ret = ret + self.__recursive_nominal_variable_search(vals, remaining_params, init_search)
        return ret
                

    def begin_search(self):
        nominal_params = []
        for param in self.param_types.keys():
            if (not self.param_types[param] in NUMBER_TYPES):
                nominal_params.append(param)
        scores = self.__recursive_nominal_variable_search({}, nominal_params, {param : self.init_centered_param(param) for param in self.param_ranges.keys()})
        
        best_score = scores[0][1]
        best_params = scores[0][0]
        for params, score in scores:
            if score < best_score:
                best_score = score
                best_params = params
                
        for key in best_params.keys():
            if self.param_types[key] is FLOAT_LOG:
                best_params[key] = math.exp(best_params[key]) - self.param_log_shifts[key]
                
        return best_params, best_score
    
    
#args of the format list of tuple: (param_name, type=(float, int, None), range)
#where range is a pair for float or int, or a list for None. If a list, the arguments
#are assumed to be unordered. That means for every step, all arguments are considered.
def binary_search(args, error_func, stop_ratio=0.01, cache_file=None):
    s = BinarySearch(args, error_func, stop_ratio, cache_file)
    return s.begin_search()

        
   


#TODO debug
#def func(x, y):
#    return (math.log(x) + 3)*(math.log(x) + 3) + (math.log(x) + 3) + (y - 2)*(y - 2) + (y - 2)
#    
#best, score = binary_search([('x', FLOAT_LOG, (math.exp(-10.0), math.exp(10.0))), ('y', float, (-10.0, 10.0))], func, stop_ratio=0.000001)
#best['x'] = math.log(best['x'])
#print(str(best) + ", " + str(score))












        
        
        
        
