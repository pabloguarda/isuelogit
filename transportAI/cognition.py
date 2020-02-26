"""
Cognition has methods to model human decision making, human learning or any other cognitive process
"""

#import pylogit as pl
import numpy as np


# class ChoiceSet:
#     def __init__(self):
#         pass

class Alternative:
    def __init__(self, attribute: dict):
        pass

class DecisionMaking:

    # def __init__(self, alternatives):
    #     self.alternatives = alternatives

    def multinomiallogit(self, preferences):
        # "TODO: implement basic model of route choice"
        pass

    def heuristics(self):
        pass

    def logit_choice(self, alternatives, preferences):

        sum_exp = np.exp(preferences*alternatives)

        np.exp(np.argmax(1))/np.exp(preferences*alternatives)

    def choice_set_generation(self):
        pass



class Learning:
    def __init__(self):
        pass

class Memory:
    def __init__(self):
        pass