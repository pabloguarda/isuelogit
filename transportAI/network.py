""" Graphical representation of the infrastructure"""

"""
Set of route to traval between 2 stations in Beijing 
https://map.baidu.com/subway/%E5%8C%97%E4%BA%AC%E5%B8%82/@12960453.129236592,4834588.04436301,18.02z/ccode%3D131%26cname%3D%25E5%258C%2597%25E4%25BA%25AC%25E5%25B8%2582
"""

"""
Engineer designs the network
-Connect the nodes in the network
- Create a train fleet,
- Define a train schedule, etc.
The role of the engineer is making the network works as a system.
"""

import numpy as np
from .infrastructure import Infrastructure

# class Engineer:
#     def __init__(self,x,y):
#         '''
#         :argument
#         '''
#         pass

class Network:
    def __init__(self,infrastructure: list):
        '''
        :argument
        '''
        self.infrastructure = []
        self._nodes = []
        
        @property
        def infrastructure(self):
            return self.infrastructure
        
        @infrastructure.setter
        def infrastructure(self, value):
            self.infrastructure = value

        @property
        def nodes(self):
            return self.nodes

        @nodes.setter
        def nodes(self, value):
            self.nodes = value


    def random_positions(self, n):

        positions = range(0,n)

        return positions

    def create_nodes(self, infrastructure: list):

        positions = self.random_positions(n = len(self))

        for element in self.infrastructure:
            self.nodes.append(Node(label = element.label, position = positions))

        return self.nodes()
        
        

class Position:
    def __init__(self,x,y):
        '''
        :argument
        '''
        self.x = x
        self.y = y
        self.pos = (x,y)

class Node:

    def __init__(self, label, pos: Position):
        '''
        :argument label:
        :argument pos: tuple with length 2 (x,y)
        '''
        self.label = label
        self.pos = pos


class Link:
    def __init__(self, index, origin_node: Node, destination_node: Node, capacity: int):
        '''
        :argument label: index of the arc. There might be more than one arc between two nodes
        :argument flow: flow in the edge
        :argument cost: cost function of the arc, which generally depend only on the flow in the edge
        '''
        #self.label = label

class Route:
    def __init__(self, origin, destination, links: list, traveltime = -1):
        self._links = links
        self._destination = destination
        self._origin = origin
        self._traveltime = traveltime  #Initialization. Travel time must be greater than 0

    @property
    def traveltime(self):
        return self._traveltime

    @traveltime.setter
    def traveltime(self, value):
        self._traveltime = value

    def compute_travel_time(self,links):
        self.traveltime = np.sum([link.traveltime for link in links])
