class Position:
    def __init__(self,x,y):
        '''
        :argument
        '''
        self.x = x
        self.y = y

class Node:

    def __init__(self, label, pos):
        '''
        :argument label:
        :argument pos: position with
        '''
        self.label = label
        self.pos = pos


class Edge:
    def __init__(self, label, flow, cost):
        '''
        :argument label: name of the node
        :argument flow: flow in the edge
        :argument cost: cost function of the arc, which generally depend only on the flow in the edge
        '''
        self.label = label




