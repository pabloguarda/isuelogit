""" ALl type of infrastructure"""

""" The arquitect has the capabilities to create the entire transportation network"""

#from .infrastructure import *

# class Arquitect:
#     # def __init__(self, value):
#     #     self.value = value
#
#     def create_infrastructure(self, elements, positions = None):
#         infrastructure = Infrastructure()
#
# #Arquitect(value = 1)

class Infrastructure:
    def __init__(self, id_, positions = None):
        '''
        :argument
        '''
        self._id_ = id_
        self._positions = positions

class Station(Infrastructure):
    def __int__(self, id_, positions, capacity, position, name, access_time):
        super.__init__(id_ = id_, positions = positions)

        self._capacity = capacity
        self._position = position
        self._name = name
        self._access_time = access_time

class MetroStation(Station):
    def __int__(self, capacity, position, name, access_time):
        super.__init__(capacity, position, name, access_time)

class BusStation(Station):
    def __int__(self, capacity, position, name):
        super.__init__(capacity, position, name, access_time = 0)

