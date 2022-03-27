"""
The system will make sure that the physical rules in the real world are met.
The equilibrium of the system obbeys a priori rules or assumptions about collective behavior
The system is a network in movement.
A picture of the system at a time timepoint will show the network, with flows and vehicles.
"""


""" The controller manage the movement of the vehicles in the network.
It can move them according to a predefined schedules, or optimize it by calling AI or ITS.

"""

# class Controller:
#     def __init__(self, system):
#         self._system = system
#
#     @property
#     def system(self):
#         return self._system
#
#     @system.setter
#     def system(self, value):
#         self._system = value
#
#     def create_system(self):
#
#         system.create_system(network = network, vehicles = vehicles, agents = agents):


class System:

    def __init__(self, network, vehicles, agents):
        self._network = network
        self._vehicles = vehicles
        self._agents = agents

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def vehicles(self):
        return self._vehicles

    @vehicles.setter
    def vehicles(self, value):
        self._vehicles = value

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, value):
        self._agents = value

class Equilibrium:
    def __init__(self, system: System):
        self._system_state = system
        self._equilibrium_type = None

    @property
    def system_state(self):
        return self._system_state

    @system_state.setter
    def system_state(self, value):
        _system_state = value

    @property
    def equilibrium_type(self):
        return self._equilibrium_type

    @equilibrium_type.setter
    def equilibrium_type(self, value):
        self._equilibrium_type = value

    def deterministic(self, system):

        self.equilibrium_type = 'deterministic'

        return self.system_state


    def stochastic(self, system):
        self.equilibrium_type = 'stochastic'

        return self.system_state


    def dynamic(self, system):
        self.equilibrium_type = 'dynamic'
        return self.system_state



