from extra.cognition import Memory
from extra.cognition import Learning
from extra.cognition import DecisionMaking


import datetime as dt

class Agent:
    def __init__(self, id_):
        self._id_ = id_

    @property
    def id_(self):
        # print("returning variables")
        return self._id_

    @id_.setter
    def id_(self, value):
        self._id_ = value


class Person(Agent):
    def __init__(self, age, name, gender, id_):
        super().__init__(id_=id_)

        self._gender = gender
        self._age = age
        self._name = name
        self.memory = Memory()
        self.learning = Learning()
        self.decisionmaking = DecisionMaking()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        pass

class TransitCard:

    def __init__(self, id_: int = -1, type_: object = ""):
        self._type = type_
        self._id_ = id_

    @property
    def id_(self):
        # print("returning variables")
        return self._id_

    @id_.setter
    def id_(self, value: int):
        self._id = value

    @property
    def type_(self, type_: str):
        return type_

    @type_.setter
    def type_(self, type_: str):
        self._type = type_


class Trip:
    def __init__(self,origin, destination, time, date):
        self._time = dt.time()
        self._date = dt.date()

class Traveller(Person):
    def __init__(self, origin, destination, id_, age = None, gender= None, name= None):
        super().__init__(age=age, gender=gender, name=name, id_ = id_)

        self._destination = destination
        self._origin = origin

        self._trips = []

        # self._transitcard = transitcard
    
    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin_(self, value):
        self._origin = value
        
    @property
    def destination(self):
        return self._destination

    @destination.setter
    def destination_(self, value):
        self._destination = value
    
    
    # @property
    # def transitcard(self):
    #     return self._transitcard
    # 
    # @transitcard.setter
    # def transitcard_(self, value):
    #     self._transitcard = value


    def travel(self, origin, destination, time: str = "", date: str = None):
        pass

class Flow:
    pass


class Trip:
    def __init__(self, transitcard: TransitCard, age: int, gender, name):
        pass
