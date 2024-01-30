from isuelogit.geographer import NodePosition

class Node:

    def __init__(self, key: 'str', position: NodePosition = None):
        '''
        :argument key: internal id
        :argument position: tuple with length 2 (x,y)
        '''

        # This is an id for internal use
        self._key = key

        #If node information is read from an external file, the id will match the id read in that file
        self._id = str()

        self._position = None

        if position is not None:
            self._position = position

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value
        
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def crs(self):
        return self._position.crs