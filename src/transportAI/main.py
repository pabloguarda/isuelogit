"""Main module."""

# architect = Arquitect()
# node = Node(label = 1, pos = (0,0))
# edge = Edge()

from src import transportAI as tt

from transportAI.ai import Alternative

import transportAI as tt

n_origins = 58
ids = range(1, n_origins + 1)

infrastructure = tt.create_infrastructure(ids = ids, positions = None)
network = tt.create_network(infrastructure = infrastructure)
travellers = tt.create_agents()

system = tt.create_system(network = network, vehicles = vehicles, travellers = travellers)


#tt.master.build_infrastructure(nodes = len(n_origins) )

