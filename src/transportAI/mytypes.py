# Do not name this module as "types" as this raises errors https://stackoverflow.com/questions/43453414/cannot-import-name-mappingproxytype-error-after-importing-functools

# Must read to avoid circular dependency for typing hints
# https://www.stefaanlippens.net/circular-imports-type-hints-python.html

from typing import Dict, List, NewType, TypeVar #, TypeVar, Iterable, Tuple,
from typing import TYPE_CHECKING
import numpy as np

from networks import TNetwork
from nodes import Node
from links import Link

# https://www.pythonsheets.com/notes/python-typing.html
from paths import Path
from geographer import NodePosition

Nodes = List[Node]
Links = List[Link]
Paths = List[Path]

Positions = List[NodePosition]

# Percentage = float
# Number between 0 and 1
Proportion = NewType('float', float)

ColumnVector = np.ndarray #[List[float]]
Vector = np.ndarray #[List[float]]
# ColumnVector = np.ndarray #List[float]
Matrix = np.ndarray #List[List[int]]
# Matrix = np.ndarray[List[List[int]]]
MultidayVector = Dict[int, ColumnVector]
MultidayMatrix = Dict[int, np.ndarray]

# Link = list[Node]


# https://stackoverflow.com/questions/52153879/how-do-i-pass-arguments-to-custom-static-type-hints-in-python-3
# TNetworks = NewType('TNetwork', [TNetwork])
TNetworks = [TNetwork]

LogitParameters= Dict[str,float]
# FeaturesLabels = List[str]
LogitFeatures = List[str]

Options = Dict[str, any]
