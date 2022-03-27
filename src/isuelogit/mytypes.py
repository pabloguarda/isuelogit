# Do not name this module as "types" as this raises errors https://stackoverflow.com/questions/43453414/cannot-import-name-mappingproxytype-error-after-importing-functools

# Must read to avoid circular dependency for typing hints
# https://www.stefaanlippens.net/circular-imports-type-hints-python.html

# https://www.pythonsheets.com/notes/python-typing.html

from typing import Dict, List, NewType, Union, TypeVar #, TypeVar, Iterable, Tuple,
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from networks import TNetwork, DiTNetwork, MultiDiTNetwork
from nodes import Node
from links import Link

from utils import Options

from paths import Path
from geographer import NodePosition



Nodes = List[Node]
Links = List[Link]
Paths = List[Path]

Positions = List[NodePosition]

# Percentage = float
# Number between 0 and 1
Proportion = NewType("Proportion", float)

ColumnVector = np.ndarray #[List[float]]
Vector = np.ndarray #[List[float]]
# ColumnVector = np.ndarray #List[float]
Matrix = np.ndarray #List[List[int]]
# Matrix = np.ndarray[List[List[int]]]
MultidayVector = Dict[int, ColumnVector]
MultidayMatrix = Dict[int, np.ndarray]
MultiDiTNetwork = MultiDiTNetwork
DiTNetwork = DiTNetwork

# Link = list[Node]


DataFrame = pd.DataFrame

# https://stackoverflow.com/questions/52153879/how-do-i-pass-arguments-to-custom-static-type-hints-in-python-3
# TNetworks = NewType('TNetwork', [TNetwork])
TNetworks = [TNetwork]

# LogitParameters = Dict[str,float]
# FeaturesLabels = List[str]
# LogitFeatures = List[str]

Feature = str
Features = List[Feature]
Values = List[float]
# Parameter = Dict[str,float]
ParametersDict = Dict[str,float]

# Options = Dict[str, any]
# Option = Dict[str, any]
