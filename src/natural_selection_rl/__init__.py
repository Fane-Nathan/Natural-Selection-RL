"""Natural Selection RL - Baldwin Effect Simulation"""

from .creature import Creature
from .environment import GridEnvironment
from .evolution import GeneticAlgorithm
from .simulation import Simulation

__version__ = "0.1.0"
__all__ = ["Creature", "GridEnvironment", "GeneticAlgorithm", "Simulation"]