# evolutionary_playground_library/__init__.py

# This file makes the directory a Python package.
# You can optionally expose key classes or functions for easier import.

from .config import (
    GRID_WIDTH, GRID_HEIGHT, FOOD_SPAWN_PROBABILITY,
    SIMULATION_STEPS_PER_GENERATION, NUMBER_OF_NEAT_GENERATIONS,
    NEAT_CONFIG_FILENAME, DEFAULT_INITIAL_AGENT_COUNT
)
from .attributes import AttributeGenome
from .agents import EvolvingAgent, Food
from .model import EvolutionaryWorld
from .evolution import eval_genomes_mesa, evolve_attributes, set_global_world

# You could also define a version for your library
__version__ = "0.1.0"

print("Evolutionary Playground Library Initialized")
