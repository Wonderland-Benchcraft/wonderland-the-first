# evolutionary_playground_library/config.py

# NEAT Configuration
NEAT_CONFIG_FILENAME = "neat_config.txt" # Name of the NEAT config file

# Simulation Core Parameters
GRID_WIDTH = 190
GRID_HEIGHT = 110
# INITIAL_AGENT_COUNT is typically set by NEAT's pop_size in the config file.
# We use a placeholder here if needed before NEAT config is loaded.
DEFAULT_INITIAL_AGENT_COUNT = 10
FOOD_SPAWN_PROBABILITY = 0.0001
FOOD_REPRODUCE_PROBABILITY=0.0005
FOOD_REPRODUCE_STEP_AGE=15
FOOD_HP_COEF=1
POSTMORTEM_FOOD_HP_COEF=1.2
SIMULATION_STEPS_PER_GENERATION = 5000
NUMBER_OF_NEAT_GENERATIONS = 1000

# Visualization Parameters (if Pygame is used)
VIS_ENABLED_DEFAULT = True # Set to False to run without Pygame visualization by default
VIS_CELL_SIZE = 5
VIS_FPS = 10
VIS_BACKGROUND_COLOR = (20, 20, 20)
VIS_GRID_COLOR = (50, 50, 50)
VIS_FOOD_COLOR = (255, 255, 0) # Yellow
VIS_POSTMORTEN_FOOD_COLOR = (255, 0, 0) # Red
VIS_TEXT_COLOR = (230, 230, 230) # Light gray for text
VIS_FONT_SIZE = 24

# Agent Attribute Defaults (can be tuned)
ATTR_MUTATION_RATE = 0.4
ATTR_MUTATION_STRENGTH = 0.2
ATTR_CROSSOVER_RATE = 0.6
ATTR_ELITE_SIZE = 0.1

# Fitness calculation parameters (can be tuned)
FITNESS_SURVIVAL_WEIGHT = 2
FITNESS_FOOD_WEIGHT = 6.0
FITNESS_HP_BONUS_WEIGHT = 2.0 # Applied to normalized HP if alive at end
FITNESS_SURVIVAL_MEDIAN_WEIGHT = 1.2
FITNESS_FOOD_MEDIAN_WEIGHT = 2

# --- Derived or Helper Configs (usually don't need to change these directly) ---
# Example fitness threshold for NEAT config (can be adjusted)
NEAT_FITNESS_THRESHOLD_FACTOR = 0.8 # Multiplied by (SIMULATION_STEPS_PER_GENERATION * FITNESS_FOOD_WEIGHT)

# Plotting output
PLOT_OUTPUT_DIR = "simulation_plots"