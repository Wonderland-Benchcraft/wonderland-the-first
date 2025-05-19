# evolutionary_playground_library/evolution.py
import random
from .attributes import AttributeGenome # Import from local package
from .config import SIMULATION_STEPS_PER_GENERATION, ATTR_ELITE_SIZE, ATTR_MUTATION_RATE, ATTR_CROSSOVER_RATE
from .agents import EvolvingAgent
# This global variable will be set by the main simulation script.
# It's a common pattern for NEAT's evaluation function if not using a class-based evaluator.
# Ideally, this would be passed or accessed via a more structured context.
evolutionary_world = None 

def set_global_world(world_instance):
    """Sets the global evolutionary_world instance for NEAT evaluation."""
    global evolutionary_world
    evolutionary_world = world_instance

def eval_genomes_mesa(genomes, config):
    """
    Fitness evaluation function called by NEAT.
    It runs the Mesa simulation for one generation to evaluate the fitness of each NEAT genome.

    Args:
        genomes (list): A list of (genome_id, genome_object) tuples from NEAT.
        config (neat.Config): The NEAT configuration object.
    """
    if evolutionary_world is None:
        raise RuntimeError("EvolutionaryWorld instance not set globally for eval_genomes_mesa.")

    # The `evolutionary_world.current_attribute_genomes_map` should have been prepared
    # by the patched_eval_genomes_mesa (or the main loop) before this function is called.
    # This map associates each NEAT genome_id with its corresponding AttributeGenome.
    
    # Setup agents in the Mesa model for this evaluation round
    evolutionary_world.setup_agents_for_evaluation(genomes, evolutionary_world.current_attribute_genomes_map)

    # Run the simulation for a fixed number of steps
    for i in range(SIMULATION_STEPS_PER_GENERATION):
        # Handle early exit if visualization window is closed or model is stopped
        if evolutionary_world.screen and not evolutionary_world.visualization_active:
            print("Visualization ended during genome evaluation. Stopping early.")
            break 
        if not evolutionary_world.running:
            print("Mesa model stopped during genome evaluation. Stopping early.")
            break
        
        if len(list(filter(lambda x: isinstance(x, (EvolvingAgent)) and x.hp >=0, evolutionary_world.schedule.agents))) == 0:
            break
        evolutionary_world.step() # Advance the Mesa model by one step
    # After the simulation for this generation, calculate and assign final fitness
    # to each NEAT genome. Also, store fitness for the attribute genomes.
    for agent in evolutionary_world.schedule.agents:
        if hasattr(agent, 'neat_genome') and hasattr(agent, 'calculate_fitness'): # Check for EvolvingAgent
            final_fitness = agent.calculate_fitness()
            agent.neat_genome.fitness = final_fitness # NEAT expects fitness on the genome object
            
            # Store this fitness score, linking it to the NEAT genome_id,
            # which is also the key for its associated attribute genome.
            evolutionary_world.attribute_genome_fitness_map[agent.neat_genome_id] = final_fitness


def evolve_attributes(current_attributes_map, fitness_scores_map, 
                      pop_size, # Added pop_size to ensure correct number of new attributes
                      elite_size_ratio=ATTR_ELITE_SIZE, 
                      mutation_rate=ATTR_MUTATION_RATE, 
                      crossover_rate=ATTR_CROSSOVER_RATE):
    """
    Evolves the population of AttributeGenomes using a genetic algorithm.

    Args:
        current_attributes_map (dict): {neat_genome_id: AttributeGenome_obj} from the evaluated generation.
        fitness_scores_map (dict): {neat_genome_id: fitness_score} achieved by those attributes.
        pop_size (int): The target population size for the next generation of attributes.
        elite_size_ratio (float): Proportion of the best individuals to carry over as elites.
        mutation_rate (float): Probability of mutating attributes in offspring.
        crossover_rate (float): Probability of performing crossover between parents.

    Returns:
        list: A list of new AttributeGenome objects for the next generation.
    """
    if not current_attributes_map or not fitness_scores_map:
        print("Warning: No attributes or fitness scores provided for attribute evolution. Returning new random attributes.")
        return [AttributeGenome() for _ in range(pop_size)]

    # Create a list of (fitness, attribute_genome_obj) for sorting and selection
    population_with_fitness = []
    for genome_id, attr_genome in current_attributes_map.items():
        # Use the fitness score associated with this attribute set (via its NEAT genome_id link)
        fitness = fitness_scores_map.get(genome_id, 0.0) # Default to 0 if no fitness recorded
        population_with_fitness.append((fitness, attr_genome))

    if not population_with_fitness: # Should not happen if current_attributes_map was not empty
        return [AttributeGenome() for _ in range(pop_size)]

    # Sort individuals by fitness in descending order (higher fitness is better)
    population_with_fitness.sort(key=lambda x: x[0], reverse=True)

    next_generation_attributes_list = []

    # 1. Elitism: Carry over the best individuals to the next generation
    num_elites = int(len(population_with_fitness) * elite_size_ratio)
    for i in range(num_elites):
        if i < len(population_with_fitness): # Ensure we don't go out of bounds
            # Elites are copied directly (or could be cloned if deep copy is needed)
            elite_attr_genome = population_with_fitness[i][1]
            next_generation_attributes_list.append(AttributeGenome(**vars(elite_attr_genome))) # Create a new instance copy

    # 2. Create the rest of the population using selection, crossover, and mutation
    def tournament_selection(pop, k=3):
        """Selects an individual using k-tournament selection."""
        if not pop: return AttributeGenome() # Fallback for empty population
        best_participant = None
        for _ in range(k):
            participant = random.choice(pop)
            if best_participant is None or participant[0] > best_participant[0]: # Compare fitness
                best_participant = participant
        return best_participant[1] # Return the AttributeGenome object

    num_offspring_needed = pop_size - len(next_generation_attributes_list)
    
    # Ensure there are individuals to select from for crossover/mutation
    selectable_population = population_with_fitness if population_with_fitness else [AttributeGenome()]

    for _ in range(num_offspring_needed):
        if not selectable_population: # Safety break if no parents to select
             next_generation_attributes_list.append(AttributeGenome()) # Add a random new one
             continue

        parent1_attrs = tournament_selection(selectable_population)
        parent2_attrs = tournament_selection(selectable_population)

        # Crossover
        if random.random() < crossover_rate:
            child_attrs = AttributeGenome.crossover(parent1_attrs, parent2_attrs)
        else: # Cloning one parent (or could be mutation only)
            child_attrs = AttributeGenome(**vars(random.choice([parent1_attrs, parent2_attrs])))
            
        # Mutation
        child_attrs.mutate(mutation_rate=mutation_rate) # Uses default mutation_strength from AttributeGenome
        
        next_generation_attributes_list.append(child_attrs)

    # Ensure the list has the exact pop_size
    while len(next_generation_attributes_list) < pop_size:
        next_generation_attributes_list.append(AttributeGenome()) # Fill with new random if too few
    
    return next_generation_attributes_list[:pop_size] # Trim if too many
