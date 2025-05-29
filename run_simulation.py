# run_simulation.py
import os
import neat
import pickle
import numpy as np 
import copy # For deepcopying the loaded genome

# Import from the local library package
from evolutionary_playground_library import (
    AttributeGenome,
    EvolutionaryWorld,
    eval_genomes_mesa, 
    evolve_attributes,
    set_global_world,  
    config as sim_config 
)

# Pygame is optional, attempt import for visualization
try:
    import pygame
except ImportError:
    pygame = None
    print("Pygame not found. Visualization will be disabled.")
    sim_config.VIS_ENABLED_DEFAULT = False 

# Matplotlib for plotting - optional, attempt import
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
except ImportError:
    plt = None
    print("Matplotlib not found. End-of-run plotting will be disabled.")


evolutionary_world_instance = None

def ensure_neat_config(config_path, pop_size, num_inputs, num_outputs):
    if os.path.exists(config_path):
        os.remove(config_path) 
    print(f"Creating or overwriting NEAT config file at '{config_path}'...")
    
    with open(config_path, 'w') as f:
        # [NEAT] section: Global NEAT parameters
        f.write("[NEAT]\n")
        f.write("fitness_criterion = max\n")
        example_threshold = (sim_config.SIMULATION_STEPS_PER_GENERATION * sim_config.FITNESS_FOOD_WEIGHT * sim_config.NEAT_FITNESS_THRESHOLD_FACTOR)
        f.write(f"fitness_threshold = {example_threshold}\n")
        f.write(f"pop_size = {pop_size}\n") 
        f.write("reset_on_extinction = False\n\n")

        # [DefaultGenome] section: Parameters for individual genomes
        f.write("[DefaultGenome]\n")
        f.write(f"num_inputs = {num_inputs}\n")
        f.write(f"num_outputs = {num_outputs}\n")
        
        f.write("activation_default = sigmoid\n")
        f.write("activation_mutate_rate = 0.1\n")
        f.write("activation_options = sigmoid tanh relu clamped\n")
        
        f.write("aggregation_default = sum\n")
        f.write("aggregation_mutate_rate = 0.0\n")
        f.write("aggregation_options = sum\n")
        
        f.write("bias_init_mean = 0.0\n")
        f.write("bias_init_stdev = 1.0\n")
        f.write("bias_max_value = 5.0\n")
        f.write("bias_min_value = -5.0\n")
        f.write("bias_mutate_power = 0.3\n")
        f.write("bias_mutate_rate = 0.6\n")
        f.write("bias_replace_rate = 0.05\n")
        
        f.write("compatibility_disjoint_coefficient = 1.0\n")
        f.write("compatibility_weight_coefficient = 0.4\n")
        
        f.write("conn_add_prob = 0.15\n")
        f.write("conn_delete_prob = 0.1\n")
        
        f.write("enabled_default = True\n")
        f.write("enabled_mutate_rate = 0.05\n")
        
        f.write("feed_forward = True\n") 
        f.write("initial_connection = partial_direct 0.5\n") 
        
        f.write("node_add_prob = 0.1\n")
        f.write("node_delete_prob = 0.05\n")
        
        f.write("num_hidden = 0\n") 
        
        f.write("response_init_mean = 1.0\n")
        f.write("response_init_stdev = 0.5\n")
        f.write("response_max_value = 3.0\n")
        f.write("response_min_value = -3.0\n")
        f.write("response_mutate_power = 0.2\n")
        f.write("response_mutate_rate = 0.3\n")
        f.write("response_replace_rate = 0.05\n")
        
        f.write("weight_init_mean = 0.0\n")
        f.write("weight_init_stdev = 1.0\n")
        f.write("weight_max_value = 3.0\n")
        f.write("weight_min_value = -3.0\n")
        f.write("weight_mutate_power = 0.3\n")
        f.write("weight_mutate_rate = 0.7\n")
        f.write("weight_replace_rate = 0.05\n")

        f.write("single_structural_mutation = True\n") 
        
        f.write("\n[DefaultSpeciesSet]\n")
        f.write("compatibility_threshold = 3.5\n\n")
        
        f.write("[DefaultStagnation]\n")
        f.write("species_fitness_func = max\n")
        f.write("max_stagnation = 15\n") 
        f.write("species_elitism = 1\n\n")
        
        f.write("[DefaultReproduction]\n")
        f.write("elitism = 1\n") 
        f.write("survival_threshold = 0.25\n") 
    print(f"NEAT config file '{config_path}' created/overwritten successfully.")

def patched_eval_genomes_for_neat(genomes, config_neat):
    global evolutionary_world_instance 
    if evolutionary_world_instance is None:
        raise RuntimeError("EvolutionaryWorld instance is not initialized.")
    temp_map_for_this_eval = {}
    prototypes = evolutionary_world_instance.attribute_genome_prototypes_for_next_eval
    for i, (gid, g_obj) in enumerate(genomes):
        if i < len(prototypes):
            temp_map_for_this_eval[gid] = prototypes[i]
        else:
            print(f"Warning: Not enough attribute prototypes for NEAT genome {gid} (index {i}). Using default.")
            temp_map_for_this_eval[gid] = AttributeGenome()
    evolutionary_world_instance.current_attribute_genomes_map = temp_map_for_this_eval
    eval_genomes_mesa(genomes, config_neat)

def generate_and_save_plots(generation_numbers, neat_stats_history, attribute_stats_history, output_dir):
    if not plt:
        print("Matplotlib not available. Skipping plot generation.")
        return
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating plot directory {output_dir}: {e}. Plots will not be saved.")
            return
    
    plt.figure(figsize=(12, 7)) 
    plt.plot(generation_numbers, neat_stats_history['best_fitness'], label="Best Fitness", marker='o', linestyle='-')
    plt.plot(generation_numbers, neat_stats_history['avg_fitness'], label="Average Fitness", marker='x', linestyle='--')
    plt.title("NEAT: Fitness over Generations", fontsize=16)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout() 
    plot_path = os.path.join(output_dir, "neat_fitness_over_generations.png")
    try:
        plt.savefig(plot_path)
        print(f"Saved NEAT fitness plot to {plot_path}")
    except Exception as e:
        print(f"Error saving NEAT fitness plot: {e}")
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.plot(generation_numbers, neat_stats_history['num_species'], label="Number of Species", marker='s', color='green')
    plt.title("NEAT: Number of Species over Generations", fontsize=16)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Number of Species", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "neat_species_over_generations.png")
    try:
        plt.savefig(plot_path)
        print(f"Saved NEAT species plot to {plot_path}")
    except Exception as e:
        print(f"Error saving NEAT species plot: {e}")
    plt.close()

    for attr_name_internal, values in attribute_stats_history.items():
        if values: 
            readable_attr_name = attr_name_internal.replace('avg_', '').replace('_', ' ').title()
            plt.figure(figsize=(12, 7))
            plt.plot(generation_numbers, values, label=f"Average {readable_attr_name}", marker='.')
            plt.title(f"Average {readable_attr_name} over Generations", fontsize=16)
            plt.xlabel("Generation", fontsize=14)
            plt.ylabel(f"Average {readable_attr_name}", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"attribute_{attr_name_internal}_over_generations.png")
            try:
                plt.savefig(plot_path)
                print(f"Saved attribute plot for {readable_attr_name} to {plot_path}")
            except Exception as e:
                print(f"Error saving attribute plot for {readable_attr_name}: {e}")
            plt.close()

# MODIFIED: Added start_from_genome_path parameter
def run_simulation(neat_config_filepath, 
                   enable_visualization_param=sim_config.VIS_ENABLED_DEFAULT,
                   start_from_genome_path=None): 
    global evolutionary_world_instance

    pygame_screen = None
    pygame_clock = None
    pygame_is_initialized_this_run = False

    if enable_visualization_param and pygame:
        try:
            pygame.init()
            pygame_is_initialized_this_run = True
            screen_width = sim_config.GRID_WIDTH * sim_config.VIS_CELL_SIZE
            screen_height = sim_config.GRID_HEIGHT * sim_config.VIS_CELL_SIZE
            pygame_screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Mesa & NEAT Evolutionary Simulation")
            pygame_clock = pygame.time.Clock()
        except Exception as e:
            print(f"Error initializing Pygame: {e}. Visualization will be disabled.")
            pygame_is_initialized_this_run = False
            enable_visualization_param = False
            if pygame and pygame.get_init():
                pygame.quit()
    elif enable_visualization_param and not pygame:
        print("Visualization was requested, but Pygame is not available. Running without visualization.")
        enable_visualization_param = False

    neat_configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     neat_config_filepath)

    evolutionary_world_instance = EvolutionaryWorld(
        width=sim_config.GRID_WIDTH, height=sim_config.GRID_HEIGHT,
        initial_agent_count_placeholder=sim_config.DEFAULT_INITIAL_AGENT_COUNT,
        food_spawn_prob=sim_config.FOOD_SPAWN_PROBABILITY,
        neat_config_obj_passed=neat_configuration,
        screen=pygame_screen, clock=pygame_clock, cell_size=sim_config.VIS_CELL_SIZE
    )
    set_global_world(evolutionary_world_instance)

    neat_pop = neat.Population(neat_configuration) # Creates initial random population
    actual_pop_size = neat_configuration.pop_size 
    print(f"NEAT population size: {actual_pop_size}")

    if start_from_genome_path and os.path.exists(start_from_genome_path):
        print(f"Attempting to start simulation from saved genome: {start_from_genome_path}")
        try:
            with open(start_from_genome_path, 'rb') as f:
                loaded_genome = pickle.load(f)
            
            genome_to_insert = copy.deepcopy(loaded_genome) # Use a copy

            # --- THIS IS THE CRUCIAL FIX ---
            # Assign the current run's genome_config to the loaded genome.
            # neat_configuration is the main neat.Config object for the current run.
            # neat_configuration.genome_config holds the parameters for genomes.
            genome_to_insert.config = neat_configuration.genome_config
            # --- END OF FIX ---

            # Basic compatibility check (can now safely access genome_to_insert.config)
            if not (genome_to_insert.config.num_inputs == neat_configuration.genome_config.num_inputs and \
                    genome_to_insert.config.num_outputs == neat_configuration.genome_config.num_outputs):
                print("Warning: Loaded genome's input/output node count in its *original* config " +
                      "differs from current config. This might lead to issues if a full compatibility check is not performed. " +
                      "The genome will now use the current run's config parameters.")
            # More robust checks could compare `input_keys` and `output_keys` if necessary.

            if neat_pop.population: # Ensure population is not empty
                ids_to_change = list(neat_pop.population.keys())[0: 2*int(len(neat_pop.population.keys())/3)]
                
                for genome_id in ids_to_change:
                    print(f"Replacing {int(4*len(neat_pop.population.keys())/4)} genomes")
                    genome_to_insert_2 = copy.deepcopy(loaded_genome) # Use a copy
                    genome_to_insert_2.key = genome_id # Assign new key from the current population
                    genome_to_insert_2.fitness = None # Reset fitness for re-evaluation
                
                    neat_pop.population[genome_id] = genome_to_insert_2
                    print(f"Successfully replaced genome {genome_id} with loaded genome (now using current config).")

                neat_pop.species.speciate(neat_configuration, neat_pop.population, neat_pop.generation) # neat_pop.generation is 0 here
                print("Re-speciated population after inserting loaded genome.")
            else:
                print("Warning: New population is empty, cannot insert loaded genome.")

        except Exception as e:
            print(f"Error loading or inserting saved genome: {e}. Starting with a random population.")
    elif start_from_genome_path:
        print(f"Warning: Specified start_from_genome_path '{start_from_genome_path}' not found. Starting with a random population.")


    neat_pop.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter() 
    neat_pop.add_reporter(stats_reporter)
    
    evolutionary_world_instance.attribute_genome_prototypes_for_next_eval = \
        [AttributeGenome() for _ in range(actual_pop_size)]

    best_neat_genome_overall = None

    generation_numbers_for_plot = []
    neat_stats_history_for_plot = {'best_fitness': [], 'avg_fitness': [], 'num_species': []}
    attribute_keys_to_track = ['max_hp', 'speed', 'aging_coeff', 'hp_regen_from_food', 
                               'action_intensity_decay_rate', 'action_hp_cost_factor']
    attribute_stats_history_for_plot = {f'avg_{key}': [] for key in attribute_keys_to_track}


    for gen_num in range(sim_config.NUMBER_OF_NEAT_GENERATIONS):
        current_generation_number_for_display = gen_num + 1 
        print(f"\n--- Starting NEAT Generation {current_generation_number_for_display}/{sim_config.NUMBER_OF_NEAT_GENERATIONS} ---")
        evolutionary_world_instance.current_generation_display_num = current_generation_number_for_display

        if enable_visualization_param and pygame_is_initialized_this_run:
            evolutionary_world_instance.screen = pygame_screen
            evolutionary_world_instance.clock = pygame_clock
            evolutionary_world_instance.visualization_active = True
            evolutionary_world_instance.running = True 
        else:
            evolutionary_world_instance.screen = None 
            evolutionary_world_instance.clock = None
            evolutionary_world_instance.visualization_active = False

        winner_this_gen = neat_pop.run(patched_eval_genomes_for_neat, 1) 

        if enable_visualization_param and pygame_is_initialized_this_run and not evolutionary_world_instance.running:
            print("Simulation run interrupted by closing visualization window.")
            break 

        if best_neat_genome_overall is None or \
           (winner_this_gen and winner_this_gen.fitness is not None and \
            (best_neat_genome_overall.fitness is None or winner_this_gen.fitness > best_neat_genome_overall.fitness)):
            best_neat_genome_overall = winner_this_gen
            print(f"New best NEAT genome found in generation {current_generation_number_for_display} with fitness: {best_neat_genome_overall.fitness:.2f}")

        generation_numbers_for_plot.append(current_generation_number_for_display)
        current_best_fitness_this_gen = 0.0
        if stats_reporter.most_fit_genomes: 
            best_genome_of_gen = stats_reporter.most_fit_genomes[-1] 
            if best_genome_of_gen and hasattr(best_genome_of_gen, 'fitness') and best_genome_of_gen.fitness is not None:
                current_best_fitness_this_gen = best_genome_of_gen.fitness
        neat_stats_history_for_plot['best_fitness'].append(current_best_fitness_this_gen)
        
        current_avg_fitness_this_gen = 0.0
        fitness_means = stats_reporter.get_fitness_mean() 
        if fitness_means: 
            current_avg_fitness_this_gen = fitness_means[-1]
        neat_stats_history_for_plot['avg_fitness'].append(current_avg_fitness_this_gen)

        species_set = neat_pop.species 
        neat_stats_history_for_plot['num_species'].append(len(species_set.species))
        
        evaluated_attributes_list = list(evolutionary_world_instance.current_attribute_genomes_map.values())
        if evaluated_attributes_list:
            for attr_key in attribute_keys_to_track:
                try:
                    avg_val = np.mean([getattr(ag, attr_key, 0) for ag in evaluated_attributes_list])
                    attribute_stats_history_for_plot[f'avg_{attr_key}'].append(avg_val)
                except AttributeError: 
                    attribute_stats_history_for_plot[f'avg_{attr_key}'].append(0) 
        else: 
             for attr_key in attribute_keys_to_track:
                attribute_stats_history_for_plot[f'avg_{attr_key}'].append(0)

        print(f"\n--- Evolving Attributes after NEAT Generation {current_generation_number_for_display} ---")
        new_attribute_prototypes_list = evolve_attributes(
            current_attributes_map=evolutionary_world_instance.current_attribute_genomes_map,
            fitness_scores_map=evolutionary_world_instance.attribute_genome_fitness_map,
            pop_size=actual_pop_size,
            elite_size_ratio=sim_config.ATTR_ELITE_SIZE, 
            mutation_rate=sim_config.ATTR_MUTATION_RATE,
            crossover_rate=sim_config.ATTR_CROSSOVER_RATE
        )
        evolutionary_world_instance.attribute_genome_prototypes_for_next_eval = new_attribute_prototypes_list


    if best_neat_genome_overall:
        print('\nBest NEAT genome found overall:\n{!s}'.format(best_neat_genome_overall))
        with open('winner_neat_genome.pkl', 'wb') as f:
            pickle.dump(best_neat_genome_overall, f)
        print("Saved best NEAT genome to winner_neat_genome.pkl")
    
    if pygame_is_initialized_this_run and pygame and pygame.get_init():
        pygame.quit()
        
    print("Simulation finished.")

    if plt and generation_numbers_for_plot: 
        generate_and_save_plots(generation_numbers_for_plot, neat_stats_history_for_plot, 
                                attribute_stats_history_for_plot, sim_config.PLOT_OUTPUT_DIR)


if __name__ == '__main__':
    num_nn_inputs = 37  # 2 (basic) + 5*4 (foods) + 5*3 (agents) = 37 inputs
    num_nn_outputs = 3
    ensure_neat_config(
        config_path=sim_config.NEAT_CONFIG_FILENAME,
        pop_size=sim_config.DEFAULT_INITIAL_AGENT_COUNT, 
        num_inputs=num_nn_inputs,
        num_outputs=num_nn_outputs
    )
    
    # Example: To start from a saved genome (if it exists)
    # previous_winner_path = "winner_neat_genome.pkl" # Or path to your specific pkl
    # run_simulation(sim_config.NEAT_CONFIG_FILENAME, 
    #                enable_visualization_param=True,
    #                start_from_genome_path=previous_winner_path)

    # To start a fresh simulation:
    run_simulation(
        sim_config.NEAT_CONFIG_FILENAME,
        enable_visualization_param=True,
        start_from_genome_path="winner_neat_genome.pkl") 
