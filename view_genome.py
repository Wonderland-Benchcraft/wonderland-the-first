import neat
import pickle
import os # For path joining

# --- Add this block at the TOP of your script ---
# Replace '/usr/bin' with the actual directory containing your 'dot' executable
# if 'which dot' showed a different path (e.g., /usr/local/bin)
graphviz_dot_dir = "/usr/bin"
current_path = os.environ.get("PATH", "")
if graphviz_dot_dir not in current_path.split(os.pathsep):
    print(f"Attempting to prepend Graphviz bin directory to PATH: {graphviz_dot_dir}")
    os.environ["PATH"] = f"{graphviz_dot_dir}{os.pathsep}{current_path}"
    print(f"Updated PATH: {os.environ['PATH']}")
# --- End of block to add ---


# --- Configuration ---
# You need the NEAT config file that was used during the simulation
# to correctly interpret the genome.
NEAT_CONFIG_FILENAME = "neat_config.txt" # Make sure this path is correct
WINNER_GENOME_FILENAME = "winner_neat_genome.pkl"
OUTPUT_IMAGE_FILENAME = "best_network.svg" # Or .png, .pdf, etc.

def visualize_genome(config_path, genome_path, output_path):
    """
    Loads a saved NEAT genome and visualizes its network structure.
    """
    # Load the NEAT configuration object
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    except Exception as e:
        print(f"Error loading NEAT config file '{config_path}': {e}")
        return

    # Load the saved genome
    try:
        with open(genome_path, 'rb') as f:
            genome = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Genome file '{genome_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading genome from '{genome_path}': {e}")
        return

    print(f"\nLoaded genome from '{genome_path}':")
    print(genome)

    # Visualize the network
    try:
        # Import the visualize module (ensure graphviz is installed)
        from neat import visualize

        # Define node names if you want custom labels (optional)
        # Based on your simulation: 5 inputs, 2 outputs
        # Inputs: norm_hp, norm_age, nearest_food_dist_norm, nearest_food_dx_norm, nearest_food_dy_norm
        # Outputs: move_x_signal, move_y_signal
        node_names = {-1: 'HP', -2: 'Age', -3: 'FoodDist', -4: 'FoodDX', -5: 'FoodDY',
                       0: 'MoveX', 1: 'MoveY'}
        # Note: Input nodes are typically negative integers, output nodes are 0, 1, ...

        visualize.draw_net(config, genome, view=False, node_names=node_names,
                           filename=output_path, show_disabled=True, fmt='svg')
        print(f"\nNetwork visualization saved to '{output_path}'")

        # If you want to try to view it immediately (might not work in all environments)
        # visualize.draw_net(config, genome, view=True, node_names=node_names, show_disabled=True)

    except ImportError:
        print("\nError: Could not import 'graphviz'. Please ensure Graphviz is installed "
              "on your system and the 'graphviz' Python package is installed (pip install graphviz).")
    except Exception as e:
        print(f"\nError during visualization: {e}")


if __name__ == '__main__':
    # Ensure paths are correct relative to where you run this script
    # If neat_config.txt is in the same directory as this script:
    script_dir = os.path.dirname(__file__) if __file__ else '.'
    config_file = os.path.join(script_dir, NEAT_CONFIG_FILENAME)
    genome_file = os.path.join(script_dir, WINNER_GENOME_FILENAME)
    output_file = os.path.join(script_dir, OUTPUT_IMAGE_FILENAME)

    visualize_genome(config_file, genome_file, output_file)