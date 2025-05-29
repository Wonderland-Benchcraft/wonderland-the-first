# evolutionary_playground_library/model.py
import math
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

from .agents import EvolvingAgent, Food, PostMortemFood # Import from local package
from .attributes import AttributeGenome # Import from local package
from .config import (VIS_BACKGROUND_COLOR, VIS_GRID_COLOR, VIS_FPS,
                     VIS_TEXT_COLOR, VIS_FONT_SIZE, SIMULATION_STEPS_PER_GENERATION) # MODIFIED/ADDED

# Pygame is an optional dependency for drawing
try:
    import pygame
except ImportError:
    pygame = None


class EvolutionaryWorld(Model):
    """
    The main Mesa model for the evolutionary simulation.
    It manages the grid, agents, food spawning, and simulation steps.
    It also handles visualization if Pygame components are provided.
    """
    def __init__(self, width, height, initial_agent_count_placeholder, 
                 food_spawn_prob, neat_config_obj_passed, 
                 screen=None, clock=None, cell_size=15):
        """
        Initializes the EvolutionaryWorld.

        Args:
            width (int): Width of the simulation grid.
            height (int): Height of the simulation grid.
            initial_agent_count_placeholder (int): Placeholder, actual count set by NEAT pop_size.
            food_spawn_prob (float): Probability of food spawning in an empty cell each step.
            neat_config_obj_passed (neat.Config): The loaded NEAT configuration object.
            screen (pygame.Surface, optional): The Pygame screen for visualization.
            clock (pygame.time.Clock, optional): The Pygame clock for controlling FPS.
            cell_size (int, optional): The pixel size of each grid cell for visualization.
        """
        super().__init__() # Initialize the base Mesa Model class
        self.grid = MultiGrid(width, height, torus=True) # Toroidal grid
        self.schedule = RandomActivation(self) # Agents activate in random order
        self.food_spawn_probability = food_spawn_prob
        self.neat_config_obj = neat_config_obj_passed # Store NEAT config for agent NN creation
        
        self.current_neat_genome_tuples = [] # Stores (genome_id, genome_obj) for current evaluation
        self.current_attribute_genomes_map = {} # Maps neat_genome_id to its AttributeGenome
        
        self.food_items_on_grid = 0
        self.running = True # Mesa's flag to control model execution
        self.current_simulation_step_in_generation = 0
        
        # Stores fitness scores for attribute genomes, keyed by neat_genome_id
        self.attribute_genome_fitness_map = {}
        
        self.next_agent_id_counter = 0 # For generating unique Mesa agent IDs
        
        # Stores a list of AttributeGenome prototypes for the next NEAT evaluation round
        self.attribute_genome_prototypes_for_next_eval = []

        # Spatial indexing for performance optimization
        self.spatial_grid_size = max(10, min(width, height) // 20)
        self.spatial_grid = {}
        self.agent_positions = {}  # Cache agent positions
        
        # Visualization attributes
        self.screen = screen
        self.clock = clock
        self.cell_size = cell_size
        # Visualization is active if a Pygame screen object is provided
        self.visualization_active = self.screen is not None
        self.pygame_font = None
        self.current_generation_display_num = 0 # ADDED: For displaying current gen number
        self.dirty_cells = set()  # Track cells that need redrawing

        if self.visualization_active and pygame:
            try:
                # pygame.font.init() # Ensure font module is initialized if not done globally
                self.pygame_font = pygame.font.Font(None, VIS_FONT_SIZE) # Use default system font
            except Exception as e:
                print(f"Error initializing Pygame font: {e}. Text overlay will be disabled.")
                self.pygame_font = None


    def get_new_agent_id(self):
        """Generates a new unique ID for agents."""
        self.next_agent_id_counter += 1
        return self.next_agent_id_counter

    def get_agents_median_attributes(self, attribute_genome=False):
        """
        Calculates and returns the median values for key attributes of all
        currently alive EvolvingAgent instances in the simulation.
        Also includes the median age (steps_survived_this_generation).
        """

        median_attributes = {}
        default_value = 0.0 # Or None, if you prefer for missing data
        agents = self.schedule.agents
        if attribute_genome:
            # Attributes from AttributeGenome
            attribute_genome_keys = ['max_hp', 'aging_coeff', 'speed', 'hp_regen_from_food',
                                    'action_intensity_decay_rate', 'action_hp_cost_factor']
            for attr_key in attribute_genome_keys:
                values = [getattr(agent.attribute_genome, attr_key, default_value) for agent in agents]
                median_attributes[f'median_{attr_key}'] = np.median(values) if values else default_value
            
        # Direct attributes from EvolvingAgent (like current HP and age)
        direct_agent_attributes = ['steps_survived_this_generation', 'hp', 'food_eaten_this_generation']
        for attr_key in direct_agent_attributes:
            values = [getattr(agent, attr_key, default_value) for agent in agents]
            median_attributes[f'median_{attr_key}'] = np.median(values) if values else default_value
            
        median_attributes['count_alive_evolving_agents'] = len(agents)
        return median_attributes
    
    def has_agent_class_at(self, x, y, agent_class):
        """
        Checks if there is a Food agent at the given (x, y) coordinates.

        Args:
            x (int): The x-coordinate on the grid.
            y (int): The y-coordinate on the grid.

        Returns:
            bool: True if food is present at (x,y), False otherwise.
        """
        if not self.grid.is_cell_empty((x,y)): # Check if cell is not empty first
            cell_contents = self.grid.get_cell_list_contents([(x, y)])
            for agent in cell_contents:
                if isinstance(agent, agent_class):
                    return True
        return False
    
    def get_random_empty_neighborhood_cell(self, pos):
        if not pos:
            return 
        possible_neighbors = self.grid.get_neighborhood(
            pos,
            moore=True, # Check 8 surrounding cells
            include_center=False # Don't spawn on own cell
        )
        for neighbor_pos in possible_neighbors:
            # Check if the cell is empty of other EvolvingAgents or Food
            cell_contents = self.grid.get_cell_list_contents(neighbor_pos)
            is_clear = True
            for agent_in_cell in cell_contents:
                if isinstance(agent_in_cell, (EvolvingAgent, Food, PostMortemFood)):
                    is_clear = False
                    break
            
            if is_clear:
                return neighbor_pos
                
    def get_norm_distance_betwen_2_positions(self, agent1, agent2, dist=None):
        if not dist:
            #calculate distance between 2 positions
            #Not my case when coding this :P
            return (0.0, 0.0, 0.0)
        
         # Normalize distance by the maximum possible distance on the grid (diagonal)
        max_grid_dist = math.sqrt(self.grid.width**2 + self.grid.height**2)
        nearest_dist_norm = dist / max_grid_dist if max_grid_dist > 0 else 0.0
        nearest_dist_norm = min(1.0, nearest_dist_norm) # Clamp

        # Normalized direction vector to food (if distance > 0)
        if dist > 0:
            nearest_dx_norm = (agent1.pos[0] - agent2.pos[0]) / dist
            nearest_dy_norm = (agent1.pos[1] - agent2.pos[1]) / dist

        return (nearest_dist_norm, nearest_dx_norm, nearest_dy_norm)
    
    def _get_spatial_key(self, pos):
        """Get spatial grid key for position."""
        return (pos[0] // self.spatial_grid_size, pos[1] // self.spatial_grid_size)
    
    def _update_spatial_index(self, agent, old_pos=None):
        """Update agent position in spatial index."""
        if old_pos is not None:
            old_key = self._get_spatial_key(old_pos)
            if old_key in self.spatial_grid and agent in self.spatial_grid[old_key]:
                self.spatial_grid[old_key].remove(agent)
                if not self.spatial_grid[old_key]:
                    del self.spatial_grid[old_key]
        
        if agent.pos is not None:
            new_key = self._get_spatial_key(agent.pos)
            if new_key not in self.spatial_grid:
                self.spatial_grid[new_key] = set()
            self.spatial_grid[new_key].add(agent)
            self.agent_positions[agent.unique_id] = agent.pos

    def get_nearest_agent_of_class(self, current_agent, target_class, search_radius=None):
        """
        Optimized nearest agent search using spatial indexing.
        """
        if current_agent.pos is None:
            return None, float('inf')
            
        nearest_agent_found = None
        min_dist_sq = float('inf')
        
        if search_radius is not None:
            # Use spatial indexing for limited radius searches
            current_key = self._get_spatial_key(current_agent.pos)
            search_range = (search_radius // self.spatial_grid_size) + 2
            
            possible_targets = set()
            for dx in range(-search_range, search_range + 1):
                for dy in range(-search_range, search_range + 1):
                    check_key = (current_key[0] + dx, current_key[1] + dy)
                    if check_key in self.spatial_grid:
                        possible_targets.update(self.spatial_grid[check_key])
        else:
            # Fall back to all agents for unlimited search
            possible_targets = self.schedule.agents

        for agent in possible_targets:
            if isinstance(agent, target_class) and agent.pos is not None:
                if agent.unique_id == current_agent.unique_id:
                    continue

                dx = agent.pos[0] - current_agent.pos[0]
                dy = agent.pos[1] - current_agent.pos[1]
                dist_sq = dx**2 + dy**2
                
                # Early exit if beyond search radius
                if search_radius is not None and dist_sq > search_radius**2:
                    continue

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    nearest_agent_found = agent
        
        if nearest_agent_found:
            return nearest_agent_found, math.sqrt(min_dist_sq)
        else:
            return None, float('inf')

    def get_nearest_agents_of_class(self, current_agent, target_class, count=5, search_radius=None):
        """
        Find the nearest N agents of a specific class using spatial indexing.
        
        Returns:
            list: List of (agent, distance) tuples, sorted by distance (closest first)
        """
        if current_agent.pos is None:
            return []
            
        candidates = []
        
        if search_radius is not None:
            # Use spatial indexing for limited radius searches
            current_key = self._get_spatial_key(current_agent.pos)
            search_range = (search_radius // self.spatial_grid_size) + 2
            
            possible_targets = set()
            for dx in range(-search_range, search_range + 1):
                for dy in range(-search_range, search_range + 1):
                    check_key = (current_key[0] + dx, current_key[1] + dy)
                    if check_key in self.spatial_grid:
                        possible_targets.update(self.spatial_grid[check_key])
        else:
            # Fall back to all agents for unlimited search
            possible_targets = self.schedule.agents

        for agent in possible_targets:
            if isinstance(agent, target_class) and agent.pos is not None:
                if agent.unique_id == current_agent.unique_id:
                    continue

                dx = agent.pos[0] - current_agent.pos[0]
                dy = agent.pos[1] - current_agent.pos[1]
                dist_sq = dx**2 + dy**2
                
                # Early exit if beyond search radius
                if search_radius is not None and dist_sq > search_radius**2:
                    continue

                distance = math.sqrt(dist_sq)
                candidates.append((agent, distance))
        
        # Sort by distance and return top N
        candidates.sort(key=lambda x: x[1])
        return candidates[:count]


    def spawn_food(self, num_food_to_spawn=None):
        """
        Spawns food items on the grid.
        If `num_food_to_spawn` is None, spawns food based on `food_spawn_probability`
        in empty cells. Otherwise, spawns a fixed number of food items in random empty cells.
        """
        if num_food_to_spawn is None: # Probabilistic spawning
            for x in range(self.grid.width):
                for y in range(self.grid.height):
                    # A slightly random varioation food spawn change
                    #TODO: Move to a better place
                    if self.random.random() < self.food_spawn_probability:
                        current_cell_contents = self.grid.get_cell_list_contents([(x,y)])
                        # Spawn food only if cell doesn't contain an EvolvingAgent
                        if not any(isinstance(agent, EvolvingAgent) for agent in current_cell_contents):
                            food_id = self.get_new_agent_id()
                            food = Food(food_id, self)
                            self.grid.place_agent(food, (x, y))
                            self.schedule.add(food) # Add to schedule if food needs to act (not in this case)
                            self.food_items_on_grid += 1
                            self.dirty_cells.add((x, y))
        else: # Spawn a fixed number of food items
            for _ in range(num_food_to_spawn):
                # Avoid overpopulating the grid with food
                if self.food_items_on_grid >= (self.grid.width * self.grid.height * 0.5):
                    break 
                
                # Try to find an empty cell for new food
                attempts = 0
                while attempts < 100: # Limit attempts to find an empty spot
                    pos = (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height))
                    if not any(isinstance(agent, EvolvingAgent) for agent in self.grid.get_cell_list_contents([pos])):
                        food_id = self.get_new_agent_id()
                        food = Food(food_id, self)
                        self.grid.place_agent(food, pos)
                        self.schedule.add(food)
                        self.food_items_on_grid += 1
                        self.dirty_cells.add(pos)
                        break # Found a spot
                    attempts += 1

    def setup_agents_for_evaluation(self, neat_genomes_to_eval_tuples, attribute_genomes_for_eval_map):
        """
        Resets the model state and populates the grid with new EvolvingAgents
        for a new NEAT evaluation round (generation).

        Args:
            neat_genomes_to_eval_tuples (list): List of (genome_id, neat.DefaultGenome) from NEAT.
            attribute_genomes_for_eval_map (dict): Maps genome_id to its AttributeGenome.
        """
        # Reset model state for the new generation
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.grid.width, self.grid.height, torus=True)
        self.food_items_on_grid = 0
        self.current_simulation_step_in_generation = 0
        self.next_agent_id_counter = 0 # Reset agent ID counter for this generation
        self.attribute_genome_fitness_map = {} # Clear fitness map for previous attributes
        
        # Reset spatial indexing
        self.spatial_grid = {}
        self.agent_positions = {}
        self.dirty_cells = set()

        self.current_neat_genome_tuples = neat_genomes_to_eval_tuples
        self.current_attribute_genomes_map = attribute_genomes_for_eval_map

        # Create and place new agents based on the provided NEAT and attribute genomes
        for genome_id, neat_genome in neat_genomes_to_eval_tuples:
            agent_unique_id = self.get_new_agent_id()
            
            attribute_genome_for_this_agent = attribute_genomes_for_eval_map.get(genome_id)
            if attribute_genome_for_this_agent is None:
                # This should ideally not happen if populations are managed correctly.
                print(f"Warning: No attribute genome found for NEAT genome {genome_id} during setup. Creating default.")
                attribute_genome_for_this_agent = AttributeGenome()
                # Store this emergency default to avoid repeated lookups if issue persists
                self.current_attribute_genomes_map[genome_id] = attribute_genome_for_this_agent

            agent = EvolvingAgent(agent_unique_id, self, (genome_id, neat_genome), attribute_genome_for_this_agent)
            self.schedule.add(agent)
            # Place agent at a random position on the grid
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            self._update_spatial_index(agent)
            self.dirty_cells.add((x, y))
            
        # Spawn initial food for the generation (e.g., 10% of grid cells)
        initial_food_count = None #int(self.grid.width * self.grid.height * 0.1)
        self.spawn_food(num_food_to_spawn=initial_food_count)
        self.current_simulation_step_in_generation = 0 # Ensure this is reset

    def remove_agent_from_grid_and_schedule(self, agent_to_remove):
        """
        Helper method to safely remove an agent (typically Food) from the grid and schedule.
        """
        if agent_to_remove.pos: # Check if agent is still on the grid
            self.dirty_cells.add(agent_to_remove.pos)
            self._update_spatial_index(agent_to_remove, agent_to_remove.pos)
            self.grid.remove_agent(agent_to_remove)
        if agent_to_remove in self.schedule.agents: # Check if agent is in schedule
            self.schedule.remove(agent_to_remove)
        
        if isinstance(agent_to_remove, Food):
            self.food_items_on_grid = max(0, self.food_items_on_grid - 1) # Ensure count doesn't go below zero

    def handle_agent_death(self, agent):
        """
        Handles the removal of a dead EvolvingAgent from the simulation.
        Currently, this just removes them from the grid.
        """
        if agent.pos:
            self.dirty_cells.add(agent.pos)
            # Make it a food point when dead
            post_morten_food_id = self.get_new_agent_id()
            post_morten_food = PostMortemFood(post_morten_food_id, self)
            self.grid.place_agent(post_morten_food, agent.pos)
            self.schedule.add(post_morten_food) # Add to schedule if food needs to act (not in this case)
            self.food_items_on_grid += 1
            self._update_spatial_index(post_morten_food)
        
            self._update_spatial_index(agent, agent.pos)
            self.grid.remove_agent(agent)
        # The agent remains in the schedule until the end of the generation for fitness collection,
        # but its `is_alive` flag prevents further actions.

    def draw_world(self):
        """
        Optimized drawing that only redraws changed areas of the simulation.
        """
        if not self.screen or not pygame:
            return

        # Only clear and redraw if there are dirty cells or this is the first frame
        if not hasattr(self, '_first_draw_done'):
            self.screen.fill(VIS_BACKGROUND_COLOR)
            self._draw_grid_lines()
            self._first_draw_done = True
            # Force redraw everything on first frame
            for x_grid in range(self.grid.width):
                for y_grid in range(self.grid.height):
                    self.dirty_cells.add((x_grid, y_grid))

        # Only redraw dirty cells
        for x_grid, y_grid in self.dirty_cells:
            self._draw_cell(x_grid, y_grid)
        
        self.dirty_cells.clear()
        
        # Always update metrics overlay
        self._draw_metrics_overlay()
        pygame.display.flip()

    def _draw_grid_lines(self):
        """Draw the grid lines once."""
        for x_coord in range(self.grid.width + 1):
            pygame.draw.line(self.screen, VIS_GRID_COLOR, 
                             (x_coord * self.cell_size, 0), 
                             (x_coord * self.cell_size, self.grid.height * self.cell_size))
        for y_coord in range(self.grid.height + 1):
            pygame.draw.line(self.screen, VIS_GRID_COLOR, 
                             (0, y_coord * self.cell_size), 
                             (self.grid.width * self.cell_size, y_coord * self.cell_size))

    def _draw_cell(self, x_grid, y_grid):
        """Draw a single cell and its contents."""
        x_pixel = x_grid * self.cell_size
        y_pixel = y_grid * self.cell_size
        
        # Clear the cell
        cell_rect = pygame.Rect(x_pixel, y_pixel, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, VIS_BACKGROUND_COLOR, cell_rect)
        
        # Redraw grid lines for this cell
        pygame.draw.rect(self.screen, VIS_GRID_COLOR, cell_rect, 1)
        
        # Draw agents/food in this cell
        cell_contents = self.grid.get_cell_list_contents([(x_grid, y_grid)])
        for agent in cell_contents:
            if hasattr(agent, 'draw'):
                agent.draw(self.screen, x_pixel, y_pixel, self.cell_size)

    def _draw_metrics_overlay(self):
        """Draw the metrics overlay."""
        if not self.pygame_font:
            return
            
        # Clear the overlay area
        overlay_height = VIS_FONT_SIZE * 5
        overlay_rect = pygame.Rect(0, 0, 300, overlay_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), overlay_rect)
        
        y_offset = 5
        texts_to_render = [
            f"Generation: {self.current_generation_display_num}",
            f"Step: {self.current_simulation_step_in_generation}/{SIMULATION_STEPS_PER_GENERATION}",
            f"Alive Agents: {sum(1 for a in self.schedule.agents if isinstance(a, EvolvingAgent) and a.is_alive)}",
            f"Food Items: {self.food_items_on_grid}"
        ]
        for text_str in texts_to_render:
            try:
                text_surface = self.pygame_font.render(text_str, True, VIS_TEXT_COLOR)
                self.screen.blit(text_surface, (5, y_offset))
                y_offset += VIS_FONT_SIZE - 5
            except Exception as e:
                print(f"Error rendering text: {e}")
                self.pygame_font = None
                break

    def step(self):
        """
        Advances the model by one time step.
        This includes handling Pygame events (if visualizing), spawning food,
        activating agents, and updating the visualization.
        """
        if self.screen and self.visualization_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # User closed the Pygame window
                    self.visualization_active = False # Signal to stop drawing
                    self.running = False # Signal Mesa model to stop
                    print("Pygame window closed by user. Simulation will stop after this generation.")
                    return # Exit step early

        if not self.running: # If Mesa simulation itself should stop (e.g., due to Pygame quit)
            return

        """ # Dynamically spawn more food if it becomes scarce relative to living agents
        living_agents_count = sum(1 for agent in self.schedule.agents if isinstance(agent, EvolvingAgent) and agent.is_alive)
        if living_agents_count > 0 and self.food_items_on_grid < (living_agents_count * 1.5):
             self.spawn_food(num_food_to_spawn=max(1, living_agents_count // 2)) # Spawn at least 1 if needed
 """
        self.schedule.step() # Activate all scheduled agents (calls their step() method)
        self.current_simulation_step_in_generation += 1

        # Update visualization if active
        if self.screen and self.visualization_active:
            self.draw_world()
            if self.clock: # Ensure clock is available
                # MODIFIED: Directly use the imported VIS_FPS from config
                self.clock.tick(VIS_FPS) 
