# evolutionary_playground_library/agents.py
import math
import random
import neat # For neat.nn.FeedForwardNetwork
from mesa import Agent
from .attributes import AttributeGenome # Import from local package
from .config import (VIS_FOOD_COLOR, FITNESS_SURVIVAL_WEIGHT, 
                     FITNESS_FOOD_WEIGHT, FITNESS_HP_BONUS_WEIGHT,
                     FOOD_REPRODUCE_PROBABILITY, FOOD_REPRODUCE_STEP_AGE,
                     POSTMORTEM_FOOD_HP_COEF, FOOD_HP_COEF, VIS_POSTMORTEN_FOOD_COLOR,
                     ATTR_MUTATION_RATE, ATTR_MUTATION_STRENGTH, 
                     FITNESS_SURVIVAL_MEDIAN_WEIGHT, FITNESS_FOOD_MEDIAN_WEIGHT)

# Pygame is an optional dependency for drawing, only import if needed for type hinting or direct use
try:
    import pygame
except ImportError:
    pygame = None # Pygame not available, drawing will be skipped or handled differently

class EvolvingAgent(Agent):
    """
    Represents an agent in the simulation whose behavior (brain) is evolved by NEAT
    and whose physical attributes are evolved by a separate GA.
    """
    def __init__(self, unique_id, model, neat_genome_tuple, attribute_genome):
        """
        Initializes an EvolvingAgent.

        Args:
            unique_id: A unique identifier for the agent.
            model: The Mesa model instance this agent belongs to.
            neat_genome_tuple (tuple): A tuple (genome_id, neat.DefaultGenome) for the agent's brain.
            attribute_genome (AttributeGenome): The agent's physical/physiological attributes.
        """
        super().__init__(unique_id, model)
        self.neat_genome_id, self.neat_genome = neat_genome_tuple
        self.neat_genome_tuple = neat_genome_tuple
        # Create the neural network from the NEAT genome and configuration
        self.net = neat.nn.FeedForwardNetwork.create(self.neat_genome, model.neat_config_obj)
        self.attribute_genome = attribute_genome

        # Initialize agent state based on its attributes
        self.hp = self.attribute_genome.max_hp
        self.age = 0
        self.food_eaten_this_generation = 0
        self.is_alive = True
        self.steps_survived_this_generation = 0
        self.last_reproduce = model.schedule.steps
        # NEAT expects a 'fitness' attribute on the genome object itself
        self.neat_genome.fitness = 0.0

    def get_inputs(self):
        """
        Prepares the sensory inputs for the agent's neural network.
        This method is critical and highly dependent on the simulation's design.
        Inputs should be normalized to a consistent range (e.g., 0-1 or -1 to 1).

        Returns:
            tuple: A tuple of input values for the neural network.
        """
        recon_map = (Food, PostMortemFood)
        # Normalize HP: current HP / max HP
        norm_hp = self.hp / self.attribute_genome.max_hp if self.attribute_genome.max_hp > 0 else 0.0
        norm_hp = max(0.0, min(1.0, norm_hp)) # Clamp between 0 and 1

        # Normalize Age: e.g., cap at 100 steps for normalization
        norm_age = min(1.0, self.age / 100.0)

        # Sensor for nearest food: (dx_normalized, dy_normalized, distance_normalized)
        (
            nearest_food_norm_dist,
            nearest_food_norm_dx,
            nearest_food_norm_dy,
            nearest_food_norm_type
        ) = (1.0, 0.0, 0.0, -1)

        (
            nearest_evolving_norm_dist,
            nearest_evolving_norm_dx,
            nearest_evolving_norm_dy,
        ) = (1.0, 0.0, 0.0)

        if self.pos is None: # Agent might have been removed from grid
            return (
                norm_hp,
                norm_age,
                nearest_food_norm_dist,
                nearest_food_norm_dx,
                nearest_food_norm_dy,
                nearest_food_norm_type,
                nearest_evolving_norm_dist,
                nearest_evolving_norm_dx,
                nearest_evolving_norm_dy,
            )

        # Iterate through food items on the grid to find the nearest one
        closest_food, dist = self.model.get_nearest_agent_of_class(self, (Food, PostMortemFood), 20)
        if closest_food:
            (
                nearest_food_norm_dist,
                nearest_food_norm_dx,
                nearest_food_norm_dy,
            ) = self.model.get_norm_distance_betwen_2_positions(closest_food, self, dist)

            for index, food_agent in enumerate(recon_map):
                if isinstance( closest_food , food_agent):
                    nearest_food_norm_type = index/len(recon_map)
        
        closest_evolving_agent, agent_dist = self.model.get_nearest_agent_of_class(self, (EvolvingAgent), 30)
        if closest_evolving_agent:
            (
            nearest_evolving_norm_dist,
            nearest_evolving_norm_dx,
            nearest_evolving_norm_dy,
        ) = self.model.get_norm_distance_betwen_2_positions(closest_evolving_agent, self, agent_dist)
                
        # The NEAT config's `num_inputs` must match the number of elements in this tuple.
        return (
                norm_hp,
                norm_age,
                nearest_food_norm_dist,
                nearest_food_norm_dx,
                nearest_food_norm_dy,
                nearest_food_norm_type,
                nearest_evolving_norm_dist,
                nearest_evolving_norm_dx,
                nearest_evolving_norm_dy,
            )

    def get_current_action_intensity(self, predicted_intensiy):

        """
        Calculates the agent's current action intensity, which decays with age.
        Intensity starts at 1.0 (full effectiveness) and decreases based on age
        and the agent's `action_intensity_decay_rate` attribute.
        A minimum intensity (e.g., 0.1) is maintained.

        Returns:
            float: The current action intensity (between 0.1 and 1.0).
        """
        intensity = predicted_intensiy - (self.age * self.attribute_genome.action_intensity_decay_rate)
        return max(0.1, intensity) # Ensure a minimum intensity
    
    def add_offspring_to_simulation(self, child_neat_genome, child_attribute_genome, spawn_pos):
        offspring_id = self.model.get_new_agent_id()
        
        # The child_neat_genome_obj is the NEAT genome object.
        # We need its ID (key) to form the tuple for EvolvingAgent.__init__
        neat_genome_id_for_child = child_neat_genome.key if hasattr(child_neat_genome, 'key') else f"offspring_{offspring_id}"

        # **THE FIX IS HERE**: Pass the NEAT genome as a tuple (id, object)
        offspring_neat_genome_tuple = (neat_genome_id_for_child, child_neat_genome)

        offspring = EvolvingAgent(
            offspring_id,
            self.model,
            offspring_neat_genome_tuple,
            child_attribute_genome)
        

        self.model.grid.place_agent(offspring, spawn_pos)
        self.model.schedule.add(offspring)
        self.model.food_items_on_grid += 1
        
    def reproduce_asexually(self):
        """
        Asexual reproduction by a single parent.
        Creates a mutated clone of the parent.
        """
        if not self.pos:
            return

        # 1. Attribute Genome Cloning and Mutation
        # Create a new instance by copying attributes, then mutate
        child_attribute_genome = AttributeGenome(**vars(self.attribute_genome))
        child_attribute_genome.mutate(mutation_rate=ATTR_MUTATION_RATE, mutation_strength=ATTR_MUTATION_STRENGTH)

        # 2. NEAT Genome Cloning (via crossover with self) and Mutation
        main_neat_config = self.model.neat_config_obj
        genome_config = main_neat_config.genome_config

        child_neat_genome_key = self.model.get_new_agent_id()
        child_neat_genome = main_neat_config.genome_type(child_neat_genome_key)
        
        # Configure by crossing over with self to clone
        child_neat_genome.configure_crossover(self.neat_genome, self.neat_genome, genome_config)
        child_neat_genome.mutate(genome_config)

        # 3. Create and add offspring
        spawn_pos = self.model.get_random_empty_neighborhood_cell(self.pos)
        if not spawn_pos:
            return


        self.add_offspring_to_simulation(child_neat_genome, child_attribute_genome, spawn_pos)        
        if self.hp <= 0: self._handle_death()

    def step(self):
        """
        Defines the agent's behavior for a single simulation step.
        This includes aging, decision-making via its neural network, action execution,
        and interaction with the environment (like eating food).
        """
        if not self.is_alive:
            return # Dead agents do nothing

        # 1. Aging Process
        self.age += 1
        self.steps_survived_this_generation += 1
        
        # HP decay due to aging, based on agent's `aging_coeff`
        age_hp_decay = self.age * self.attribute_genome.aging_coeff
        self.hp -= age_hp_decay

        if self.hp <= 0:
            self.is_alive = False
            self.neat_genome.fitness = self.calculate_fitness() # Set final fitness upon death
            if self.pos: # If still on grid, tell model to handle removal
                self.model.handle_agent_death(self)
            return

        # 2. Neural Network Decision-Making
        inputs = self.get_inputs()
        outputs = self.net.activate(inputs) # Get outputs from the NEAT network

        # Interpret NN outputs (e.g., for movement)
        # Assuming outputs are in [0,1], map to [-1, 1] for directional signals
        move_x_signal = outputs[0] * 2 - 1
        move_y_signal = outputs[1] * 2 - 1
        move_intensity = outputs[2]

        # 3. Calculate Action and Associated HP Cost
        current_intensity = self.get_current_action_intensity(move_intensity)
        
        # Effective speed is base speed attribute scaled by current intensity
        effective_speed = self.attribute_genome.speed * current_intensity
        
        # Calculate movement delta (dx, dy)
        dx = int(round(move_x_signal * effective_speed))
        dy = int(round(move_y_signal * effective_speed))

        # HP cost for performing the action (e.g., movement)
        # Cost is proportional to distance moved and agent's `action_hp_cost_factor`
        action_magnitude = abs(dx) + abs(dy) # Simple Manhattan distance for cost
        action_hp_penalty = action_magnitude * self.attribute_genome.action_hp_cost_factor
        self.hp -= action_hp_penalty
        if self.hp <= 0:
            self.is_alive = False
            self.neat_genome.fitness = self.calculate_fitness()
            if self.pos:
                self.model.handle_agent_death(self)
            return

        # 4. Execute Movement
        if self.pos and (dx != 0 or dy != 0): # Ensure agent is on grid before moving
            new_pos = (self.pos[0] + dx, self.pos[1] + dy)
            self.model.grid.move_agent(self, self.model.grid.torus_adj(new_pos)) # Torus grid

        # 5. Interact with Environment (e.g., eat food)
        if self.pos: # Check if agent is still on the grid
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for obj in cellmates:
                #If has food in cell
                if isinstance(obj, (Food, PostMortemFood)):
                    self.hp += obj.hp_coef * self.attribute_genome.hp_regen_from_food
                    self.hp = min(self.hp, self.attribute_genome.max_hp) # Cap HP at max
                    self.model.remove_agent_from_grid_and_schedule(obj) # Food is consumed
                    self.food_eaten_this_generation += 1
                    break # Eat only one food item per step
        
        if self.model.schedule.steps - self.last_reproduce > self.attribute_genome.reproduce_cooldown and \
        random.random() < self.attribute_genome.reproduction_chance:
            self.reproduce_asexually()
            self.hp -= self.attribute_genome.reproduce_hp_damage
            self.last_reproduce = self.model.schedule.steps
            
    def calculate_fitness(self):
        """
        Calculates the final fitness score for this agent's performance in the generation.
        This fitness is used for both NEAT (brain evolution) and the attribute GA.
        The fitness function is crucial and should be designed to reward desired behaviors.

        Returns:
            float: The calculated fitness score (must be non-negative for NEAT).
        """
        # Example fitness: combines survival time, food eaten, and a bonus for remaining HP if alive.
        # Weights for each component can be tuned in config.py.
        fitness_score = (self.steps_survived_this_generation * FITNESS_SURVIVAL_WEIGHT) + \
                        (self.food_eaten_this_generation * FITNESS_FOOD_WEIGHT)
        
        if self.is_alive and self.attribute_genome.max_hp > 0:
            hp_bonus = (self.hp / self.attribute_genome.max_hp) * FITNESS_HP_BONUS_WEIGHT
            fitness_score += hp_bonus
            
        generation_median_attributes = self.model.get_agents_median_attributes()
        
        if generation_median_attributes:
            fitness_score += FITNESS_SURVIVAL_MEDIAN_WEIGHT * generation_median_attributes["median_steps_survived_this_generation"]
            fitness_score += FITNESS_FOOD_MEDIAN_WEIGHT * generation_median_attributes["median_food_eaten_this_generation"]
        
        
        return max(0.0, fitness_score) # Ensure fitness is not negative

    def draw(self, surface, x_pixel, y_pixel, cell_size):
        """
        Draws the EvolvingAgent on a Pygame surface if Pygame is available.
        The agent's color can indicate its HP (e.g., red for low, green for high).

        Args:
            surface (pygame.Surface): The Pygame surface to draw on.
            x_pixel (int): The x-coordinate (in pixels) of the top-left of the grid cell.
            y_pixel (int): The y-coordinate (in pixels) of the top-left of the grid cell.
            cell_size (int): The size (in pixels) of one grid cell.
        """
        if not pygame or not self.is_alive: # Skip drawing if Pygame not loaded or agent is dead
            return
        
        # Calculate HP ratio (0.0 to 1.0) for color coding
        hp_ratio = self.hp / self.attribute_genome.max_hp if self.attribute_genome.max_hp > 0 else 0.0
        hp_ratio = max(0.0, min(1.0, hp_ratio))

        # Interpolate color from red (low HP) to green (high HP)
        color_r = int((1 - hp_ratio) * 255)  # More red when hp_ratio is low
        color_g = int(hp_ratio * 255)        # More green when hp_ratio is high
        color_b = 0
        agent_color = (color_r, color_g, color_b)

        # Draw agent as a circle within the cell
        radius = int(cell_size * 0.4) # Agent occupies about 80% of the cell width
        center_x = x_pixel + cell_size // 2
        center_y = y_pixel + cell_size // 2
        pygame.draw.circle(surface, agent_color, (center_x, center_y), radius)
        
        # Optional: Draw a black border around the agent for better visibility
        pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y), radius, 1)


class Food(Agent):
    """
    A simple agent representing a food item in the simulation.
    Food is stationary and can be consumed by EvolvingAgents.
    """
    hp_coef = FOOD_HP_COEF

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.born_step = model.schedule.steps

    
    def step(self):
        """Food is passive and does not perform actions on its own."""
        if self.model.schedule.steps - self.born_step > FOOD_REPRODUCE_STEP_AGE and random.random() < FOOD_REPRODUCE_PROBABILITY:
            self.reproduce()
        pass
    
    def reproduce(self):
            # Try a random deslocation from the food reproducing
            empty_neighbor_cell_pos = self.model.get_random_empty_neighborhood_cell(self.pos)
            
            if not empty_neighbor_cell_pos:
                return
            # Add a food point
            food_id = self.model.get_new_agent_id()
            food = Food(food_id, self.model)
            self.model.grid.place_agent(food, empty_neighbor_cell_pos)
            self.model.schedule.add(food) # Add to schedule if food needs to act (not in this case)
            self.model.food_items_on_grid += 1

    def draw(self, surface, x_pixel, y_pixel, cell_size):
        """
        Draws the Food item on a Pygame surface if Pygame is available.
        Food is typically drawn as a small, distinct shape (e.g., a yellow square).

        Args:
            surface (pygame.Surface): The Pygame surface to draw on.
            x_pixel (int): The x-coordinate (in pixels) of the top-left of the grid cell.
            y_pixel (int): The y-coordinate (in pixels) of the top-left of the grid cell.
            cell_size (int): The size (in pixels) of one grid cell.
        """
        if not pygame: # Skip drawing if Pygame not loaded
            return
        
        # Draw food as a smaller square within the cell
        food_size = cell_size * 0.5 # Food takes up 50% of cell width/height
        offset = (cell_size - food_size) / 2 # Center the food square
        
        food_rect = pygame.Rect(x_pixel + offset, 
                                y_pixel + offset,
                                food_size, 
                                food_size)
        pygame.draw.rect(surface, VIS_FOOD_COLOR, food_rect)

class PostMortemFood(Agent):
    """
    A simple agent representing a food item in the simulation.
    Food is stationary and can be consumed by EvolvingAgents.
    """
    hp_coef = POSTMORTEM_FOOD_HP_COEF

    def __init__(self, unique_id, model):
        self.born_step = model.schedule.step
        super().__init__(unique_id, model)

    def step(self):
        """Food is passive and does not perform actions on its own."""
        pass

    def draw(self, surface, x_pixel, y_pixel, cell_size):
        """
        Draws the Food item on a Pygame surface if Pygame is available.
        Food is typically drawn as a small, distinct shape (e.g., a yellow square).

        Args:
            surface (pygame.Surface): The Pygame surface to draw on.
            x_pixel (int): The x-coordinate (in pixels) of the top-left of the grid cell.
            y_pixel (int): The y-coordinate (in pixels) of the top-left of the grid cell.
            cell_size (int): The size (in pixels) of one grid cell.
        """
        if not pygame: # Skip drawing if Pygame not loaded
            return
        
        # Draw food as a smaller square within the cell
        food_size = cell_size * 0.5 # Food takes up 50% of cell width/height
        offset = (cell_size - food_size) / 2 # Center the food square
        
        food_rect = pygame.Rect(x_pixel + offset, 
                                y_pixel + offset,
                                food_size, 
                                food_size)
        pygame.draw.rect(surface, VIS_POSTMORTEN_FOOD_COLOR, food_rect)
