# evolutionary_playground_library/attributes.py
import random

class AttributeGenome:
    """
    Represents the evolvable physical/physiological attributes of an agent.
    These attributes are evolved using a separate Genetic Algorithm.
    """
    def __init__(self,
                 max_hp=None,
                 aging_coeff=None,
                 speed=None,
                 hp_regen_from_food=None,
                 action_intensity_decay_rate=None,
                 action_hp_cost_factor=None,
                 reproduce_cooldown=None,
                 reproduce_hp_damage=None,
                 reproduction_chance=None):
        """
        Initializes the attribute genome.
        If specific values are not provided, they are initialized with random values
        within predefined sensible ranges.

        Args:
            max_hp (float, optional): Maximum health points.
            aging_coeff (float, optional): HP decay per age unit.
            speed (float, optional): Max movement units per step.
            hp_regen_from_food (float, optional): HP restored from eating food.
            action_intensity_decay_rate (float, optional): Rate at which action effectiveness diminishes with age.
            action_hp_cost_factor (float, optional): Base HP cost per unit of action.
        """
        self.max_hp = max_hp if max_hp is not None else random.uniform(80, 250)
        self.aging_coeff = aging_coeff if aging_coeff is not None else random.uniform(0.005, 0.02)
        self.speed = speed if speed is not None else random.uniform(0.8, 1.5)
        self.hp_regen_from_food = hp_regen_from_food if hp_regen_from_food is not None else random.uniform(15, 35)
        self.action_intensity_decay_rate = action_intensity_decay_rate if action_intensity_decay_rate is not None else random.uniform(0.005, 0.02)
        self.action_hp_cost_factor = action_hp_cost_factor if action_hp_cost_factor is not None else random.uniform(0.2, 0.8)
        self.reproduce_cooldown = reproduce_cooldown if reproduce_cooldown is not None else random.uniform(10, 100)
        self.reproduce_hp_damage = reproduce_hp_damage if reproduce_hp_damage is not None else random.uniform(self.max_hp*0.1, self.max_hp)
        self.reproduction_chance = reproduction_chance if reproduction_chance is not None else random.uniform(0.0001, 0.005)
    def mutate(self, mutation_rate=0.2, mutation_strength=0.1):
        """
        Mutates the attributes with a given probability and strength.
        Each attribute has a `mutation_rate` chance of being mutated.
        If mutated, its value is perturbed by a factor related to `mutation_strength`.

        Args:
            mutation_rate (float): Probability of each attribute mutating.
            mutation_strength (float): Factor determining the magnitude of mutation.
        """
        attributes_to_mutate = ['max_hp', 'aging_coeff', 'speed', 'hp_regen_from_food',
                                'action_intensity_decay_rate', 'action_hp_cost_factor',
                                'reproduce_hp_damage', 'reproduce_cooldown',
                                'reproduction_chance']

        for attr_name in attributes_to_mutate:
            if random.random() < mutation_rate:
                current_val = getattr(self, attr_name)
                # Mutation is a random percentage of the current value, scaled by strength
                mutation_amount = current_val * mutation_strength * random.uniform(-1, 1)
                setattr(self, attr_name, current_val + mutation_amount)

        # Clamp values to sensible predefined ranges after mutation to prevent extreme values.
        self.max_hp = max(30.0, min(self.max_hp, 250.0))
        self.aging_coeff = max(0.004, min(self.aging_coeff, 0.05))
        self.speed = max(0.2, min(self.speed, 3.0))
        self.hp_regen_from_food = max(5.0, min(self.hp_regen_from_food, 60.0))
        self.action_intensity_decay_rate = max(0.001, min(self.action_intensity_decay_rate, 0.05))
        self.action_hp_cost_factor = max(0.05, min(self.action_hp_cost_factor, 1.5))
        self.reproduce_cooldown = max(30, min(self.reproduce_cooldown, 1000))
        self.reproduce_hp_damage =  max(self.max_hp*0.1, min(self.reproduce_hp_damage, self.max_hp))
        self.reproduction_chance = max(0.0001, min(self.reproduction_chance, 0.005))

    @staticmethod
    def crossover(parent1_attrs, parent2_attrs):
        """
        Performs crossover (recombination) between two parent AttributeGenomes.
        This example uses simple averaging for each attribute. More sophisticated
        crossover methods (e.g., single-point, uniform) could be implemented.

        Args:
            parent1_attrs (AttributeGenome): The first parent.
            parent2_attrs (AttributeGenome): The second parent.

        Returns:
            AttributeGenome: A new AttributeGenome representing the child.
        """
        child_attrs = AttributeGenome() # Create a new instance for the child
        attributes_for_crossover = ['max_hp', 'aging_coeff', 'speed', 'hp_regen_from_food',
                                    'action_intensity_decay_rate', 'action_hp_cost_factor',
                                    'reproduce_hp_damage', 'reproduce_cooldown']
        for attr_name in attributes_for_crossover:
            # Averaging crossover: child's attribute is the average of parents'
            val_parent1 = getattr(parent1_attrs, attr_name)
            val_parent2 = getattr(parent2_attrs, attr_name)
            setattr(child_attrs, attr_name, (val_parent1 + val_parent2) / 2)
            # Alternative: Randomly pick one parent's attribute for the child
            # setattr(child_attrs, attr_name, getattr(random.choice([parent1_attrs, parent2_attrs]), attr_name))
        return child_attrs

    def __repr__(self):
        """String representation for easy debugging and logging."""
        return (f"Attrs(MaxHP:{self.max_hp:.1f}, AgeCoeff:{self.aging_coeff:.3f}, Spd:{self.speed:.1f}, "
                f"FoodHP:{self.hp_regen_from_food:.1f}, ActDecay:{self.action_intensity_decay_rate:.3f}, "
                f"ActCost:{self.action_hp_cost_factor:.2f})")
