"""Grid environment for multi-agent natural selection simulation."""

import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from .creature import Creature


class GridEnvironment:
    """A 2D grid environment for creatures to interact in."""
    
    def __init__(self, 
                 width: int = 100,
                 height: int = 100,
                 num_food: int = 50,
                 food_value: float = 20.0,
                 vision_range: float = 10.0):
        """
        Initialize the grid environment.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            num_food: Number of food items to spawn
            food_value: Energy value of each food item
            vision_range: How far creatures can see
        """
        self.width = width
        self.height = height
        self.num_food = num_food
        self.food_value = food_value
        self.vision_range = vision_range
        
        # Food locations (x, y)
        self.food_locations = set()
        self.spawn_food()
        
        # Environment state
        self.time_step = 0
        
    def spawn_food(self):
        """Randomly spawn food items in the environment."""
        self.food_locations.clear()
        for _ in range(self.num_food):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            self.food_locations.add((x, y))
    
    def get_vision_input(self, creature: Creature, num_rays: int = 8) -> np.ndarray:
        """
        Get vision input for a creature using ray casting.
        
        Args:
            creature: The creature to get vision for
            num_rays: Number of vision rays
            
        Returns:
            Array of distances to nearest objects for each ray
        """
        vision = np.full(num_rays, self.vision_range)  # Initialize with max distance
        
        # Cast rays in different directions
        for i in range(num_rays):
            angle = creature.angle + (2 * np.pi * i / num_rays)
            
            # Ray direction
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Check for food along the ray
            min_distance = self.vision_range
            
            for food_x, food_y in self.food_locations:
                # Vector from creature to food
                to_food_x = food_x - creature.x
                to_food_y = food_y - creature.y
                
                # Project food position onto ray direction
                projection = to_food_x * dx + to_food_y * dy
                
                if projection > 0:  # Food is in front of creature
                    # Distance from ray to food
                    cross_product = abs(to_food_x * dy - to_food_y * dx)
                    
                    if cross_product < 2.0:  # Food is close enough to ray
                        distance = np.sqrt(to_food_x**2 + to_food_y**2)
                        if distance < min_distance:
                            min_distance = distance
            
            vision[i] = min_distance
        
        # Normalize vision input to [0, 1]
        vision = vision / self.vision_range
        
        return vision
    
    def check_food_collision(self, creature: Creature) -> float:
        """
        Check if creature collides with food and return reward.
        
        Args:
            creature: The creature to check
            
        Returns:
            Reward value (energy gained from food)
        """
        reward = 0.0
        collision_radius = 2.0
        
        food_to_remove = []
        for food_x, food_y in self.food_locations:
            distance = np.sqrt((creature.x - food_x)**2 + (creature.y - food_y)**2)
            if distance < collision_radius:
                reward += self.food_value
                creature.energy += self.food_value
                food_to_remove.append((food_x, food_y))
        
        # Remove consumed food
        for food in food_to_remove:
            self.food_locations.discard(food)
        
        return reward
    
    def check_boundaries(self, creature: Creature) -> float:
        """
        Check boundary collisions and apply penalties.
        
        Args:
            creature: The creature to check
            
        Returns:
            Penalty for hitting boundaries
        """
        penalty = 0.0
        
        # Wrap around boundaries (torus topology)
        if creature.x < 0:
            creature.x = self.width
            penalty = -5.0
        elif creature.x > self.width:
            creature.x = 0
            penalty = -5.0
            
        if creature.y < 0:
            creature.y = self.height
            penalty = -5.0
        elif creature.y > self.height:
            creature.y = 0
            penalty = -5.0
            
        return penalty
    
    def step(self, creatures: List[Creature]) -> Dict[str, Any]:
        """
        Step the environment forward one time step.
        
        Args:
            creatures: List of creatures in the environment
            
        Returns:
            Dictionary with step information
        """
        total_rewards = []
        
        for creature in creatures:
            # Get vision input
            vision = self.get_vision_input(creature)
            
            # Get action from creature's policy
            steering_angle, speed = creature.act(vision)
            
            # Update creature position
            old_x, old_y = creature.x, creature.y
            creature.update_position(steering_angle, speed)
            
            # Check collisions and boundaries
            food_reward = self.check_food_collision(creature)
            boundary_penalty = self.check_boundaries(creature)
            
            # Calculate movement reward (encourage exploration)
            distance_moved = np.sqrt((creature.x - old_x)**2 + (creature.y - old_y)**2)
            movement_reward = distance_moved * 0.1
            
            # Total reward for this step
            total_reward = food_reward + boundary_penalty + movement_reward
            creature.fitness += total_reward
            total_rewards.append(total_reward)
            
            # Add experience for potential learning
            next_vision = self.get_vision_input(creature)
            creature.add_experience(vision, (steering_angle, speed), total_reward, next_vision)
        
        # Respawn food periodically
        if len(self.food_locations) < self.num_food // 2:
            self.spawn_food()
        
        self.time_step += 1
        
        return {
            'rewards': total_rewards,
            'food_remaining': len(self.food_locations),
            'time_step': self.time_step
        }
    
    def reset(self):
        """Reset the environment."""
        self.spawn_food()
        self.time_step = 0
    
    def render(self, creatures: List[Creature], save_path: str = None):
        """
        Render the current state of the environment.
        
        Args:
            creatures: List of creatures to render
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(10, 10))
        
        # Draw food
        if self.food_locations:
            food_x, food_y = zip(*self.food_locations)
            plt.scatter(food_x, food_y, c='green', s=20, marker='o', label='Food')
        
        # Draw creatures
        if creatures:
            creature_x = [c.x for c in creatures]
            creature_y = [c.y for c in creatures]
            creature_colors = [c.fitness for c in creatures]
            
            scatter = plt.scatter(creature_x, creature_y, c=creature_colors, 
                                s=50, marker='^', cmap='viridis', label='Creatures')
            plt.colorbar(scatter, label='Fitness')
            
            # Draw creature orientations
            for creature in creatures:
                dx = 3 * np.cos(creature.angle)
                dy = 3 * np.sin(creature.angle)
                plt.arrow(creature.x, creature.y, dx, dy, 
                         head_width=1, head_length=1, fc='red', ec='red')
        
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Natural Selection Environment (Step {self.time_step})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def get_stats(self, creatures: List[Creature]) -> Dict[str, float]:
        """Get environment statistics."""
        if not creatures:
            return {}
            
        fitnesses = [c.fitness for c in creatures]
        energies = [c.energy for c in creatures]
        ages = [c.age for c in creatures]
        
        return {
            'mean_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'mean_energy': np.mean(energies),
            'mean_age': np.mean(ages),
            'food_remaining': len(self.food_locations)
        }