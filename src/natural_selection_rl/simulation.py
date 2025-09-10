"""Main simulation class demonstrating the Baldwin Effect."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import os
from .creature import Creature
from .environment import GridEnvironment
from .evolution import GeneticAlgorithm


class Simulation:
    """
    Main simulation class that demonstrates natural selection with RL learning.
    This implements the Baldwin Effect where learning within lifetime influences evolution.
    """
    
    def __init__(self,
                 population_size: int = 50,
                 environment_size: int = 100,
                 num_food: int = 50,
                 vision_rays: int = 8,
                 hidden_size: int = 16,
                 enable_lifetime_learning: bool = True,
                 learning_iterations: int = 5):
        """
        Initialize the simulation.
        
        Args:
            population_size: Number of creatures in population
            environment_size: Size of the environment grid
            num_food: Number of food items in environment
            vision_rays: Number of vision rays for creatures
            hidden_size: Hidden layer size for neural networks
            enable_lifetime_learning: Whether to allow RL learning within lifetime
            learning_iterations: Number of RL iterations per lifetime
        """
        self.population_size = population_size
        self.enable_lifetime_learning = enable_lifetime_learning
        self.learning_iterations = learning_iterations
        
        # Initialize environment
        self.environment = GridEnvironment(
            width=environment_size,
            height=environment_size,
            num_food=num_food
        )
        
        # Initialize genetic algorithm
        self.genetic_algorithm = GeneticAlgorithm(population_size=population_size)
        
        # Create initial population
        self.population = self.genetic_algorithm.create_initial_population(
            vision_rays=vision_rays,
            hidden_size=hidden_size
        )
        
        # Initialize creatures in environment
        self._place_creatures_randomly()
        
        # Statistics tracking
        self.generation_stats = []
        self.episode_stats = []
        
    def _place_creatures_randomly(self):
        """Place creatures at random positions in the environment."""
        for creature in self.population:
            creature.x = np.random.uniform(0, self.environment.width)
            creature.y = np.random.uniform(0, self.environment.height)
            creature.angle = np.random.uniform(0, 2 * np.pi)
            creature.reset()
    
    def run_episode(self, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Run a single episode (lifetime) for the current population.
        
        Args:
            max_steps: Maximum number of steps per episode
            
        Returns:
            Episode statistics
        """
        self.environment.reset()
        self._place_creatures_randomly()
        
        episode_rewards = []
        
        for step in range(max_steps):
            # Environment step
            step_info = self.environment.step(self.population)
            episode_rewards.extend(step_info['rewards'])
            
            # Remove creatures with no energy
            alive_creatures = [c for c in self.population if c.energy > 0]
            if not alive_creatures:
                break
            
            # Apply lifetime learning periodically
            if self.enable_lifetime_learning and step % 100 == 0:
                for creature in alive_creatures:
                    creature.lifetime_learning(self.learning_iterations)
        
        # Calculate final fitness for each creature
        for creature in self.population:
            # Fitness includes survival bonus
            if creature.energy > 0:
                creature.fitness += creature.energy * 0.1
                creature.fitness += creature.age * 0.05
        
        episode_stats = {
            'episode_length': step + 1,
            'survivors': len([c for c in self.population if c.energy > 0]),
            'mean_fitness': np.mean([c.fitness for c in self.population]),
            'max_fitness': np.max([c.fitness for c in self.population]),
            'total_rewards': sum(episode_rewards)
        }
        
        self.episode_stats.append(episode_stats)
        return episode_stats
    
    def run_generation(self, episodes_per_generation: int = 3, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Run a complete generation (multiple episodes followed by evolution).
        
        Args:
            episodes_per_generation: Number of episodes per generation
            max_steps: Maximum steps per episode
            
        Returns:
            Generation statistics
        """
        generation_fitness = []
        
        # Run multiple episodes for this generation
        for episode in range(episodes_per_generation):
            episode_stats = self.run_episode(max_steps)
            generation_fitness.append([c.fitness for c in self.population])
        
        # Average fitness across episodes
        avg_fitness = np.mean(generation_fitness, axis=0)
        for i, creature in enumerate(self.population):
            creature.fitness = avg_fitness[i]
        
        # Get generation stats before evolution
        gen_stats = self.genetic_algorithm.get_population_stats(self.population)
        gen_stats.update(self.environment.get_stats(self.population))
        
        # Apply adaptive mutation
        self.genetic_algorithm.adaptive_mutation(self.population)
        
        # Evolve population
        self.population = self.genetic_algorithm.evolve_population(self.population)
        
        # Store stats
        self.generation_stats.append(gen_stats)
        
        print(f"Generation {gen_stats['generation']}: "
              f"Max Fitness: {gen_stats['max_fitness']:.2f}, "
              f"Mean Fitness: {gen_stats['mean_fitness']:.2f}, "
              f"Diversity: {gen_stats['weights_diversity']:.2f}")
        
        return gen_stats
    
    def run_simulation(self, 
                      num_generations: int = 50,
                      episodes_per_generation: int = 3,
                      max_steps: int = 1000,
                      save_interval: int = 10,
                      save_directory: str = "simulation_results") -> List[Dict[str, Any]]:
        """
        Run the complete simulation for multiple generations.
        
        Args:
            num_generations: Number of generations to simulate
            episodes_per_generation: Episodes per generation
            max_steps: Maximum steps per episode
            save_interval: How often to save visualizations
            save_directory: Directory to save results
            
        Returns:
            List of generation statistics
        """
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        print(f"Starting simulation with {num_generations} generations")
        print(f"Population size: {self.population_size}")
        print(f"Lifetime learning: {self.enable_lifetime_learning}")
        
        for generation in range(num_generations):
            # Run generation
            gen_stats = self.run_generation(episodes_per_generation, max_steps)
            
            # Save visualization periodically
            if generation % save_interval == 0:
                self.visualize_population(
                    save_path=f"{save_directory}/generation_{generation:03d}.png"
                )
                self.plot_evolution_progress(
                    save_path=f"{save_directory}/evolution_progress_gen_{generation:03d}.png"
                )
        
        # Final visualization
        self.visualize_population(
            save_path=f"{save_directory}/final_generation.png"
        )
        self.plot_evolution_progress(
            save_path=f"{save_directory}/final_evolution_progress.png"
        )
        
        print(f"Simulation completed! Results saved to {save_directory}")
        return self.generation_stats
    
    def visualize_population(self, save_path: Optional[str] = None):
        """Visualize the current population in the environment."""
        self.environment.render(self.population, save_path)
    
    def plot_evolution_progress(self, save_path: Optional[str] = None):
        """Plot evolution progress over generations."""
        if not self.generation_stats:
            print("No generation stats available for plotting")
            return
        
        generations = [stats['generation'] for stats in self.generation_stats]
        max_fitness = [stats['max_fitness'] for stats in self.generation_stats]
        mean_fitness = [stats['mean_fitness'] for stats in self.generation_stats]
        diversity = [stats.get('weights_diversity', 0) for stats in self.generation_stats]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Fitness plot
        ax1.plot(generations, max_fitness, 'r-', label='Max Fitness', linewidth=2)
        ax1.plot(generations, mean_fitness, 'b-', label='Mean Fitness', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Evolution Progress: Fitness over Generations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Diversity plot
        ax2.plot(generations, diversity, 'g-', label='Weight Diversity', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Diversity')
        ax2.set_title('Population Diversity over Generations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def compare_with_without_learning(self, num_generations: int = 30) -> Dict[str, List]:
        """
        Compare evolution with and without lifetime learning to demonstrate Baldwin Effect.
        
        Args:
            num_generations: Number of generations to compare
            
        Returns:
            Comparison results
        """
        print("Running comparison: Evolution with vs without lifetime learning")
        
        # Save original settings
        original_learning = self.enable_lifetime_learning
        original_stats = self.generation_stats.copy()
        
        # Run without learning
        print("\n=== Running WITHOUT lifetime learning ===")
        self.enable_lifetime_learning = False
        self.generation_stats = []
        self.population = self.genetic_algorithm.create_initial_population()
        self._place_creatures_randomly()
        
        no_learning_stats = []
        for gen in range(num_generations):
            stats = self.run_generation(episodes_per_generation=2)
            no_learning_stats.append(stats)
        
        # Run with learning
        print("\n=== Running WITH lifetime learning ===")
        self.enable_lifetime_learning = True
        self.generation_stats = []
        self.population = self.genetic_algorithm.create_initial_population()
        self._place_creatures_randomly()
        
        with_learning_stats = []
        for gen in range(num_generations):
            stats = self.run_generation(episodes_per_generation=2)
            with_learning_stats.append(stats)
        
        # Restore original settings
        self.enable_lifetime_learning = original_learning
        self.generation_stats = original_stats
        
        # Plot comparison
        self._plot_baldwin_effect_comparison(no_learning_stats, with_learning_stats)
        
        return {
            'without_learning': no_learning_stats,
            'with_learning': with_learning_stats
        }
    
    def _plot_baldwin_effect_comparison(self, no_learning_stats: List, with_learning_stats: List):
        """Plot comparison showing Baldwin Effect."""
        generations = list(range(len(no_learning_stats)))
        
        no_learning_fitness = [stats['max_fitness'] for stats in no_learning_stats]
        with_learning_fitness = [stats['max_fitness'] for stats in with_learning_stats]
        
        plt.figure(figsize=(12, 6))
        plt.plot(generations, no_learning_fitness, 'r--', label='Without Lifetime Learning', linewidth=2)
        plt.plot(generations, with_learning_fitness, 'b-', label='With Lifetime Learning', linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Max Fitness')
        plt.title('Baldwin Effect: Impact of Lifetime Learning on Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotation explaining the effect
        plt.text(0.6, 0.2, 
                'Baldwin Effect:\nLifetime learning accelerates\nevolution by guiding selection\ntoward learnable traits',
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        print("\nBaldwin Effect Analysis:")
        print(f"Final fitness without learning: {no_learning_fitness[-1]:.2f}")
        print(f"Final fitness with learning: {with_learning_fitness[-1]:.2f}")
        improvement = ((with_learning_fitness[-1] - no_learning_fitness[-1]) / no_learning_fitness[-1]) * 100
        print(f"Improvement with learning: {improvement:.1f}%")