"""Genetic Algorithm for evolving creature neural network weights."""

import numpy as np
from typing import List, Tuple
from .creature import Creature


class GeneticAlgorithm:
    """Simple genetic algorithm for evolving neural network weights."""
    
    def __init__(self,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.2,
                 crossover_rate: float = 0.7,
                 elitism_ratio: float = 0.1):
        """
        Initialize genetic algorithm parameters.
        
        Args:
            population_size: Number of creatures in population
            mutation_rate: Probability of mutation per weight
            mutation_strength: Standard deviation of mutation noise
            crossover_rate: Probability of crossover between parents
            elitism_ratio: Fraction of best individuals to preserve
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.generation = 0
        
    def create_initial_population(self, 
                                vision_rays: int = 8,
                                hidden_size: int = 16,
                                learning_rate: float = 0.01) -> List[Creature]:
        """
        Create initial random population.
        
        Args:
            vision_rays: Number of vision rays for creatures
            hidden_size: Hidden layer size for neural networks
            learning_rate: Learning rate for lifetime learning
            
        Returns:
            List of randomly initialized creatures
        """
        population = []
        for _ in range(self.population_size):
            creature = Creature(
                vision_rays=vision_rays,
                hidden_size=hidden_size,
                learning_rate=learning_rate
            )
            population.append(creature)
        return population
    
    def selection(self, population: List[Creature]) -> List[Creature]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            population: Current population of creatures
            
        Returns:
            Selected parents for next generation
        """
        parents = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament = np.random.choice(population, tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
            
        return parents
    
    def crossover(self, parent1: Creature, parent2: Creature) -> Tuple[Creature, Creature]:
        """
        Create offspring using uniform crossover.
        
        Args:
            parent1: First parent creature
            parent2: Second parent creature
            
        Returns:
            Two offspring creatures
        """
        # Get parent weights
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Create offspring weight arrays
        offspring1_weights = np.copy(weights1)
        offspring2_weights = np.copy(weights2)
        
        if np.random.random() < self.crossover_rate:
            # Uniform crossover
            crossover_mask = np.random.random(len(weights1)) < 0.5
            
            offspring1_weights[crossover_mask] = weights2[crossover_mask]
            offspring2_weights[crossover_mask] = weights1[crossover_mask]
        
        # Create offspring creatures
        offspring1 = Creature(
            vision_rays=parent1.vision_rays,
            hidden_size=parent1.hidden_size,
            learning_rate=parent1.learning_rate,
            weights=offspring1_weights
        )
        
        offspring2 = Creature(
            vision_rays=parent2.vision_rays,
            hidden_size=parent2.hidden_size,
            learning_rate=parent2.learning_rate,
            weights=offspring2_weights
        )
        
        return offspring1, offspring2
    
    def mutate(self, creature: Creature) -> Creature:
        """
        Apply mutation to a creature's weights.
        
        Args:
            creature: Creature to mutate
            
        Returns:
            Mutated creature
        """
        weights = creature.get_weights()
        
        # Apply mutation
        mutation_mask = np.random.random(len(weights)) < self.mutation_rate
        mutation_noise = np.random.normal(0, self.mutation_strength, len(weights))
        weights[mutation_mask] += mutation_noise[mutation_mask]
        
        # Create mutated creature
        mutated_creature = Creature(
            vision_rays=creature.vision_rays,
            hidden_size=creature.hidden_size,
            learning_rate=creature.learning_rate,
            weights=weights
        )
        
        return mutated_creature
    
    def evolve_population(self, population: List[Creature]) -> List[Creature]:
        """
        Evolve population for one generation.
        
        Args:
            population: Current population
            
        Returns:
            Next generation population
        """
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Elitism: preserve best individuals
        elite_count = int(self.population_size * self.elitism_ratio)
        next_generation = []
        
        # Add elite individuals
        for i in range(elite_count):
            elite_copy = population[i].copy()
            elite_copy.reset()  # Reset stats for next generation
            next_generation.append(elite_copy)
        
        # Generate rest of population through selection, crossover, and mutation
        while len(next_generation) < self.population_size:
            # Selection
            parents = self.selection(population)
            
            # Crossover and mutation
            for i in range(0, len(parents) - 1, 2):
                if len(next_generation) >= self.population_size:
                    break
                    
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                # Reset offspring stats
                offspring1.reset()
                offspring2.reset()
                
                # Add to next generation
                if len(next_generation) < self.population_size:
                    next_generation.append(offspring1)
                if len(next_generation) < self.population_size:
                    next_generation.append(offspring2)
        
        self.generation += 1
        return next_generation[:self.population_size]
    
    def get_population_stats(self, population: List[Creature]) -> dict:
        """Get statistics about the current population."""
        if not population:
            return {}
            
        fitnesses = [c.fitness for c in population]
        weights_diversity = self._calculate_weights_diversity(population)
        
        return {
            'generation': self.generation,
            'population_size': len(population),
            'mean_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'std_fitness': np.std(fitnesses),
            'weights_diversity': weights_diversity
        }
    
    def _calculate_weights_diversity(self, population: List[Creature]) -> float:
        """Calculate genetic diversity in population based on weight differences."""
        if len(population) < 2:
            return 0.0
            
        # Sample a subset for efficiency
        sample_size = min(10, len(population))
        sample = np.random.choice(population, sample_size, replace=False)
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                weights_i = sample[i].get_weights()
                weights_j = sample[j].get_weights()
                distance = np.linalg.norm(weights_i - weights_j)
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def adaptive_mutation(self, population: List[Creature]):
        """
        Adapt mutation parameters based on population diversity.
        This helps maintain diversity and avoid premature convergence.
        """
        diversity = self._calculate_weights_diversity(population)
        
        # If diversity is low, increase mutation
        if diversity < 1.0:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            self.mutation_strength = min(0.5, self.mutation_strength * 1.1)
        else:
            # If diversity is high, decrease mutation
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
            self.mutation_strength = max(0.1, self.mutation_strength * 0.95)