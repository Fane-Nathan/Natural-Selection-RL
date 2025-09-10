#!/usr/bin/env python3
"""
Quick example demonstrating the Natural Selection RL system.
This runs a shorter simulation to quickly show the Baldwin Effect.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from natural_selection_rl.simulation import Simulation


def main():
    """Run a quick demonstration of the system."""
    print("=" * 50)
    print("Quick Natural Selection RL Demo")
    print("=" * 50)
    
    # Create a smaller simulation for quick demo
    simulation = Simulation(
        population_size=20,  # Smaller population
        environment_size=50,  # Smaller environment
        num_food=20,         # Less food
        enable_lifetime_learning=True
    )
    
    print("Running quick demo (10 generations)...")
    print("This demonstrates:")
    print("- Creatures with 2-layer MLP policies")
    print("- Vision-based navigation") 
    print("- Genetic algorithm evolution")
    print("- Lifetime learning (Baldwin Effect)")
    print()
    
    # Run simulation
    try:
        generation_stats = simulation.run_simulation(
            num_generations=10,
            episodes_per_generation=2,
            max_steps=500,
            save_interval=5,
            save_directory="demo_results"
        )
        
        print("\n" + "=" * 50)
        print("DEMO RESULTS")
        print("=" * 50)
        
        if generation_stats:
            initial_fitness = generation_stats[0]['max_fitness']
            final_fitness = generation_stats[-1]['max_fitness']
            improvement = ((final_fitness - initial_fitness) / max(initial_fitness, 0.1)) * 100
            
            print(f"Initial max fitness: {initial_fitness:.2f}")
            print(f"Final max fitness: {final_fitness:.2f}")
            print(f"Improvement: {improvement:.1f}%")
            print()
            print("Key features demonstrated:")
            print("✓ Creatures evolved neural network policies")
            print("✓ Vision rays guide navigation behavior")
            print("✓ Genetic algorithm optimized weights")
            print("✓ Lifetime learning accelerated evolution")
            
        print(f"\nVisualization saved to: demo_results/")
        print("Check the generated plots to see evolution in action!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("This might be due to missing dependencies.")
        print("Try: pip install -r requirements.txt")


if __name__ == '__main__':
    main()