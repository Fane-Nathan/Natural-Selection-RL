#!/usr/bin/env python3
"""
Main script demonstrating Natural Selection with Reinforcement Learning.
This showcases the Baldwin Effect - how learning within lifetime can accelerate evolution.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from natural_selection_rl.simulation import Simulation


def main():
    """Main function to run the natural selection RL simulation."""
    parser = argparse.ArgumentParser(
        description='Natural Selection RL - Baldwin Effect Simulation'
    )
    
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations to simulate (default: 50)')
    parser.add_argument('--population', type=int, default=50,
                       help='Population size (default: 50)')
    parser.add_argument('--environment-size', type=int, default=100,
                       help='Environment grid size (default: 100)')
    parser.add_argument('--no-learning', action='store_true',
                       help='Disable lifetime learning (default: enabled)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare evolution with and without learning')
    parser.add_argument('--output-dir', type=str, default='simulation_results',
                       help='Output directory for results (default: simulation_results)')
    parser.add_argument('--episodes-per-gen', type=int, default=3,
                       help='Episodes per generation (default: 3)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Natural Selection with Reinforcement Learning")
    print("Demonstrating the Baldwin Effect")
    print("=" * 60)
    
    # Create simulation
    simulation = Simulation(
        population_size=args.population,
        environment_size=args.environment_size,
        enable_lifetime_learning=not args.no_learning
    )
    
    if args.compare:
        # Run Baldwin Effect comparison
        print("\nRunning Baldwin Effect comparison...")
        comparison_results = simulation.compare_with_without_learning(
            num_generations=min(30, args.generations)
        )
        
        print("\nComparison completed!")
        print("Check the generated plots to see the Baldwin Effect in action.")
        
    else:
        # Run main simulation
        print(f"\nRunning simulation for {args.generations} generations...")
        print(f"Population size: {args.population}")
        print(f"Environment size: {args.environment_size}x{args.environment_size}")
        print(f"Lifetime learning: {'Enabled' if not args.no_learning else 'Disabled'}")
        print(f"Output directory: {args.output_dir}")
        
        # Run the simulation
        generation_stats = simulation.run_simulation(
            num_generations=args.generations,
            episodes_per_generation=args.episodes_per_gen,
            max_steps=args.max_steps,
            save_directory=args.output_dir
        )
        
        # Print final results
        if generation_stats:
            final_stats = generation_stats[-1]
            print("\n" + "=" * 60)
            print("FINAL RESULTS")
            print("=" * 60)
            print(f"Final generation: {final_stats['generation']}")
            print(f"Best fitness achieved: {final_stats['max_fitness']:.2f}")
            print(f"Average fitness: {final_stats['mean_fitness']:.2f}")
            print(f"Population diversity: {final_stats['weights_diversity']:.2f}")
            
            # Calculate improvement over generations
            if len(generation_stats) > 1:
                initial_fitness = generation_stats[0]['max_fitness']
                final_fitness = final_stats['max_fitness']
                improvement = ((final_fitness - initial_fitness) / initial_fitness) * 100
                print(f"Fitness improvement: {improvement:.1f}%")
        
        print(f"\nResults and visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred: {e}")
        sys.exit(1)