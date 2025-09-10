#!/usr/bin/env python3
"""Quick test of the Baldwin Effect comparison."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from natural_selection_rl.simulation import Simulation


def main():
    """Run a quick Baldwin Effect comparison."""
    print("=" * 60)
    print("Baldwin Effect Demonstration")
    print("=" * 60)
    
    # Create simulation with smaller parameters for speed
    simulation = Simulation(
        population_size=15,
        environment_size=40,
        num_food=15,
        enable_lifetime_learning=True
    )
    
    print("This will compare evolution with and without lifetime learning")
    print("to demonstrate the Baldwin Effect...")
    print()
    
    # Run comparison with fewer generations for speed
    try:
        results = simulation.compare_with_without_learning(num_generations=8)
        
        print("\nBaldwin Effect demonstration completed!")
        print("The plots show how lifetime learning accelerates evolution.")
        
    except Exception as e:
        print(f"Error during Baldwin Effect test: {e}")
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    print("\n" + "=" * 60)
    if success:
        print("✅ Baldwin Effect demonstration successful!")
    else:
        print("❌ Baldwin Effect demonstration failed!")
    print("=" * 60)