#!/usr/bin/env python3
"""Basic functionality test to verify the system works."""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from natural_selection_rl.creature import Creature
from natural_selection_rl.environment import GridEnvironment
from natural_selection_rl.evolution import GeneticAlgorithm


def test_creature_creation():
    """Test creature creation and neural network functionality."""
    print("Testing creature creation...")
    creature = Creature(vision_rays=8, hidden_size=16)
    
    # Test forward pass
    vision_input = np.random.rand(8)
    steering_angle, speed = creature.act(vision_input)
    
    print(f"  ✓ Creature created with {creature.get_weight_count()} weights")
    print(f"  ✓ Forward pass: angle={steering_angle:.3f}, speed={speed:.3f}")
    
    # Test weight manipulation
    weights = creature.get_weights()
    new_creature = Creature(vision_rays=8, hidden_size=16, weights=weights)
    
    steering_angle2, speed2 = new_creature.act(vision_input)
    assert abs(steering_angle - steering_angle2) < 1e-6
    assert abs(speed - speed2) < 1e-6
    print(f"  ✓ Weight serialization/deserialization works")
    
    return True


def test_environment():
    """Test environment functionality."""
    print("Testing environment...")
    env = GridEnvironment(width=50, height=50, num_food=10)
    
    creature = Creature()
    creature.x = 25
    creature.y = 25
    
    # Test vision
    vision = env.get_vision_input(creature)
    print(f"  ✓ Vision input shape: {vision.shape}")
    
    # Test step
    creatures = [creature]
    step_info = env.step(creatures)
    print(f"  ✓ Environment step completed: {step_info.keys()}")
    
    return True


def test_genetic_algorithm():
    """Test genetic algorithm functionality."""
    print("Testing genetic algorithm...")
    ga = GeneticAlgorithm(population_size=10)
    
    # Create population
    population = ga.create_initial_population(vision_rays=8, hidden_size=16)
    print(f"  ✓ Created population of {len(population)} creatures")
    
    # Set some fitness values
    for i, creature in enumerate(population):
        creature.fitness = np.random.uniform(0, 100)
    
    # Test evolution
    new_population = ga.evolve_population(population)
    print(f"  ✓ Evolution completed: {len(new_population)} creatures")
    
    # Test stats
    stats = ga.get_population_stats(new_population)
    print(f"  ✓ Population stats: generation={stats['generation']}, diversity={stats['weights_diversity']:.3f}")
    
    return True


def test_integration():
    """Test basic integration of all components."""
    print("Testing system integration...")
    
    # Create small system
    env = GridEnvironment(width=30, height=30, num_food=5)
    ga = GeneticAlgorithm(population_size=5)
    population = ga.create_initial_population()
    
    # Place creatures
    for i, creature in enumerate(population):
        creature.x = 10 + i * 2
        creature.y = 15
        creature.fitness = 0
    
    # Run a few steps
    for step in range(10):
        step_info = env.step(population)
        
    print(f"  ✓ Integration test completed: {len(population)} creatures simulated")
    
    # Test evolution
    new_population = ga.evolve_population(population)
    print(f"  ✓ Evolution after simulation: {len(new_population)} creatures")
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Natural Selection RL - Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_creature_creation,
        test_environment,
        test_genetic_algorithm,
        test_integration
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} PASSED\n")
            else:
                print(f"✗ {test.__name__} FAILED\n")
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}\n")
    
    print("=" * 50)
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    print("=" * 50)
    
    if passed == len(tests):
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)