# Natural Selection with Reinforcement Learning

A simulation demonstrating the **Baldwin Effect** - how learning within an individual's lifetime can accelerate evolutionary adaptation. This project implements creatures with tiny neural network policies (2-layer MLPs) that evolve their weights through genetic algorithms while optionally learning within their lifetimes through reinforcement learning.

## Features

🧬 **Evolution + Learning**: Combines genetic algorithms with reinforcement learning  
🤖 **Neural Creatures**: Each creature has a 2-layer MLP mapping vision rays to steering  
👁️ **Vision System**: Creatures use ray-casting to detect food and navigate  
🔄 **Baldwin Effect**: Demonstrates how lifetime learning guides evolution  
📊 **Visualization**: Real-time plots and environment visualization  
⚡ **Fast Simulation**: Custom grid environment optimized for multi-agent scenarios  

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Fane-Nathan/Natural-Selection-RL.git
cd Natural-Selection-RL

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Quick Demo

Run a quick demonstration:

```bash
python example.py
```

This will run a 10-generation simulation showing the core concepts.

### Full Simulation

Run a complete simulation:

```bash
# Basic simulation with default parameters
python main.py

# Custom simulation
python main.py --generations 100 --population 100 --environment-size 150

# Compare evolution with and without lifetime learning (Baldwin Effect)
python main.py --compare

# Disable lifetime learning to see pure evolution
python main.py --no-learning --generations 50
```

## System Architecture

### Creature Class (`creature.py`)
- **2-layer MLP**: Maps vision input to steering actions
- **Vision System**: 8 rays detecting food and obstacles  
- **Neural Network**: Input → Hidden (ReLU) → Output (Tanh)
- **Lifetime Learning**: Simple gradient-based policy updates

### Environment (`environment.py`)
- **Grid World**: 2D continuous space with food items
- **Vision Rays**: Ray-casting for creature perception
- **Multi-agent**: Supports multiple creatures simultaneously
- **Food Mechanics**: Energy system with food consumption

### Evolution (`evolution.py`)
- **Genetic Algorithm**: Tournament selection, crossover, mutation
- **Weight Evolution**: Direct evolution of neural network parameters
- **Adaptive Mutation**: Dynamic mutation rates based on diversity
- **Elitism**: Preserves best individuals across generations

### Simulation (`simulation.py`)
- **Baldwin Effect**: Orchestrates evolution + learning
- **Statistics Tracking**: Fitness, diversity, and population metrics
- **Visualization**: Environment rendering and evolution plots
- **Comparison Tools**: With/without learning experiments

## The Baldwin Effect

The Baldwin Effect is a key concept in evolutionary biology where learning ability influences the direction of evolution. In this simulation:

1. **Lifetime Learning**: Creatures can improve their policies during their lifetime using simple RL
2. **Genetic Evolution**: Population evolves through genetic algorithms over generations  
3. **Interaction**: Learning helps creatures survive better, creating selection pressure for learnable traits
4. **Acceleration**: Evolution proceeds faster when lifetime learning is enabled

### Experimental Results

Run the comparison to see the Baldwin Effect:

```bash
python main.py --compare
```

This will show that populations with lifetime learning achieve higher fitness faster than those relying on evolution alone.

## Configuration Options

### Command Line Arguments

- `--generations`: Number of generations (default: 50)
- `--population`: Population size (default: 50)  
- `--environment-size`: Grid size (default: 100)
- `--no-learning`: Disable lifetime learning
- `--compare`: Run Baldwin Effect comparison
- `--episodes-per-gen`: Episodes per generation (default: 3)
- `--max-steps`: Steps per episode (default: 1000)

### Key Parameters

**Neural Network:**
- Vision rays: 8 (configurable)
- Hidden layer: 16 neurons (configurable)
- Output: 2 (steering angle, speed)

**Evolution:**
- Selection: Tournament selection
- Crossover: Uniform crossover (70% rate)
- Mutation: Gaussian noise (10% rate, adaptive)
- Elitism: 10% of population preserved

**Environment:**
- Continuous 2D grid
- Food respawning mechanics
- Energy-based survival
- Boundary wrapping

## Extending the System

### Adding New Selection Methods
Extend `GeneticAlgorithm` class with new selection strategies:

```python
def roulette_wheel_selection(self, population):
    # Implement roulette wheel selection
    pass
```

### Custom Environments
Create new environments by inheriting from a base class:

```python
class MazeEnvironment(GridEnvironment):
    def __init__(self, maze_layout):
        # Custom environment implementation
        pass
```

### Advanced Learning Algorithms
Replace simple gradient updates with more sophisticated RL:

```python
def policy_gradient_learning(self):
    # Implement PPO, A2C, or other RL algorithms
    pass
```

## Research Applications

This codebase is suitable for research in:

- **Evolutionary Algorithms**: Testing new evolution strategies
- **Neuroevolution**: Evolving neural network topologies (extend to NEAT)
- **Baldwin Effect Studies**: Quantifying learning-evolution interactions  
- **Multi-agent Systems**: Studying emergent behaviors
- **Artificial Life**: Investigating adaptive behaviors

## Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Visualization and plotting
- `torch`: Neural network operations (optional, for advanced features)
- `gymnasium`: RL environment compatibility (future extensions)

## Contributing

Contributions are welcome! Areas for improvement:

- **NEAT Implementation**: Replace simple GA with NEAT
- **Advanced RL**: Implement PPO, SAC, or other modern RL algorithms
- **PettingZoo Integration**: Add multi-agent RL environment support
- **Performance Optimization**: GPU acceleration, parallel simulation
- **Visualization**: Interactive web-based visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Baldwin, J. M. (1896). "A new factor in evolution"
2. Hinton, G. E., & Nowlan, S. J. (1987). "How learning can guide evolution"
3. Stanley, K. O., & Miikkulainen, R. (2002). "Evolving neural networks through augmenting topologies"

## Citation

If you use this code in research, please cite:

```bibtex
@software{natural_selection_rl,
  title={Natural Selection with Reinforcement Learning},
  author={Natural Selection RL Contributors},
  year={2024},
  url={https://github.com/Fane-Nathan/Natural-Selection-RL}
}
```