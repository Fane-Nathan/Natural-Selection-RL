"""Creature class with 2-layer MLP policy for vision-to-steering mapping."""

import numpy as np
from typing import Tuple, Optional


class Creature:
    """A creature with a 2-layer MLP policy that maps vision rays to steering actions."""
    
    def __init__(self, 
                 vision_rays: int = 8,
                 hidden_size: int = 16,
                 learning_rate: float = 0.01,
                 weights: Optional[np.ndarray] = None):
        """
        Initialize creature with neural network policy.
        
        Args:
            vision_rays: Number of vision rays for input
            hidden_size: Size of hidden layer
            learning_rate: Learning rate for RL training within lifetime
            weights: Pre-initialized weights (for evolution), if None random init
        """
        self.vision_rays = vision_rays
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Network architecture: vision_rays -> hidden_size -> 2 (steering: angle, speed)
        self.input_size = vision_rays
        self.output_size = 2  # steering angle and speed
        
        # Initialize weights
        if weights is not None:
            self.set_weights(weights)
        else:
            self._initialize_random_weights()
            
        # Stats for fitness evaluation
        self.fitness = 0.0
        self.age = 0
        self.energy = 100.0
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0
        
        # RL training history
        self.experiences = []
        
    def _initialize_random_weights(self):
        """Initialize weights randomly using Xavier initialization."""
        # Input to hidden weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros(self.hidden_size)
        
        # Hidden to output weights and biases
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.output_size)
        
    def get_weights(self) -> np.ndarray:
        """Get all weights as a flattened array for evolution."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
        
    def set_weights(self, weights: np.ndarray):
        """Set weights from a flattened array."""
        idx = 0
        
        # Reshape W1
        w1_size = self.input_size * self.hidden_size
        self.W1 = weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        # Reshape b1
        self.b1 = weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        
        # Reshape W2
        w2_size = self.hidden_size * self.output_size
        self.W2 = weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        # Reshape b2
        self.b2 = weights[idx:idx + self.output_size]
        
    def get_weight_count(self) -> int:
        """Get total number of weights in the network."""
        return (self.input_size * self.hidden_size + self.hidden_size + 
                self.hidden_size * self.output_size + self.output_size)
    
    def forward(self, vision_input: np.ndarray) -> Tuple[float, float]:
        """
        Forward pass through the neural network.
        
        Args:
            vision_input: Array of vision ray distances
            
        Returns:
            Tuple of (steering_angle, speed) actions
        """
        # Ensure input is the right shape
        if len(vision_input.shape) == 1:
            vision_input = vision_input.reshape(1, -1)
            
        # Forward pass
        # Layer 1: Linear + ReLU
        z1 = np.dot(vision_input, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation
        
        # Layer 2: Linear + Tanh (for bounded output)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.tanh(z2)  # Tanh activation for bounded output
        
        # Return steering angle (-1 to 1) and speed (0 to 1)
        steering_angle = a2[0, 0]  # -1 to 1
        speed = (a2[0, 1] + 1) / 2  # Convert from -1,1 to 0,1
        
        return steering_angle, speed
    
    def act(self, vision_input: np.ndarray) -> Tuple[float, float]:
        """Get action from the policy given vision input."""
        return self.forward(vision_input)
    
    def update_position(self, steering_angle: float, speed: float, dt: float = 0.1):
        """Update creature position based on action."""
        # Update angle based on steering
        self.angle += steering_angle * dt
        
        # Update position based on speed and angle
        self.x += speed * np.cos(self.angle) * dt
        self.y += speed * np.sin(self.angle) * dt
        
        # Age the creature
        self.age += 1
        
        # Reduce energy based on movement
        self.energy -= speed * 0.1
        
    def add_experience(self, state: np.ndarray, action: Tuple[float, float], 
                      reward: float, next_state: np.ndarray):
        """Add experience for potential RL training within lifetime."""
        self.experiences.append((state, action, reward, next_state))
        
        # Keep only recent experiences to limit memory
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-1000:]
    
    def lifetime_learning(self, iterations: int = 10):
        """
        Simple gradient-based learning within lifetime.
        This represents the Baldwin effect - learning during lifetime.
        """
        if len(self.experiences) < 10:
            return
            
        # Simple policy gradient-like update
        for _ in range(iterations):
            # Sample a batch of experiences
            batch_size = min(32, len(self.experiences))
            indices = np.random.choice(len(self.experiences), batch_size, replace=False)
            
            total_grad_W1 = np.zeros_like(self.W1)
            total_grad_b1 = np.zeros_like(self.b1)
            total_grad_W2 = np.zeros_like(self.W2)
            total_grad_b2 = np.zeros_like(self.b2)
            
            for idx in indices:
                state, action, reward, _ = self.experiences[idx]
                
                # Forward pass to get current action
                current_action = self.forward(state)
                
                # Simple loss: encourage actions that led to positive rewards
                if reward > 0:
                    # Compute gradients (simplified)
                    # This is a very basic approximation of policy gradient
                    action_diff = np.array(action) - np.array(current_action)
                    
                    # Backward pass (simplified)
                    grad_output = action_diff * reward * 0.01
                    
                    # Accumulate gradients (this is highly simplified)
                    total_grad_W2 += np.outer(np.maximum(0, np.dot(state, self.W1) + self.b1), grad_output)
                    total_grad_b2 += grad_output
            
            # Update weights
            self.W2 += self.learning_rate * total_grad_W2 / batch_size
            self.b2 += self.learning_rate * total_grad_b2 / batch_size
    
    def copy(self) -> 'Creature':
        """Create a copy of this creature."""
        new_creature = Creature(
            vision_rays=self.vision_rays,
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            weights=self.get_weights()
        )
        return new_creature
    
    def reset(self):
        """Reset creature state for new episode."""
        self.fitness = 0.0
        self.age = 0
        self.energy = 100.0
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0
        self.experiences = []