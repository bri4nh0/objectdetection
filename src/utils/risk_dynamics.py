"""
Risk Dynamics Mathematics Framework
YOUR ORIGINAL RESEARCH CONTRIBUTION
Author: [Your Name] 
Date: November 2024

Novel mathematical framework for modeling threat evolution using calculus.
Key Innovations:
1. Risk Velocity - First derivative of risk evolution
2. Risk Acceleration - Second derivative of risk escalation  
3. Phase Transition Detection - Critical points in threat dynamics
4. Risk Momentum - Persistence of risk trends
"""

import numpy as np
import torch

class RiskDynamicsMathematics:
    """
    YOUR mathematical framework for temporal risk assessment
    
    This is YOUR core research contribution that transforms your work
    from tool implementation to mathematical innovation.
    """
    
    def __init__(self, diff_method='central', velocity_threshold=0.3, acceleration_threshold=0.2):
        """
        YOUR parameter decisions - document why you choose these values
        
        Args:
            diff_method: YOUR choice of numerical differentiation
                'forward' - faster computation, less accurate
                'central' - more accurate, handles noise better
            velocity_threshold: YOUR threshold for significant risk change
            acceleration_threshold: YOUR threshold for rapid escalation
        """
        # YOUR DECISIONS - these become part of your research contribution
        self.diff_method = diff_method
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        
        # Track history for analysis and validation
        self.velocity_history = []
        self.acceleration_history = []
        self.phase_transitions = []
        
        print(f"✅ YOUR Risk Dynamics Framework Initialized")
        print(f"   Differentiation: {diff_method}, Velocity Threshold: {velocity_threshold}")
    
    def compute_risk_velocity(self, risk_sequence):
        """
        YOUR implementation of risk velocity: v_risk = dr/dt
        
        Computes how quickly risk level changes over time.
        Positive = risk escalating, Negative = de-escalating.
        
        Args:
            risk_sequence: List or array of risk levels over time
            
        Returns:
            risk_velocity: Array showing rate of risk change
        """
        if len(risk_sequence) < 2:
            return np.array([0.0])
        
        # YOUR IMPLEMENTATION CHOICE - try different methods
        if self.diff_method == 'forward':
            # Forward difference: simple and fast
            velocity = risk_sequence[1:] - risk_sequence[:-1]
            # Pad to maintain original length
            velocity = np.concatenate([[velocity[0]], velocity]) if len(velocity) > 0 else np.array([0.0])
            
        elif self.diff_method == 'central':
            # Central difference: more accurate for your application
            if len(risk_sequence) < 3:
                velocity = np.array([0.0] * len(risk_sequence))
            else:
                velocity = np.zeros_like(risk_sequence)
                # Central difference for middle points
                for i in range(1, len(risk_sequence)-1):
                    velocity[i] = (risk_sequence[i+1] - risk_sequence[i-1]) / 2.0
                # Handle boundaries with forward/backward difference
                velocity[0] = risk_sequence[1] - risk_sequence[0]
                velocity[-1] = risk_sequence[-1] - risk_sequence[-2]
        else:
            raise ValueError(f"Unknown differentiation method: {self.diff_method}")
        
        # Store for YOUR analysis
        self.velocity_history.append(velocity)
        
        return velocity
    
    def compute_risk_acceleration(self, risk_sequence):
        """
        YOUR implementation of risk acceleration: a_risk = d²r/dt²
        
        Computes whether risk escalation is speeding up or slowing down.
        
        Args:
            risk_sequence: List or array of risk levels over time
            
        Returns:
            risk_acceleration: Array showing acceleration of risk change
        """
        if len(risk_sequence) < 3:
            return np.array([0.0] * len(risk_sequence))
        
        # YOUR APPROACH: Compute velocity first, then differentiate
        velocity = self.compute_risk_velocity(risk_sequence)
        acceleration = self.compute_risk_velocity(velocity)  # Reuse same method
        
        # Store for YOUR analysis
        self.acceleration_history.append(acceleration)
        
        return acceleration
    
    def detect_phase_transitions(self, risk_sequence):
        """
        YOUR novel method for detecting critical points in risk dynamics
        
        Phase transitions occur when risk dynamics change fundamentally.
        This is YOUR innovative contribution to threat assessment.
        
        Args:
            risk_sequence: List or array of risk levels over time
            
        Returns:
            transitions: List of dictionaries describing detected transitions
        """
        if len(risk_sequence) < 3:
            return []
            
        velocity = self.compute_risk_velocity(risk_sequence)
        acceleration = self.compute_risk_acceleration(risk_sequence)
        
        transitions = []
        
        # YOUR DETECTION CRITERIA - tune these based on your experiments
        for i in range(len(risk_sequence)):
            # Check if this point meets phase transition criteria
            if (abs(velocity[i]) > self.velocity_threshold and 
                abs(acceleration[i]) > self.acceleration_threshold):
                
                # YOUR CLASSIFICATION LOGIC - define what each transition means
                if velocity[i] > 0 and acceleration[i] > 0:
                    transition_type = "RAPID_ESCALATION"
                    severity = "HIGH"
                    description = "Threat escalating rapidly and accelerating"
                elif velocity[i] > 0 and acceleration[i] < 0:
                    transition_type = "DAMPENED_ESCALATION" 
                    severity = "MEDIUM"
                    description = "Threat escalating but rate is slowing"
                elif velocity[i] < 0 and acceleration[i] < 0:
                    transition_type = "RAPID_DEESCALATION"
                    severity = "MEDIUM" 
                    description = "Threat de-escalating rapidly"
                else:
                    transition_type = "DAMPENED_DEESCALATION"
                    severity = "LOW"
                    description = "Threat de-escalating gradually"
                
                transition = {
                    'frame': i,
                    'type': transition_type,
                    'severity': severity,
                    'description': description,
                    'velocity': float(velocity[i]),
                    'acceleration': float(acceleration[i]),
                    'risk_level': float(risk_sequence[i])
                }
                transitions.append(transition)
                
                # Store for YOUR analysis
                self.phase_transitions.append(transition)
        
        return transitions
    
    def calculate_risk_momentum(self, risk_sequence, window_size=5):
        """
        YOUR concept of risk momentum - integrated risk velocity
        
        Represents persistence of risk trends. High momentum means
        risk trends are likely to continue in same direction.
        
        Args:
            risk_sequence: List or array of risk levels over time
            window_size: YOUR choice of integration window
            
        Returns:
            momentum: Array showing risk momentum over time
        """
        velocity = self.compute_risk_velocity(risk_sequence)
        
        momentum = np.zeros_like(velocity)
        
        # YOUR INTEGRATION METHOD - experiment with different windows
        for i in range(len(velocity)):
            start_idx = max(0, i - window_size + 1)
            # Integrate velocity over the window (simple sum)
            momentum[i] = np.sum(velocity[start_idx:i+1])
        
        return momentum
    
    def get_dynamics_summary(self, risk_sequence):
        """
        YOUR comprehensive risk dynamics analysis
        
        Provides complete picture of how risk is evolving over time.
        This summary becomes part of your real-time threat assessment.
        """
        if len(risk_sequence) == 0:
            return {
                'current_risk': 0.0,
                'current_velocity': 0.0,
                'current_acceleration': 0.0,
                'current_momentum': 0.0,
                'recent_transitions': [],
                'avg_velocity': 0.0,
                'max_velocity': 0.0,
                'escalation_trend': 'STABLE'
            }
            
        velocity = self.compute_risk_velocity(risk_sequence)
        acceleration = self.compute_risk_acceleration(risk_sequence)
        transitions = self.detect_phase_transitions(risk_sequence)
        momentum = self.calculate_risk_momentum(risk_sequence)
        
        # YOUR SUMMARY METRICS - define what matters for threat assessment
        current_risk = risk_sequence[-1] if len(risk_sequence) > 0 else 0.0
        current_velocity = velocity[-1] if len(velocity) > 0 else 0.0
        current_acceleration = acceleration[-1] if len(acceleration) > 0 else 0.0
        current_momentum = momentum[-1] if len(momentum) > 0 else 0.0
        
        # YOUR TREND CLASSIFICATION
        if current_momentum > 0.1:
            escalation_trend = 'INCREASING'
        elif current_momentum < -0.1:
            escalation_trend = 'DECREASING'
        else:
            escalation_trend = 'STABLE'
        
        summary = {
            'current_risk': float(current_risk),
            'current_velocity': float(current_velocity),
            'current_acceleration': float(current_acceleration),
            'current_momentum': float(current_momentum),
            'recent_transitions': transitions[-3:],  # Last 3 transitions
            'avg_velocity': float(np.mean(np.abs(velocity))),
            'max_velocity': float(np.max(np.abs(velocity))),
            'escalation_trend': escalation_trend,
            'risk_sequence_length': len(risk_sequence)
        }
        
        return summary
    
    def reset_history(self):
        """
        Clear history for new sequence analysis
        """
        self.velocity_history = []
        self.acceleration_history = []
        self.phase_transitions = []

# Example usage and testing
if __name__ == "__main__":
    print("Testing YOUR Risk Dynamics Mathematics Framework...")
    
    # Create test scenario: escalating threat
    test_sequence = [0.1, 0.3, 0.6, 0.9, 1.2, 1.6, 2.1, 2.7]
    
    # Initialize YOUR framework with YOUR parameters
    analyzer = RiskDynamicsMathematics(
        diff_method='central',      # YOUR choice
        velocity_threshold=0.3,     # YOUR tuning  
        acceleration_threshold=0.2  # YOUR tuning
    )
    
    # Test YOUR implementation
    velocity = analyzer.compute_risk_velocity(test_sequence)
    acceleration = analyzer.compute_risk_acceleration(test_sequence)
    transitions = analyzer.detect_phase_transitions(test_sequence)
    momentum = analyzer.calculate_risk_momentum(test_sequence)
    summary = analyzer.get_dynamics_summary(test_sequence)
    
    print(f"Test Sequence: {test_sequence}")
    print(f"Risk Velocity: {[f'{v:.2f}' for v in velocity]}")
    print(f"Risk Acceleration: {[f'{a:.2f}' for a in acceleration]}")
    print(f"Phase Transitions: {len(transitions)} detected")
    for trans in transitions:
        print(f"  - Frame {trans['frame']}: {trans['type']} (v={trans['velocity']:.2f}, a={trans['acceleration']:.2f})")
    print(f"Risk Momentum: {[f'{m:.2f}' for m in momentum]}")
    print(f"Summary: {summary['escalation_trend']} trend (momentum: {summary['current_momentum']:.2f})")
    
    print("\n✅ YOUR Risk Dynamics Mathematics framework is working!")