import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics.pairwise import euclidean_distances
import os

class HealthIndicatorBuilder:
    """
    Build complete health indicator trajectories from pipeline data
    """
    
    def __init__(self, cfar_detector):
        self.detector = cfar_detector
        
    def build_trajectory(self, file_path, start_percent=0, end_percent=100, time_resolution=60):
        """
        Build health indicator trajectory from pipeline data
        
        Parameters:
        - file_path: path to .wfs file
        - start_percent, end_percent: portion of file to process (0-100)
        - time_resolution: seconds per HI point
        """
        print(f"Building HI trajectory from {start_percent}% to {end_percent}%")
        
        # Calculate file parameters
        file_size = os.path.getsize(file_path)
        total_samples = file_size // 2
        
        start_sample = int(start_percent / 100 * total_samples)
        end_sample = int(end_percent / 100 * total_samples)
        
        samples_per_window = time_resolution * 1000000  # samples per time window
        num_windows = (end_sample - start_sample) // samples_per_window
        
        print(f"Processing {num_windows} windows of {time_resolution}s each")
        
        # Initialize trajectory
        time_points = []
        hi_values = []
        cumulative_hits = 0
        
        for i in range(num_windows):
            window_start = start_sample + i * samples_per_window
            current_time = i * time_resolution
            
            # Load window data
            with open(file_path, 'rb') as f:
                f.seek(window_start * 2)
                data_bytes = f.read(samples_per_window * 2)
                window_data = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32)
            
            # Detect AE hits in this window
            results = self.detector.detect_ae_hits_adaptive(window_data)
            window_hits = len(results['hit_indices'])
            
            # Update cumulative hits
            cumulative_hits += window_hits
            
            time_points.append(current_time)
            hi_values.append(cumulative_hits)
            
            if i % 10 == 0:  # Progress update
                print(f"  Window {i}/{num_windows}: {cumulative_hits} total hits")
        
        return {
            'time': np.array(time_points),
            'hi': np.array(hi_values),
            'total_time': num_windows * time_resolution,
            'total_hits': cumulative_hits
        }

class DerivativeConvolutionalDomain:
    """
    Implement the derivative convolutional domain from the paper
    """
    
    def __init__(self, n_points=1000):
        self.n_points = n_points
        
    def sawtooth_kernel(self, t):
        """
        Generate sawtooth function as in equation (9)
        """
        n = self.n_points
        if 0 <= t < n:
            return 1 - 2*t/(n-1)
        else:
            return 0
    
    def transform(self, hi_signal):
        """
        Apply derivative convolution transform (equation 8)
        """
        # Create sawtooth kernel
        kernel = np.array([self.sawtooth_kernel(t) for t in range(self.n_points)])
        
        # Apply convolution
        dc_signal = np.convolve(hi_signal, kernel, mode='same')
        
        return dc_signal

class SimilarityBasedRULPredictor:
    """
    Main RUL prediction class implementing the paper's methodology
    """
    
    def __init__(self):
        self.dc_transform = DerivativeConvolutionalDomain()
        self.reference_trajectories = []
        
    def add_reference_trajectory(self, trajectory, pipeline_name, total_lifetime):
        """
        Add a reference (historical) trajectory
        """
        # Transform to derivative convolutional domain
        dc_trajectory = self.dc_transform.transform(trajectory['hi'])
        
        # Create segments as in the paper (with step of 100,000 points -> adapted to our resolution)
        segment_step = max(1, len(dc_trajectory) // 20)  # Create ~20 segments
        segments = []
        
        for i in range(0, len(dc_trajectory) - segment_step, segment_step):
            segment = dc_trajectory[i:i+segment_step]
            remaining_time = total_lifetime - trajectory['time'][i] if i < len(trajectory['time']) else 0
            
            segments.append({
                'segment': segment,
                'rul': max(0, remaining_time),
                'start_time': trajectory['time'][i] if i < len(trajectory['time']) else total_lifetime
            })
        
        self.reference_trajectories.append({
            'name': pipeline_name,
            'trajectory': trajectory,
            'dc_trajectory': dc_trajectory,
            'segments': segments,
            'total_lifetime': total_lifetime
        })
        
        print(f"Added reference trajectory: {pipeline_name}")
        print(f"  Total lifetime: {total_lifetime}s")
        print(f"  Created {len(segments)} reference segments")
    
    def predict_rul(self, test_trajectory):
        """
        Predict RUL for a test trajectory
        """
        # Transform test trajectory to derivative convolutional domain
        test_dc = self.dc_transform.transform(test_trajectory['hi'])
        
        if len(self.reference_trajectories) == 0:
            raise ValueError("No reference trajectories available")
        
        # Collect all reference segments
        all_segments = []
        all_ruls = []
        
        for ref_traj in self.reference_trajectories:
            for segment in ref_traj['segments']:
                all_segments.append(segment['segment'])
                all_ruls.append(segment['rul'])
        
        print(f"Comparing against {len(all_segments)} reference segments")
        
        # Calculate similarities (Euclidean distances in DC domain)
        test_length = len(test_dc)
        distances = []
        valid_ruls = []
        
        for i, ref_segment in enumerate(all_segments):
            if len(ref_segment) >= test_length:
                # Use first part of reference segment
                ref_portion = ref_segment[:test_length]
                distance = np.linalg.norm(test_dc - ref_portion)
                distances.append(distance)
                valid_ruls.append(all_ruls[i])
        
        if len(distances) == 0:
            print("Warning: No compatible reference segments found")
            return None
        
        distances = np.array(distances)
        valid_ruls = np.array(valid_ruls)
        
        # Apply adaptive fuzzy weighting (equation 10-12)
        weights = self._calculate_fuzzy_weights(distances)
        
        # Calculate weighted RUL prediction
        predicted_rul = np.sum(weights * valid_ruls) / np.sum(weights)
        
        return {
            'predicted_rul': predicted_rul,
            'distances': distances,
            'weights': weights,
            'reference_ruls': valid_ruls,
            'num_references': len(distances)
        }
    
    def _calculate_fuzzy_weights(self, distances):
        """
        Calculate adaptive fuzzy weights (equations 10-11)
        """
        # Calculate adaptive sigma
        sigma = np.std(distances)
        if sigma == 0:
            sigma = 1.0  # Avoid division by zero
        
        # Gaussian membership function (equation 10)
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        return weights
    
    def plot_similarity_analysis(self, test_trajectory, prediction_results):
        """
        Visualize the similarity analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Test trajectory vs references
        test_dc = self.dc_transform.transform(test_trajectory['hi'])
        
        ax1.plot(test_trajectory['time'], test_trajectory['hi'], 'r-', linewidth=2, label='Test Trajectory')
        for ref in self.reference_trajectories:
            ax1.plot(ref['trajectory']['time'], ref['trajectory']['hi'], '--', alpha=0.7, label=f"Ref: {ref['name']}")
        ax1.set_title('Health Indicator Trajectories')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cumulative AE Hits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Derivative Convolutional Domain
        ax2.plot(test_dc, 'r-', linewidth=2, label='Test (DC Domain)')
        for ref in self.reference_trajectories:
            ax2.plot(ref['dc_trajectory'], '--', alpha=0.7, label=f"Ref: {ref['name']} (DC)")
        ax2.set_title('Derivative Convolutional Domain')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('DC Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distance vs RUL
        if prediction_results:
            distances = prediction_results['distances']
            reference_ruls = prediction_results['reference_ruls']
            weights = prediction_results['weights']
            
            scatter = ax3.scatter(distances, reference_ruls, c=weights, cmap='viridis', s=50, alpha=0.7)
            ax3.set_title('Distance vs Reference RUL (colored by weight)')
            ax3.set_xlabel('Euclidean Distance')
            ax3.set_ylabel('Reference RUL (s)')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Fuzzy Weight')
        
        # Plot 4: Weight distribution
        if prediction_results:
            ax4.hist(weights, bins=20, alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(weights), color='red', linestyle='--', label=f'Mean: {np.mean(weights):.3f}')
            ax4.set_title('Fuzzy Weight Distribution')
            ax4.set_xlabel('Weight Value')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def demonstrate_rul_prediction():
    """
    Demonstrate the complete RUL prediction pipeline
    """
    print("=== RUL Prediction Demonstration ===")
    
    from test_segments import AdaptiveCFARDetector  # Import our CFAR detector
    
    # Initialize components
    detector = AdaptiveCFARDetector()
    hi_builder = HealthIndicatorBuilder(detector)
    rul_predictor = SimilarityBasedRULPredictor()
    
    data_path = r"D:\Pipeline RUL Data\B.wfs"
    
    # Build reference trajectory (first 80% of pipeline life)
    print("\n1. Building reference trajectory...")
    ref_trajectory = hi_builder.build_trajectory(data_path, 0, 80, time_resolution=300)  # 5-minute intervals
    total_lifetime = ref_trajectory['total_time'] * 1.25  # Assume 80% represents 80% of life
    
    # Add reference to predictor
    rul_predictor.add_reference_trajectory(ref_trajectory, "Pipeline B", total_lifetime)
    
    # Build test trajectory (first 60% - simulating current state)
    print("\n2. Building test trajectory...")
    test_trajectory = hi_builder.build_trajectory(data_path, 0, 60, time_resolution=300)
    
    # Predict RUL
    print("\n3. Predicting RUL...")
    prediction = rul_predictor.predict_rul(test_trajectory)
    
    if prediction:
        print(f"\nPREDICTION RESULTS:")
        print(f"  Predicted RUL: {prediction['predicted_rul']:.1f} seconds ({prediction['predicted_rul']/3600:.2f} hours)")
        print(f"  Used {prediction['num_references']} reference segments")
        print(f"  Distance range: {np.min(prediction['distances']):.2e} to {np.max(prediction['distances']):.2e}")
        print(f"  Weight range: {np.min(prediction['weights']):.3f} to {np.max(prediction['weights']):.3f}")
        
        # Calculate actual remaining time for comparison
        actual_remaining = total_lifetime - test_trajectory['total_time']
        error = abs(prediction['predicted_rul'] - actual_remaining)
        error_percent = error / actual_remaining * 100 if actual_remaining > 0 else 0
        
        print(f"\nACCURACY ASSESSMENT:")
        print(f"  Actual remaining time: {actual_remaining:.1f} seconds ({actual_remaining/3600:.2f} hours)")
        print(f"  Prediction error: {error:.1f} seconds ({error_percent:.1f}%)")
        
        # Visualize results
        rul_predictor.plot_similarity_analysis(test_trajectory, prediction)
    
    return rul_predictor, test_trajectory, prediction

if __name__ == "__main__":
    predictor, test_traj, pred_results = demonstrate_rul_prediction()
    print("\nRUL prediction demonstration completed!")