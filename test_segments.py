import numpy as np
import matplotlib.pyplot as plt
import os

def load_pipeline_data_segment(file_path, start_percent=0, duration_seconds=10):
    """
    Load data from a specific time segment of the pipeline file
    start_percent: 0-100, where to start in the file (0=beginning, 90=near end)
    """
    # Get file size to calculate position
    file_size = os.path.getsize(file_path)
    total_samples = file_size // 2  # 2 bytes per int16 sample
    
    # Calculate start position
    start_sample = int(start_percent / 100 * total_samples)
    num_samples = int(duration_seconds * 1000000)  # 10 seconds at 1MHz
    
    with open(file_path, 'rb') as f:
        f.seek(start_sample * 2)  # Seek to start position (2 bytes per sample)
        data_bytes = f.read(num_samples * 2)
        data = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32)
    
    print(f"Loaded {len(data):,} samples from {start_percent}% position")
    print(f"File position: sample {start_sample:,} to {start_sample + len(data):,}")
    return data

class AdaptiveCFARDetector:
    """
    CFAR Detector with adaptive threshold capability
    """
    
    def __init__(self, 
                 false_alarm_prob=1e-4,  # Relaxed from 1e-7
                 cell_size=500,
                 guard_cells=10,
                 training_cells=20,
                 sampling_rate=1000000):
        
        self.Pfa = false_alarm_prob
        self.cell_size = cell_size
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.sampling_rate = sampling_rate
        
        # Calculate threshold factor
        N = training_cells
        self.alpha = N * (self.Pfa**(-1/N) - 1)
        
        print(f"Adaptive CFAR Detector:")
        print(f"  False alarm prob: {self.Pfa} (relaxed)")
        print(f"  Threshold factor (α): {self.alpha:.2f}")
    
    def calculate_energy(self, signal_data):
        """Calculate energy with preprocessing"""
        # Remove DC and apply high-pass filter to emphasize transients
        signal_data = signal_data - np.mean(signal_data)
        
        # Simple high-pass filter (emphasizes AE transients)
        from scipy import signal as sp_signal
        b, a = sp_signal.butter(4, 0.01, 'high')  # High-pass at 10kHz
        signal_data = sp_signal.filtfilt(b, a, signal_data)
        
        # Calculate energy in cells
        num_cells = len(signal_data) // self.cell_size
        energy = np.zeros(num_cells)
        
        for i in range(num_cells):
            start_idx = i * self.cell_size
            end_idx = start_idx + self.cell_size
            cell_data = signal_data[start_idx:end_idx]
            
            # Use RMS energy instead of sum of squares (more stable)
            energy[i] = np.sqrt(np.mean(cell_data**2))
        
        return energy
    
    def detect_ae_hits_adaptive(self, signal_data):
        """CFAR detection with adaptive threshold"""
        energy = self.calculate_energy(signal_data)
        
        detections = []
        detection_indices = []
        thresholds = []
        
        for i in range(self.training_cells + self.guard_cells, 
                       len(energy) - self.training_cells - self.guard_cells):
            
            cut_energy = energy[i]
            
            # Get training cells
            left_start = i - self.guard_cells - self.training_cells
            left_end = i - self.guard_cells
            right_start = i + self.guard_cells + 1
            right_end = i + self.guard_cells + self.training_cells + 1
            
            left_training = energy[left_start:left_end]
            right_training = energy[right_start:right_end]
            training_energies = np.concatenate([left_training, right_training])
            
            # Robust noise estimation (use median instead of mean)
            Pn = np.median(training_energies)
            
            # Adaptive threshold
            threshold = self.alpha * Pn
            thresholds.append(threshold)
            
            if cut_energy > threshold:
                detections.append(cut_energy)
                detection_indices.append(i)
        
        print(f"Adaptive CFAR: {len(detections)} hits from {len(thresholds)} cells")
        if len(detections) > 0:
            print(f"  Energy range: {np.min(detections):.1e} to {np.max(detections):.1e}")
            print(f"  Threshold range: {np.min(thresholds):.1e} to {np.max(thresholds):.1e}")
        
        return {
            'hit_indices': detection_indices,
            'hit_energies': detections,
            'energy_signal': energy,
            'thresholds': np.array(thresholds),
            'processed_indices': list(range(self.training_cells + self.guard_cells, 
                                          len(energy) - self.training_cells - self.guard_cells))
        }

def test_multiple_segments():
    """Test CFAR on multiple segments of the pipeline data"""
    data_path = r"D:\Pipeline RUL Data\B.wfs"
    
    # Test different segments: beginning, middle, and end
    test_segments = [
        (0, "Beginning (0%)"),
        (30, "Early-Middle (30%)"),
        (60, "Late-Middle (60%)"),
        (85, "Near End (85%)")
    ]
    
    detector = AdaptiveCFARDetector()
    results_summary = []
    
    for segment_percent, segment_name in test_segments:
        print(f"\n{'='*50}")
        print(f"Testing {segment_name}")
        print(f"{'='*50}")
        
        try:
            # Load data segment
            data = load_pipeline_data_segment(data_path, segment_percent, duration_seconds=5)
            
            print(f"Data stats: mean={np.mean(data):.1f}, std={np.std(data):.1f}")
            print(f"Data range: [{np.min(data):.0f}, {np.max(data):.0f}]")
            
            # Run CFAR detection
            results = detector.detect_ae_hits_adaptive(data)
            
            # Calculate metrics
            hit_rate = len(results['hit_indices']) / len(results['processed_indices']) * 100
            avg_energy = np.mean(results['energy_signal'])
            
            results_summary.append({
                'segment': segment_name,
                'percent': segment_percent,
                'hits': len(results['hit_indices']),
                'hit_rate': hit_rate,
                'avg_energy': avg_energy,
                'data_std': np.std(data)
            })
            
            # Plot if hits detected
            if len(results['hit_indices']) > 0:
                plot_segment_results(results, data, segment_name)
        
        except Exception as e:
            print(f"Error processing {segment_name}: {e}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("PIPELINE DEGRADATION ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Segment':<20} {'Position':<10} {'Hits':<8} {'Hit Rate %':<12} {'Avg Energy':<15} {'Data Std':<10}")
    print("-" * 70)
    
    for result in results_summary:
        print(f"{result['segment']:<20} {result['percent']:>3}% {result['hits']:>8} "
              f"{result['hit_rate']:>8.2f}%  {result['avg_energy']:>12.2e}  {result['data_std']:>8.1f}")
    
    return results_summary

def plot_segment_results(results, data, segment_name):
    """Plot results for a specific segment"""
    energy = results['energy_signal']
    hit_indices = results['hit_indices']
    hit_energies = results['hit_energies']
    thresholds = results['thresholds']
    processed_indices = results['processed_indices']
    
    # Time vectors
    time_signal = np.arange(len(data)) / 1000000
    time_energy = np.arange(len(energy)) * 500 / 1000000
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Original signal (first 1 second)
    plot_samples = min(1000000, len(data))
    ax1.plot(time_signal[:plot_samples], data[:plot_samples], 'b-', alpha=0.7, linewidth=0.5)
    ax1.set_title(f'Acoustic Emission Signal - {segment_name}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    
    # CFAR results
    ax2.semilogy(time_energy, energy, 'b-', alpha=0.7, label='Energy Signal', linewidth=1)
    
    threshold_times = time_energy[processed_indices]
    ax2.semilogy(threshold_times, thresholds, 'r--', alpha=0.8, label='CFAR Threshold', linewidth=1)
    
    if hit_indices:
        detection_times = time_energy[hit_indices]
        ax2.semilogy(detection_times, hit_energies, 'ro', markersize=4, 
                    label=f'AE Hits ({len(hit_indices)})', alpha=0.8)
    
    ax2.set_title(f'CFAR Detection Results - {segment_name}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def construct_health_indicator_from_summary(results_summary):
    """Construct overall pipeline health progression"""
    positions = [r['percent'] for r in results_summary]
    hit_counts = [r['hits'] for r in results_summary]
    
    plt.figure(figsize=(12, 8))
    
    # Plot hit progression
    plt.subplot(2, 1, 1)
    plt.plot(positions, hit_counts, 'go-', linewidth=2, markersize=8)
    plt.title('Pipeline Health Degradation - AE Hit Count vs Time')
    plt.xlabel('Pipeline Lifetime (%)')
    plt.ylabel('AE Hits (per 5 seconds)')
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative hits (simulated health indicator)
    plt.subplot(2, 1, 2)
    cumulative_hits = np.cumsum(hit_counts)
    plt.plot(positions, cumulative_hits, 'r.-', linewidth=2, markersize=8)
    plt.title('Cumulative Health Indicator')
    plt.xlabel('Pipeline Lifetime (%)')
    plt.ylabel('Cumulative AE Hits')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return positions, cumulative_hits

if __name__ == "__main__":
    print("Testing CFAR on multiple pipeline segments...")
    summary = test_multiple_segments()
    
    print("\nConstructing health indicator progression...")
    positions, cumulative_hits = construct_health_indicator_from_summary(summary)
    
    print(f"\nAnalysis complete!")
    print(f"Pipeline degradation pattern:")
    print(f"  Early stage (0-30%): {sum(r['hits'] for r in summary[:2])} total hits")
    print(f"  Late stage (60-85%): {sum(r['hits'] for r in summary[2:])} total hits")