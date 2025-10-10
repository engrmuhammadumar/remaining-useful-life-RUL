"""
Express-8 AE System WFS File Header Analyzer
Helps identify header structure and data start position
"""

import struct
import numpy as np
from pathlib import Path


def analyze_express8_header(filepath, bytes_to_read=4096):
    """
    Analyze Express-8 WFS file header
    """
    print("="*80)
    print("EXPRESS-8 WFS FILE HEADER ANALYSIS")
    print("="*80)
    
    with open(filepath, 'rb') as f:
        header_bytes = f.read(bytes_to_read)
    
    # 1. Try to find ASCII strings in header
    print("\n[1] ASCII STRINGS IN HEADER:")
    print("-"*80)
    
    ascii_strings = []
    current_string = []
    
    for i, byte in enumerate(header_bytes):
        if 32 <= byte <= 126 or byte in [10, 13]:  # Printable ASCII or newline
            current_string.append(chr(byte))
        else:
            if len(current_string) > 3:  # Only keep strings longer than 3 chars
                ascii_strings.append(''.join(current_string))
            current_string = []
    
    for s in ascii_strings[:20]:  # Show first 20 strings
        print(f"  '{s}'")
    
    # 2. Look for common patterns
    print("\n[2] SEARCHING FOR COMMON PATTERNS:")
    print("-"*80)
    
    # Look for common keywords
    keywords = [b'VERSION', b'SAMPLE', b'RATE', b'CHANNEL', b'DATA', b'START',
                b'HEADER', b'SIZE', b'TYPE', b'FORMAT', b'SENSOR', b'AE']
    
    for keyword in keywords:
        idx = header_bytes.find(keyword)
        if idx != -1:
            print(f"  Found '{keyword.decode()}' at byte position: {idx}")
            # Show context around keyword
            context_start = max(0, idx - 20)
            context_end = min(len(header_bytes), idx + 50)
            context = header_bytes[context_start:context_end]
            try:
                print(f"    Context: {context}")
            except:
                print(f"    Context (hex): {context.hex()}")
    
    # 3. Try to find where data starts (look for repeated patterns)
    print("\n[3] DETECTING DATA START POSITION:")
    print("-"*80)
    
    # Method 1: Look for transition from varied bytes to uniform/structured data
    # Calculate byte value variance in sliding windows
    window_size = 100
    variances = []
    
    for i in range(0, len(header_bytes) - window_size, 10):
        window = header_bytes[i:i+window_size]
        variance = np.var(list(window))
        variances.append((i, variance))
    
    # Find first position where variance stabilizes
    if variances:
        avg_variance = np.mean([v[1] for v in variances])
        print(f"  Average byte variance in file: {avg_variance:.2f}")
        
        # Look for first sustained high variance (likely data region)
        for i, (pos, var) in enumerate(variances[10:], 10):  # Skip first few
            if var > avg_variance * 1.5:
                print(f"  Potential data start around byte: {pos}")
                break
    
    # 4. Common Express-8 header sizes to try
    print("\n[4] COMMON HEADER SIZES TO TEST:")
    print("-"*80)
    common_sizes = [512, 1024, 2048, 4096, 8192]
    
    for size in common_sizes:
        if size < len(header_bytes):
            # Try interpreting data after this offset as int16
            test_data = struct.unpack(f'<10h', header_bytes[size:size+20])
            print(f"  Offset {size:5d} bytes -> First 10 int16 values: {test_data}")
            
            # Check if values look reasonable for AE data
            if all(-32768 < x < 32768 for x in test_data):
                # Calculate some statistics
                mean_val = np.mean(np.abs(test_data))
                if 10 < mean_val < 10000:  # Reasonable range for AE signals
                    print(f"    ✓ VALUES LOOK REASONABLE (mean abs: {mean_val:.1f})")
    
    # 5. Hexdump first 512 bytes
    print("\n[5] HEXDUMP OF FIRST 512 BYTES:")
    print("-"*80)
    
    for i in range(0, min(512, len(header_bytes)), 16):
        hex_str = ' '.join(f'{b:02x}' for b in header_bytes[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in header_bytes[i:i+16])
        print(f"  {i:04x}:  {hex_str:<48s}  {ascii_str}")
    
    # 6. Byte distribution analysis
    print("\n[6] BYTE VALUE DISTRIBUTION ANALYSIS:")
    print("-"*80)
    
    # First 1024 bytes (likely header)
    header_part = header_bytes[:1024]
    unique_bytes_header = len(set(header_part))
    
    # Next 1024 bytes (likely data)
    if len(header_bytes) >= 2048:
        data_part = header_bytes[1024:2048]
        unique_bytes_data = len(set(data_part))
        
        print(f"  Unique byte values in first 1024 bytes: {unique_bytes_header}")
        print(f"  Unique byte values in bytes 1024-2048: {unique_bytes_data}")
        print(f"  Ratio: {unique_bytes_data / max(unique_bytes_header, 1):.2f}")
        
        if unique_bytes_data > unique_bytes_header * 1.5:
            print("  → Data region likely has more diverse byte values")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("\n1. Look for 'START' or 'DATA' markers in the ASCII strings above")
    print("2. Try the header sizes that showed reasonable int16 values (marked with ✓)")
    print("3. Express-8 systems often use 1024 or 2048 byte headers")
    print("4. The data type is likely int16 (2 bytes per sample)")
    print("\n" + "="*80)


def test_data_interpretation(filepath, offset, num_samples=10000, dtype='int16'):
    """
    Test reading and visualizing data from a specific offset
    """
    print("\n" + "="*80)
    print(f"TESTING DATA READING: offset={offset}, dtype={dtype}")
    print("="*80)
    
    import matplotlib.pyplot as plt
    
    # Read data
    with open(filepath, 'rb') as f:
        f.seek(offset)
        if dtype == 'int16':
            bytes_to_read = num_samples * 2
            raw_bytes = f.read(bytes_to_read)
            data = np.frombuffer(raw_bytes, dtype=np.int16)
        elif dtype == 'float32':
            bytes_to_read = num_samples * 4
            raw_bytes = f.read(bytes_to_read)
            data = np.frombuffer(raw_bytes, dtype=np.float32)
        else:
            print(f"Unsupported dtype: {dtype}")
            return
    
    # Statistics
    print(f"\n[STATISTICS]")
    print(f"  Samples read: {len(data)}")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Std: {np.std(data):.4f}")
    print(f"  Min: {np.min(data):.4f}")
    print(f"  Max: {np.max(data):.4f}")
    print(f"  First 20 values: {data[:20]}")
    
    # Check if data looks like noise (good sign)
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
    zcr = zero_crossings / len(data)
    print(f"  Zero crossing rate: {zcr:.4f}")
    
    if zcr > 0.1:
        print("  ✓ Data looks like waveform (good zero crossing rate)")
    else:
        print("  ✗ Low zero crossing rate - might be wrong offset/dtype")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Time domain
    time_axis = np.arange(len(data)) / 1_000_000  # Convert to seconds
    axes[0].plot(time_axis[:1000], data[:1000], linewidth=0.5)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('First 1000 Samples - Time Domain')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(data, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Amplitude')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Amplitude Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_interpretation_test.png', dpi=150)
    print(f"\n  Plot saved to: data_interpretation_test.png")
    plt.show()
    
    print("\n" + "="*80)


def find_header_size_automatically(filepath, max_offset=10000):
    """
    Automatically find the most likely header size
    """
    print("\n" + "="*80)
    print("AUTOMATIC HEADER SIZE DETECTION")
    print("="*80)
    
    with open(filepath, 'rb') as f:
        data = f.read(max_offset)
    
    scores = []
    
    # Try different offsets
    for offset in range(0, max_offset - 1000, 16):
        try:
            # Try int16
            test_data = np.frombuffer(data[offset:offset+1000], dtype=np.int16)
            
            # Score based on:
            # 1. Zero crossing rate (should be moderate for waveforms)
            # 2. Standard deviation (should be reasonable)
            # 3. Value range (should be within int16 range but not constant)
            
            if len(test_data) < 100:
                continue
            
            zero_crossings = np.sum(np.diff(np.sign(test_data)) != 0)
            zcr = zero_crossings / len(test_data)
            std = np.std(test_data)
            value_range = np.ptp(test_data)
            mean_abs = np.mean(np.abs(test_data))
            
            # Good waveform characteristics:
            # - ZCR between 0.1 and 0.8
            # - Std > 10
            # - Range > 100
            # - Mean abs value reasonable
            
            score = 0
            if 0.1 < zcr < 0.8:
                score += 10
            if std > 10:
                score += 5
            if value_range > 100:
                score += 5
            if 10 < mean_abs < 5000:
                score += 5
            
            scores.append((offset, score, zcr, std, mean_abs))
        
        except:
            continue
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most likely header sizes:")
    print("-"*80)
    print(f"{'Offset':>8s} | {'Score':>6s} | {'ZCR':>8s} | {'Std':>10s} | {'Mean Abs':>10s}")
    print("-"*80)
    
    for offset, score, zcr, std, mean_abs in scores[:10]:
        print(f"{offset:8d} | {score:6.1f} | {zcr:8.4f} | {std:10.2f} | {mean_abs:10.2f}")
    
    if scores:
        best_offset = scores[0][0]
        print(f"\n✓ RECOMMENDED HEADER SIZE: {best_offset} bytes")
        print("="*80)
        return best_offset
    
    return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    filepath = r"D:\Pipeline RUL Data\data\raw\B.wfs"
    
    # Step 1: Analyze header
    analyze_express8_header(filepath, bytes_to_read=8192)
    
    # Step 2: Automatically detect header size
    best_offset = find_header_size_automatically(filepath, max_offset=10000)
    
    # Step 3: Test the detected offset
    if best_offset is not None:
        print("\n" + "="*80)
        print("TESTING DETECTED OFFSET")
        print("="*80)
        response = input(f"\nTest reading data with offset={best_offset}? (y/n): ")
        if response.lower() == 'y':
            test_data_interpretation(filepath, best_offset, num_samples=100000, dtype='int16')
    else:
        # Manual test
        print("\nAutomatic detection failed. Please manually test an offset.")
        print("Common offsets: 512, 1024, 2048, 4096")
        try:
            manual_offset = int(input("Enter offset to test: "))
            test_data_interpretation(filepath, manual_offset, num_samples=100000, dtype='int16')
        except:
            print("Invalid input. Exiting.")