#!/usr/bin/env python3
"""
Analyze contact pressure distribution and compare with Hertz theory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from typing import Tuple


def hertz_pressure(x: np.ndarray, p_0: float, a: float) -> np.ndarray:
    """
    Hertz contact pressure distribution.
    p(x) = p_0 * sqrt(1 - (x/a)^2) for |x| <= a
    """
    result = np.zeros_like(x)
    mask = np.abs(x) <= a
    result[mask] = p_0 * np.sqrt(1 - (x[mask]/a)**2)
    return result


def analyze_timestep(filepath: Path, timestep: int, output_dir: Path):
    """Analyze contact pressure for a single timestep."""
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Filter active contact
    threshold = 1e-10
    active = df[df['contact_pressure'] > threshold].copy()
    
    if len(active) < 3:
        print(f"Timestep {timestep}: Insufficient contact points ({len(active)})")
        return
    
    # Sort by x
    active = active.sort_values('x').reset_index(drop=True)
    
    x_data = active['x'].values
    p_data = active['contact_pressure'].values
    
    # Estimate contact radius (x where pressure drops to near zero)
    a_estimate = x_data[-1] * 1.2  # Slightly larger than last contact point
    
    # Fit Hertz model
    try:
        # Initial guess: p_0 = max pressure, a = estimated contact radius
        p0_guess = p_data.max()
        popt, pcov = curve_fit(hertz_pressure, x_data, p_data, 
                              p0=[p0_guess, a_estimate],
                              bounds=([0, 0], [np.inf, 1.0]),
                              maxfev=10000)
        p_0_fit, a_fit = popt
        
        # Generate fitted curve
        x_fit = np.linspace(0, a_fit, 200)
        p_fit = hertz_pressure(x_fit, p_0_fit, a_fit)
        
        # Calculate theoretical pressure at measured points
        p_theory = hertz_pressure(x_data, p_0_fit, a_fit)
        
        # Calculate errors
        errors = (p_data - p_theory) / p_theory * 100  # Percentage error
        rmse = np.sqrt(np.mean((p_data - p_theory)**2))
        
        # Special attention to center node (x=0 or closest)
        center_idx = np.argmin(np.abs(x_data))
        x_center = x_data[center_idx]
        p_center_measured = p_data[center_idx]
        p_center_theory = p_theory[center_idx]
        center_error = errors[center_idx]
        
        # Maximum pressure node
        max_idx = np.argmax(p_data)
        x_max = x_data[max_idx]
        p_max = p_data[max_idx]
        
    except Exception as e:
        print(f"Timestep {timestep}: Fitting failed - {e}")
        return
    
    # Create detailed plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- Subplot 1: Pressure distribution comparison ---
    ax1.plot(x_fit, p_fit, 'r-', linewidth=2, label='Hertz Theory (fitted)')
    ax1.plot(x_data, p_data, 'bo', markersize=8, label='FEA Nodal Data', zorder=5)
    
    # Highlight center node
    ax1.plot(x_center, p_center_measured, 'gs', markersize=12, 
            label=f'Center Node (x={x_center:.4f})', zorder=6)
    
    # Highlight max node if different from center
    if max_idx != center_idx:
        ax1.plot(x_max, p_max, 'r^', markersize=12,
                label=f'Max Node (x={x_max:.4f})', zorder=6)
    
    ax1.set_xlabel('x coordinate (m)', fontsize=12)
    ax1.set_ylabel('Contact Pressure (Pa)', fontsize=12)
    ax1.set_title(f'Contact Pressure vs Hertz Theory (Timestep {timestep})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add fitting parameters
    fit_text = (
        f'Hertz Fit Parameters:\n'
        f'$p_0$ = {p_0_fit:.2f} Pa\n'
        f'$a$ = {a_fit:.4f} m\n'
        f'RMSE = {rmse:.2f} Pa\n'
        f'\n'
        f'Center Node:\n'
        f'Measured: {p_center_measured:.2f} Pa\n'
        f'Theory: {p_center_theory:.2f} Pa\n'
        f'Error: {center_error:.2f}%'
    )
    ax1.text(0.98, 0.97, fit_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # --- Subplot 2: Error distribution ---
    ax2.plot(x_data, errors, 'ro-', markersize=6, linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.plot(x_center, center_error, 'gs', markersize=12, 
            label=f'Center Error: {center_error:.2f}%')
    
    # Highlight if center error is negative (pressure too low)
    if center_error < -1.0:
        ax2.axhspan(center_error, 0, alpha=0.2, color='red',
                   label='Center Underprediction')
    
    ax2.set_xlabel('x coordinate (m)', fontsize=12)
    ax2.set_ylabel('Percentage Error (%)', fontsize=12)
    ax2.set_title('FEA vs Hertz Theory - Percentage Error', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'hertz_analysis_timestep_{timestep:04d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print detailed analysis
    print(f"\n{'='*70}")
    print(f"Timestep {timestep} - Hertz Contact Analysis")
    print(f"{'='*70}")
    print(f"Contact radius (fitted): a = {a_fit:.4f} m")
    print(f"Center pressure (theory): p_0 = {p_0_fit:.2f} Pa")
    print(f"\nCenter Node Analysis:")
    print(f"  Position: x = {x_center:.6f} m")
    print(f"  Measured pressure: {p_center_measured:.2f} Pa")
    print(f"  Theoretical pressure: {p_center_theory:.2f} Pa")
    print(f"  Error: {center_error:+.2f}%")
    
    if max_idx != center_idx:
        print(f"\n⚠️  WARNING: Maximum pressure NOT at center!")
        print(f"  Max pressure location: x = {x_max:.6f} m")
        print(f"  Max pressure value: {p_max:.2f} Pa")
        print(f"  Theoretical pressure at x={x_max:.6f}: {p_theory[max_idx]:.2f} Pa")
        print(f"  This violates Hertz theory!")
        
        # Calculate how much the center SHOULD be higher
        theoretical_ratio = p_center_theory / p_theory[max_idx]
        actual_ratio = p_center_measured / p_max
        print(f"\n  Theory predicts center should be {(theoretical_ratio-1)*100:.1f}% higher than x={x_max:.6f}")
        print(f"  But center is actually {(actual_ratio-1)*100:.1f}% lower than max node")
        print(f"  Total discrepancy: {((theoretical_ratio - actual_ratio))*100:.1f}%")
    
    print(f"\nOverall fitting quality:")
    print(f"  RMSE: {rmse:.2f} Pa")
    print(f"  Mean absolute error: {np.mean(np.abs(errors)):.2f}%")
    print(f"  Max error: {np.max(np.abs(errors)):.2f}%")
    print(f"\nSaved: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze contact pressure against Hertz theory'
    )
    parser.add_argument('--directory', type=str, default='.',
                       help='Directory containing contact_nodes CSV files')
    parser.add_argument('--output', type=str, default='hertz_analysis',
                       help='Output directory for plots')
    parser.add_argument('--timestep', type=int, default=None,
                       help='Specific timestep to analyze (if not set, analyze all)')
    
    args = parser.parse_args()
    
    # Setup directories
    work_dir = Path(args.directory)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Find all contact nodes files
    files = list(work_dir.glob("*_contact_nodes_*.csv"))
    
    def extract_number(filepath: Path) -> int:
        try:
            stem = filepath.stem
            num_str = stem.split('_')[-1]
            return int(num_str)
        except (ValueError, IndexError):
            return -1
    
    files = sorted(files, key=extract_number)
    
    if not files:
        print(f"No contact_nodes files found in {work_dir}")
        return
    
    print(f"Found {len(files)} contact_nodes files")
    
    # Process files
    if args.timestep is not None:
        # Process specific timestep
        target_file = None
        for f in files:
            num = extract_number(f)
            if num == args.timestep:
                target_file = f
                break
        
        if target_file is None:
            print(f"Timestep {args.timestep} not found")
            return
        
        analyze_timestep(target_file, args.timestep, output_dir)
    else:
        # Process all files with sufficient contact
        for filepath in files:
            timestep = extract_number(filepath)
            if timestep >= 1:  # Skip timestep 0 (no contact)
                analyze_timestep(filepath, timestep, output_dir)


if __name__ == '__main__':
    main()
