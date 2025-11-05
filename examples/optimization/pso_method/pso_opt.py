import argparse
from pathlib import Path
import yaml
import numpy as np
import time

def rastrigin_3d(x):
    A = 10
    return A*3 + np.sum(x**2 - A * np.cos(2*np.pi*x), axis=1)

class MOOSEObjectiveFunction:
    """Objective function wrapper for MOOSE simulations."""
    def __init__(self, config):
        print("Initializing MOOSE Objective Function...")

        
def run_pso_optimization(config: dict):
    """
    Run PSO optimization
    
    Args:
        config (dict): Configuration dictionary
    """
    from pyswarms.single.global_best import GlobalBestPSO

    print("="*80)
    print(f"Particle Swarm Optimization (MOOSE)")
    print("="*80)

    print(f"\nRunning PSO Optimization...")

    # Create objective function
    # objective_func = 
    objective_func = rastrigin_3d

    # Parameters configuration
    param_config = config['parameters']
    lower_bounds = np.array(param_config['lower_bounds'])
    upper_bounds = np.array(param_config['upper_bounds'])
    bounds = (lower_bounds, upper_bounds)
    print(f"\nParameter Configuration:")
    for i, name in enumerate(param_config['names']):
        print(f"  {name}: [{lower_bounds[i]}, {upper_bounds[i]}]")

    # PSO parameters
    pso_config = config['pso']
    n_particles = pso_config['n_particles']
    max_iters = pso_config['max_iterations']

    options = {
        'c1': pso_config['cognitive_param'],  # cognitive parameter
        'c2': pso_config['social_param'],     # social parameter
        'w': pso_config['inertia_weight']     # inertia weight
    }

    print(f"\nPSO Configuration:")
    print(f"  Number of particles: {n_particles}")
    print(f"  Maximum iterations:  {max_iters}")
    print(f"  Cognitive parameter: {options['c1']}")
    print(f"  Social parameter:    {options['c2']}")
    print(f"  Inertia weight:      {options['w']}")

    # Create PSO optimizer
    print("\nStarting optimization...")
    print("-"*80)

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(param_config['names']),
        options=options,
        bounds=bounds
    )

    # Perform optimization
    start_time = time.time()
    cost, pos = optimizer.optimize(
        objective_func,
        iters=max_iters
    )

    elapsed_time = time.time() - start_time

    # Output results
    print("-"*80)
    print(f"Optimization completed in {elapsed_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description='PSO + MOOSE Optimization')
    parser.add_argument('--config', default='pso_config.yaml',
                       help='config file path (default: pso_config.yaml)')
    args = parser.parse_args()

    # Load configuration
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Error: config file does not exist: {config_file}")
        print(f"Please create the config file.")
        return

    print(f"Using config file: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # print("\nConfiguration:")
    # for key, value in config.items():
    #     print(f"  {key}: {value}")

    # Run Optimization
    run_pso_optimization(config)

if __name__ == "__main__":
    main()