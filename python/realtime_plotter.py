#!/usr/bin/env python3
"""
Ailuro Real-time Plotter
Receives plotting data via stdin and generates real-time plots.
"""
import sys
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Ailuro real-time plotting tool')
    parser.add_argument('--output', required=True, help='Output image file path')
    parser.add_argument('--xlabel', default='', help='X-axis label')
    parser.add_argument('--ylabel', default='', help='Y-axis label')
    parser.add_argument('--title', default='', help='Plot title')
    parser.add_argument('--figsize', default='5,4', help='Figure size (width,height)')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--style', default='ro-', help='Plot style')
    parser.add_argument('--markersize', type=int, default=5, help='Marker size')
    
    # only parse command line arguments, ignore stdin data stream
    args = parser.parse_args(sys.argv[1:])
    
    # Parse figure size
    try:
        figwidth, figheight = map(float, args.figsize.split(','))
    except:
        figwidth, figheight = 5.0, 4.0
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    plot_line, = ax.plot([], [], args.style, markersize=args.markersize, markerfacecolor='none')
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    
    x_data = []
    y_data = []
    
    sys.stderr.flush()
    
    # Read data from stdin
    for line in sys.stdin:
        line = line.strip()
        
        if line == 'CLOSE':
            break
        
        if not line or line.startswith('#'):
            continue
        
        try:
            # batch data mode: DATA:N followed by N lines of x,y data
            if line.startswith('DATA:'):
                num_points = int(line.split(':')[1])
                x_data.clear()
                y_data.clear()
                
                for _ in range(num_points):
                    data_line = sys.stdin.readline().strip()
                    if data_line == 'END':
                        break
                    x_str, y_str = data_line.split(',')
                    x_data.append(float(x_str))
                    y_data.append(float(y_str))
                
                # Read END marker (if not already read)
                end_marker = sys.stdin.readline().strip()
                if end_marker != 'END':
                    print(f'WARNING: Expected END marker, got: {end_marker}', flush=True)
                
                # Update plot
                plot_line.set_data(x_data, y_data)
                ax.relim()
                ax.autoscale_view()
                
                # Save image
                plt.tight_layout()
                fig.savefig(args.output, dpi=args.dpi)
                
                sys.stderr.flush()
        except ValueError as e:
            print(f'ERROR: Invalid data format: {line}', flush=True)
            sys.stderr.write(f'ERROR: {e}\n')
            sys.stderr.flush()
        except Exception as e:
            print(f'ERROR: {e}', flush=True)
            sys.stderr.write(f'ERROR: {e}\n')
            sys.stderr.flush()
    
    sys.stderr.flush()


if __name__ == '__main__':
    main()
