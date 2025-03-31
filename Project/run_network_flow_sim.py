#!/usr/bin/env python3
"""
Command-line tool to run the Network Flow Simulator with Edmonds-Karp Optimization
Demonstrates the application of graph theory and flow algorithms to network traffic optimization
"""

import argparse
import os
import sys
from network_flow_simulator import NetworkFlowSimulator

def main():
    parser = argparse.ArgumentParser(
        description='Network Flow Simulator with Edmonds-Karp Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to store output files')
    parser.add_argument('--csv', type=str, 
                        help='Load network from CSV file instead of generating')
    parser.add_argument('--bandwidth', type=int, default=1000,
                        help='Total network bandwidth in Mbps')
    parser.add_argument('--simulation-duration', type=int, default=60,
                        help='Duration for traffic simulation in seconds')
    parser.add_argument('--simulation-samples', type=int, default=10,
                        help='Number of samples to take during simulation')
    parser.add_argument('--analyze', action='store_true',
                        help='Run algorithm complexity analysis')
    parser.add_argument('--min-nodes', type=int, default=5,
                        help='Minimum nodes for complexity analysis')
    parser.add_argument('--max-nodes', type=int, default=25,
                        help='Maximum nodes for complexity analysis')
    parser.add_argument('--analyze-samples', type=int, default=3,
                        help='Number of samples per size for analysis')
    parser.add_argument('--report-only', action='store_true',
                        help='Generate report from existing data')
    parser.add_argument('--full-analysis', action='store_true',
                        help='Run complete analysis workflow')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create simulator instance
    simulator = NetworkFlowSimulator(output_dir=args.output_dir)
    simulator.total_bandwidth = args.bandwidth
    
    # Handle command options
    if args.report_only:
        print("Generating report from existing data...")
        report_file = simulator.generate_report()
        print(f"Report generated at {report_file}")
        return
    
    if args.full_analysis:
        print("Running full analysis workflow...")
        results = simulator.run_full_analysis()
        print("\nAnalysis complete!")
        print(f"Network before optimization: {results['network_before']}")
        print(f"Network after optimization: {results['network_after']}")
        print(f"Report: {results['report']}")
        return
    
    # Load or generate network
    if args.csv:
        if not simulator.load_from_csv(args.csv):
            print(f"Error: Failed to load network from {args.csv}")
            return
    else:
        simulator.discover_network_devices()
        simulator.save_devices_to_csv()
    
    # Run traffic simulation
    simulator.simulate_traffic(
        duration=args.simulation_duration,
        samples=args.simulation_samples,
        time_factor=0.1
    )
    
    # Visualize initial network
    simulator.visualize_network(output_file="network_before.png")
    
    # Optimize the network
    simulator.optimize_bandwidth_allocation()
    
    # Visualize optimized network
    simulator.visualize_network(output_file="network_after.png", show_optimization=True)
    
    # Run algorithm analysis if requested
    if args.analyze:
        simulator.analyze_algorithm_complexity(
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            samples=args.analyze_samples
        )
    
    # Generate report
    report_file = simulator.generate_report()
    print(f"Analysis complete! Report generated at {report_file}")

if __name__ == "__main__":
    main()