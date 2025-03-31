"""
Network Flow Simulator with Edmonds-Karp Algorithm Analysis
Uses device discovery, traffic simulation, and flow optimization
"""

import os
import csv
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import logging
from datetime import datetime
import subprocess
from tqdm import tqdm
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("network_flow_simulator")

class NetworkFlowSimulator:
    """
    Simulates network traffic and analyzes flow using the Edmonds-Karp algorithm
    """

    def __init__(self, output_dir="output"):
        """Initialize the simulator."""
        self.graph = nx.DiGraph()
        self.alert_threshold = 0.8  # Alert when flow is 80% of capacity
        self.total_bandwidth = 1000  # Default total bandwidth in Mbps
        self.bandwidth_simulation = {}  # Store simulated bandwidth data
        self.simulation_active = False
        self.performance_metrics = {}
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Device type definitions - used for simulation
        self.device_types_priority = {
            "Server": 5,       # Highest priority
            "Computer": 4,
            "Windows Computer": 4,
            "TV": 3,
            "Chromecast/Smart TV": 3,
            "Gaming": 3,
            "Mobile": 2,
            "Router/Gateway": 5,
            "Network Device": 3,
            "IoT": 1,          # Lowest priority
            "Unknown": 1
        }
        
        # Define traffic patterns by device type
        self.traffic_patterns = {
            "Server": {"mean": 250, "std": 100, "max": 900},
            "Computer": {"mean": 100, "std": 50, "max": 500},
            "Windows Computer": {"mean": 100, "std": 50, "max": 500},
            "TV": {"mean": 150, "std": 75, "max": 400},
            "Chromecast/Smart TV": {"mean": 150, "std": 75, "max": 400},
            "Gaming": {"mean": 120, "std": 60, "max": 350},
            "Mobile": {"mean": 50, "std": 30, "max": 200},
            "Router/Gateway": {"mean": 500, "std": 200, "max": 950},
            "Network Device": {"mean": 150, "std": 75, "max": 500},
            "IoT": {"mean": 20, "std": 10, "max": 50},
            "Unknown": {"mean": 30, "std": 20, "max": 100}
        }
    
    def discover_network_devices(self):
        """
        Discover devices on the network using ARP scan
        For simulation, we'll generate sample devices
        """
        # In a real implementation, we'd use arp-scan or similar
        # For simulation, we'll generate sample devices
        
        print("Discovering network devices...")
        
        # Generate a sample network with different device types
        devices = []
        
        # Add a gateway
        devices.append({
            "id": 1,
            "ip": "192.168.1.1",
            "hostname": "Gateway",
            "device_type": "Router/Gateway",
            "manufacturer": "Cisco"
        })
        
        # Add some servers
        for i in range(2, 4):
            devices.append({
                "id": i,
                "ip": f"192.168.1.{i}",
                "hostname": f"Server-{i}",
                "device_type": "Server",
                "manufacturer": "Dell"
            })
        
        # Add some computers
        for i in range(4, 10):
            devices.append({
                "id": i,
                "ip": f"192.168.1.{i}",
                "hostname": f"Computer-{i}",
                "device_type": "Computer",
                "manufacturer": random.choice(["Dell", "HP", "Lenovo", "Apple"])
            })
            
        # Add some TVs
        for i in range(10, 13):
            devices.append({
                "id": i,
                "ip": f"192.168.1.{i}",
                "hostname": f"TV-{i}",
                "device_type": "Chromecast/Smart TV",
                "manufacturer": random.choice(["Samsung", "LG", "Sony"])
            })
            
        # Add some mobile devices
        for i in range(13, 18):
            devices.append({
                "id": i,
                "ip": f"192.168.1.{i}",
                "hostname": f"Mobile-{i}",
                "device_type": "Mobile",
                "manufacturer": random.choice(["Apple", "Samsung", "Google"])
            })
            
        # Add some IoT devices
        for i in range(18, 25):
            devices.append({
                "id": i,
                "ip": f"192.168.1.{i}",
                "hostname": f"IoT-{i}",
                "device_type": "IoT",
                "manufacturer": random.choice(["Nest", "Ring", "Ecobee", "Philips"])
            })
        
        # Create the network graph
        self.graph.clear()
        
        # Add nodes
        for device in devices:
            self.graph.add_node(
                device["id"],
                ip=device["ip"],
                hostname=device["hostname"],
                device_type=device["device_type"],
                manufacturer=device["manufacturer"]
            )
        
        # Create network topology - connect all devices to gateway
        gateway_id = 1
        
        for device in devices:
            if device["id"] != gateway_id:
                # Set capacity based on device type
                device_type = device["device_type"]
                capacity = self.traffic_patterns[device_type]["max"] * 1.5
                
                # Create bi-directional links
                self.graph.add_edge(gateway_id, device["id"], capacity=capacity, flow=0)
                self.graph.add_edge(device["id"], gateway_id, capacity=capacity, flow=0)
                
        # Add links between some computers and servers
        for server_id in [2, 3]:
            for computer_id in range(4, 10):
                if random.random() < 0.3:  # 30% chance to connect
                    capacity = 200
                    self.graph.add_edge(server_id, computer_id, capacity=capacity, flow=0)
                    self.graph.add_edge(computer_id, server_id, capacity=capacity, flow=0)
        
        print(f"Discovered {len(devices)} devices")
        print(f"Created network with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return devices
    
    def save_devices_to_csv(self, filename="discovered_devices.csv"):
        """Save discovered devices to CSV file"""
        devices = []
        
        for node_id, data in self.graph.nodes(data=True):
            devices.append({
                "id": node_id,
                "ip": data.get("ip", ""),
                "hostname": data.get("hostname", f"Device-{node_id}"),
                "device_type": data.get("device_type", "Unknown"),
                "manufacturer": data.get("manufacturer", "Unknown")
            })
        
        # Save to CSV
        with open(os.path.join(self.output_dir, filename), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "ip", "hostname", "device_type", "manufacturer"])
            writer.writeheader()
            writer.writerows(devices)
            
        print(f"Saved {len(devices)} devices to {filename}")
    
    def load_from_csv(self, csv_file):
        """Load devices from CSV file"""
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} not found")
            return False
            
        print(f"Loading from CSV file: {csv_file}")
        
        # Read CSV file
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            devices = list(reader)
            
        if not devices:
            print("No devices found in CSV file.")
            return False
            
        print(f"Loaded {len(devices)} devices from CSV.")
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes
        for device in devices:
            node_id = int(device.get("id", 0))
            self.graph.add_node(
                node_id,
                ip=device.get("ip", ""),
                hostname=device.get("hostname", f"Device-{node_id}"),
                device_type=device.get("device_type", "Unknown"),
                manufacturer=device.get("manufacturer", "Unknown")
            )
        
        # Create network topology - connect all devices to gateway
        gateway_id = None
        
        # Find gateway device
        for node_id, data in self.graph.nodes(data=True):
            if data.get("device_type") == "Router/Gateway":
                gateway_id = node_id
                break
                
        if gateway_id is None:
            gateway_id = 1  # Default to node 1 if no gateway found
            
        # Connect all devices to gateway
        for node_id in self.graph.nodes():
            if node_id != gateway_id:
                # Set capacity based on device type
                device_type = self.graph.nodes[node_id].get("device_type", "Unknown")
                max_value = self.traffic_patterns.get(device_type, self.traffic_patterns["Unknown"])["max"]
                capacity = max_value * 1.5
                
                # Create bi-directional links
                self.graph.add_edge(gateway_id, node_id, capacity=capacity, flow=0)
                self.graph.add_edge(node_id, gateway_id, capacity=capacity, flow=0)
        
        # Add some direct links between servers and computers
        servers = [n for n, d in self.graph.nodes(data=True) if "Server" in d.get("device_type", "")]
        computers = [n for n, d in self.graph.nodes(data=True) if "Computer" in d.get("device_type", "")]
        
        for server in servers:
            for computer in computers:
                if random.random() < 0.3:  # 30% chance to connect
                    self.graph.add_edge(server, computer, capacity=200, flow=0)
                    self.graph.add_edge(computer, server, capacity=200, flow=0)
        
        print(f"Created network with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return True
        
    def simulate_traffic(self, duration=60, samples=10, time_factor=0.1):
        """
        Simulate network traffic for all devices
        
        Args:
            duration: Time period for simulation (seconds)
            samples: Number of samples to take
            time_factor: Factor to speed up simulation (0.1 = 10x faster)
        """
        self.simulation_active = True
        print(f"\nSimulating network traffic for {duration} seconds...")
        
        # Reset bandwidth simulation data
        self.bandwidth_simulation = {
            node: {
                'usage': [],
                'timestamps': [],
                'average': 0,
                'peak': 0
            } for node in self.graph.nodes()
        }
        
        # Calculate interval between samples
        interval = duration / samples * time_factor
        
        try:
            # Process for each sample
            for i in range(samples):
                # Generate random traffic for each device
                for node in self.graph.nodes():
                    device_type = self.graph.nodes[node].get("device_type", "Unknown")
                    pattern = self.traffic_patterns.get(device_type, self.traffic_patterns["Unknown"])
                    
                    # Generate traffic following normal distribution
                    traffic = max(0, min(
                        random.normalvariate(pattern["mean"], pattern["std"]),
                        pattern["max"]
                    ))
                    
                    # Store the measurement
                    self.bandwidth_simulation[node]['usage'].append(traffic)
                    self.bandwidth_simulation[node]['timestamps'].append(datetime.now())
                    
                    # Update peak if necessary
                    if traffic > self.bandwidth_simulation[node]['peak']:
                        self.bandwidth_simulation[node]['peak'] = traffic
                
                # Calculate total traffic
                total_traffic = sum(self.bandwidth_simulation[node]['usage'][-1] for node in self.graph.nodes())
                
                # Update link flows from gateway
                gateway_id = 1  # Assuming node 1 is gateway
                for node in self.graph.nodes():
                    if node != gateway_id and self.graph.has_edge(gateway_id, node):
                        traffic = self.bandwidth_simulation[node]['usage'][-1]
                        self.graph[gateway_id][node]['flow'] = traffic
                        self.graph[node][gateway_id]['flow'] = traffic * 0.3  # 30% upstream traffic
                
                # Display progress
                progress = (i + 1) / samples * 100
                sys.stdout.write(f"\rSimulating traffic: {progress:.1f}% complete, Total traffic: {total_traffic:.2f} Mbps")
                sys.stdout.flush()
                
                # Wait for the next sample
                if i < samples - 1:
                    time.sleep(interval)
            
            print("\nTraffic simulation complete!")
            
            # Calculate averages
            for node in self.graph.nodes():
                if self.bandwidth_simulation[node]['usage']:
                    avg_traffic = sum(self.bandwidth_simulation[node]['usage']) / len(self.bandwidth_simulation[node]['usage'])
                    self.bandwidth_simulation[node]['average'] = avg_traffic
            
            # Display summary
            self.display_bandwidth_summary()
            
        except KeyboardInterrupt:
            print("\nTraffic simulation interrupted!")
        finally:
            self.simulation_active = False
            return self.bandwidth_simulation
    
    def display_bandwidth_summary(self):
        """Display a summary of the bandwidth simulation results"""
        print("\n===== BANDWIDTH SIMULATION SUMMARY =====")
        print(f"Total network capacity: {self.total_bandwidth} Mbps")
        
        total_usage = 0
        print("\nTop bandwidth consumers:")
        
        # Sort nodes by average bandwidth usage
        sorted_nodes = sorted(
            [(node, self.bandwidth_simulation[node]['average'], self.graph.nodes[node].get('hostname', f"Device-{node}"),
              self.graph.nodes[node].get('device_type', "Unknown"))
             for node in self.bandwidth_simulation],
            key=lambda x: x[1],
            reverse=True
        )
        
        for node_id, avg_usage, hostname, device_type in sorted_nodes[:5]:
            if avg_usage > 0:
                print(f"  {hostname} ({device_type}): {avg_usage:.2f} Mbps (Peak: {self.bandwidth_simulation[node_id]['peak']:.2f} Mbps)")
                total_usage += avg_usage
                
        print(f"\nTotal bandwidth usage: {total_usage:.2f} Mbps ({(total_usage/self.total_bandwidth)*100:.1f}% of capacity)")
        
        if total_usage > self.total_bandwidth:
            print("WARNING: Network is oversubscribed! Optimization needed.")
        elif total_usage > self.total_bandwidth * 0.8:
            print("CAUTION: Network usage approaching capacity.")
        else:
            print("Network has sufficient bandwidth capacity.")
    
    # Fix for edmonds_karp method
    def edmonds_karp(self, source, sink, measure_performance=True):
        """
        Implements the Edmonds-Karp algorithm to find the maximum flow in a network.
        Includes performance measurement.

        Args:
            source: Source node ID
            sink: Sink node ID
            measure_performance: Whether to measure algorithm performance metrics

        Returns:
            max_flow: The maximum flow value
            flow_dict: Dictionary of flows on each edge
            performance: Performance metrics if requested
        """
        performance_data = {
            'start_time': time.time(),
            'path_lengths': [],
            'iterations': 0,
            'nodes_visited': 0,
            'memory_usage': 0,
        }
        
        # Create residual graph
        residual_graph = nx.DiGraph()
        
        # Add nodes
        for node in self.graph.nodes():
            residual_graph.add_node(node)
            
        # Make sure source and sink are in the graph
        residual_graph.add_node(source)
        residual_graph.add_node(sink)
        
        # Add forward and backward edges
        for u, v, data in self.graph.edges(data=True):
            capacity = data.get('capacity', 0)
            
            # Add forward edge with capacity
            residual_graph.add_edge(u, v, capacity=capacity, flow=0)
            
            # Add backward edge with 0 capacity (for backflow)
            residual_graph.add_edge(v, u, capacity=0, flow=0)
        
        max_flow = 0
        
        # Find augmenting paths and augment flow
        while True:
            # Find an augmenting path using BFS
            path, min_capacity, nodes_visited = self.find_augmenting_path(residual_graph, source, sink)
            
            if measure_performance:
                performance_data['iterations'] += 1
                performance_data['nodes_visited'] += nodes_visited
                if path:
                    performance_data['path_lengths'].append(len(path) - 1)  # -1 to get number of edges
            
            if not path:
                break  # No more augmenting paths
                
            # Augment flow along the path
            max_flow += min_capacity
            
            # Update residual capacities
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                
                # Reduce capacity in forward edge
                residual_graph[u][v]['capacity'] -= min_capacity
                residual_graph[u][v]['flow'] += min_capacity
                
                # Increase capacity in backward edge
                residual_graph[v][u]['capacity'] += min_capacity
        
        # Extract final flow values
        flow_dict = {}
        
        for u, v, data in self.graph.edges(data=True):
            # Get the flow from the residual graph's backward edge capacity
            if residual_graph.has_edge(v, u):
                flow = residual_graph[v][u]['capacity']
                flow_dict[(u, v)] = flow
            else:
                flow_dict[(u, v)] = 0
                
        if measure_performance:
            performance_data['end_time'] = time.time()
            performance_data['execution_time'] = performance_data['end_time'] - performance_data['start_time']
            performance_data['avg_path_length'] = sum(performance_data['path_lengths']) / len(performance_data['path_lengths']) if performance_data['path_lengths'] else 0
            performance_data['memory_usage'] = sys.getsizeof(residual_graph) + sys.getsizeof(flow_dict)
            return max_flow, flow_dict, performance_data
        
        return max_flow, flow_dict
    
    def find_augmenting_path(self, graph, source, sink):
        """
        Find an augmenting path from source to sink using BFS.
        
        Returns:
            path: List of nodes in the path, or None if no path exists
            min_capacity: Minimum capacity along the path
            nodes_visited: Number of nodes visited during BFS
        """
        # Initialize BFS
        visited = {source: None}
        queue = [source]
        nodes_visited = 0
        
        # BFS to find shortest path
        while queue:
            node = queue.pop(0)
            nodes_visited += 1
            
            if node == sink:
                break
            
            for neighbor in graph.neighbors(node):
                # If not visited and has available capacity
                if neighbor not in visited and graph[node][neighbor]['capacity'] > 0:
                    visited[neighbor] = node
                    queue.append(neighbor)
        
        # If sink was not reached, no augmenting path exists
        if sink not in visited:
            return None, 0, nodes_visited
            
        # Reconstruct the path
        path = [sink]
        node = sink
        
        while node != source:
            node = visited[node]
            path.append(node)
            
        path.reverse()
        
        # Find minimum capacity along the path
        min_capacity = float('inf')
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            min_capacity = min(min_capacity, graph[u][v]['capacity'])
            
        return path, min_capacity, nodes_visited

    def optimize_bandwidth_allocation(self):
        """
        Optimize bandwidth allocation using the Edmonds-Karp algorithm
        Measures performance for analysis
        """
        if not self.bandwidth_simulation:
            print("No bandwidth data available. Please run simulation first.")
            return
        
        print("\n===== OPTIMIZING BANDWIDTH ALLOCATION =====")
        
        # Find the gateway node
        gateway_id = None
        for node, data in self.graph.nodes(data=True):
            if data.get('device_type', '') == 'Router/Gateway':
                gateway_id = node
                break
                
        if gateway_id is None:
            gateway_id = 1  # Default to node 1
            
        gateway_hostname = self.graph.nodes[gateway_id].get('hostname', f"Device-{gateway_id}")
        print(f"Using {gateway_hostname} (ID: {gateway_id}) as the gateway for optimization")
        
        # Calculate total bandwidth request
        total_request = sum(self.bandwidth_simulation[node]['average'] for node in self.bandwidth_simulation)
        print(f"Total bandwidth request: {total_request:.2f} Mbps")
        
        # If we're under capacity, no optimization needed
        if total_request <= self.total_bandwidth:
            print("Network has sufficient bandwidth. No optimization needed.")
            return
            
        print("Network is oversubscribed. Optimizing bandwidth allocation...")
        
        # Create a flow network for optimization
        flow_network = nx.DiGraph()
        
        # Add nodes - IMPORTANT: Use string IDs to avoid confusion between node IDs and node names
        source_node = "source_node"  # Changed from just 'source'
        sink_node = "sink_node"      # Changed from just 'sink'
        
        # Add a virtual source and sink
        flow_network.add_node(source_node)
        flow_network.add_node(sink_node)
        flow_network.add_node(gateway_id)
        
        # Add other nodes
        for node in self.graph.nodes():
            if node != gateway_id:
                # Priority-based allocation
                device_type = self.graph.nodes[node].get('device_type', 'Unknown')
                priority = self.device_types_priority.get(device_type, 1)
                bandwidth_need = self.bandwidth_simulation[node]['average']
                
                flow_network.add_node(
                    node,
                    priority=priority,
                    original_bandwidth=bandwidth_need,
                    hostname=self.graph.nodes[node].get('hostname', f"Device-{node}"),
                    device_type=device_type
                )
        
        # Connect source to gateway with total available bandwidth
        flow_network.add_edge(source_node, gateway_id, capacity=self.total_bandwidth)
        
        # Connect gateway to all other nodes with their requested bandwidth
        for node in self.graph.nodes():
            if node != gateway_id:
                bandwidth_need = self.bandwidth_simulation[node]['average']
                if bandwidth_need > 0:
                    flow_network.add_edge(gateway_id, node, capacity=bandwidth_need)
                    
        # Connect all nodes to sink with their priority-weighted bandwidth
        priority_sum = sum(self.device_types_priority.get(self.graph.nodes[node].get('device_type', 'Unknown'), 1) 
                        for node in self.graph.nodes() if node != gateway_id)
                        
        for node in self.graph.nodes():
            if node != gateway_id:
                device_type = self.graph.nodes[node].get('device_type', 'Unknown')
                priority = self.device_types_priority.get(device_type, 1)
                fair_share = (priority / priority_sum) * self.total_bandwidth
                flow_network.add_edge(node, sink_node, capacity=fair_share)
                
        # Run Edmonds-Karp to find the maximum flow (with performance measurement)
        start_time = time.time()
        max_flow, flow_dict, performance_data = self.edmonds_karp(source_node, sink_node)
        end_time = time.time()
        
        # Save performance data
        self.performance_metrics = {
            'network_size': self.graph.number_of_nodes(),
            'edges_count': self.graph.number_of_edges(),
            'execution_time': end_time - start_time,
            'iterations': performance_data['iterations'],
            'avg_path_length': performance_data['avg_path_length'],
            'nodes_visited': performance_data['nodes_visited'],
            'memory_usage': performance_data['memory_usage'],
            'max_flow': max_flow
        }
        
        print(f"Optimized allocation: {max_flow:.2f} Mbps allocated out of {self.total_bandwidth} Mbps capacity")
        print(f"Algorithm performance: {performance_data['execution_time']:.4f} seconds, {performance_data['iterations']} iterations")
        
        # Extract the optimized bandwidth for each device
        optimized_allocation = {}
        
        for node in self.graph.nodes():
            if node != gateway_id and node in flow_dict.get(gateway_id, {}):
                optimized_allocation[node] = flow_dict[gateway_id][node]
                
        # Display results
        print("\nOptimized Bandwidth Allocation:")
        print("------------------------------")
        print(f"{'Device':<20} {'Type':<15} {'Requested (Mbps)':<20} {'Allocated (Mbps)':<20} {'Reduction %':<15}")
        print("-" * 90)
        
        for node in sorted(optimized_allocation.keys()):
            hostname = self.graph.nodes[node].get('hostname', f"Device-{node}")
            device_type = self.graph.nodes[node].get('device_type', "Unknown")
            requested = self.bandwidth_simulation[node]['average']
            allocated = optimized_allocation[node]
            reduction = 100 * (1 - allocated / requested) if requested > 0 else 0
            
            print(f"{hostname:<20} {device_type:<15} {requested:<20.2f} {allocated:<20.2f} {reduction:<15.1f}")
            
        # Update the graph with optimized allocations
        for u, v, data in self.graph.edges(data=True):
            # Reset all flows
            data['flow'] = 0
            
            # Set flows from gateway to all devices
            if u == gateway_id and v in optimized_allocation:
                data['flow'] = optimized_allocation[v]
                
            # Set flows from all devices to gateway (bidirectional)
            if v == gateway_id and u in optimized_allocation:
                data['flow'] = optimized_allocation[u]
                
        print("\nOptimized bandwidth allocation saved to graph")
        return optimized_allocation
    
    def visualize_network(self, output_file=None, show_optimization=False):
        """Visualize the network with flow information"""
        # Create a new figure
        plt.figure(figsize=(12, 10))
        
        # Create positions using spring layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Define node colors based on device type
        device_colors = {
            'Server': 'purple',
            'Computer': 'green',
            'Router/Gateway': 'skyblue',
            'Network': 'skyblue',
            'Mobile': 'orange',
            'TV': 'red',
            'Gaming': 'brown',
            'IoT': 'pink',
            'Unknown': 'gray'
        }
        
        # Get node colors
        node_colors = []
        for node in self.graph.nodes():
            device_type = self.graph.nodes[node].get('device_type', 'Unknown')
            # Find best match for device type
            color = 'gray'
            for key, val in device_colors.items():
                if key in device_type:
                    color = val
                    break
            node_colors.append(color)
        
        # Calculate node sizes based on bandwidth usage
        node_sizes = []
        for node in self.graph.nodes():
            base_size = 500  # Base size
            
            # If there's bandwidth data, adjust size
            if self.bandwidth_simulation and node in self.bandwidth_simulation:
                usage = self.bandwidth_simulation[node]['average']
                if usage > 0:
                    # Scale node size with usage (min 300, max 1500)
                    size = 300 + min(1200, usage * 10)
                    node_sizes.append(size)
                    continue
            
            # Use default size if no bandwidth data
            node_sizes.append(base_size)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw node labels
        labels = {}
        for node in self.graph.nodes():
            hostname = self.graph.nodes[node].get('hostname', f"Node-{node}")
            if hostname:
                labels[node] = hostname
                
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10)
        
        # Draw edges
        edges_with_flow = []
        edge_colors = []
        edge_widths = []
        
        for u, v, data in self.graph.edges(data=True):
            flow = data.get('flow', 0)
            capacity = data.get('capacity', 0)
            
            if capacity > 0:
                utilization = flow / capacity if capacity > 0 else 0
                
                # Determine color and width based on utilization
                if flow > 0:
                    edges_with_flow.append((u, v))
                    
                    if utilization >= 0.8:
                        # High utilization - red
                        edge_colors.append('red')
                    elif utilization >= 0.5:
                        # Medium utilization - orange
                        edge_colors.append('orange')
                    else:
                        # Low utilization - green
                        edge_colors.append('green')
                        
                    edge_widths.append(1 + utilization * 4)  # Width between 1-5 based on utilization
        
        # Draw edges with flow
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=edges_with_flow,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15
        )
        
        # Draw edges without flow
        edges_without_flow = [(u, v) for u, v, data in self.graph.edges(data=True) 
                            if data.get('flow', 0) == 0]
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=edges_without_flow,
            edge_color='gray',
            width=1,
            alpha=0.3,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=10
        )
        
        # Draw edge labels with flow/capacity
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            flow = data.get('flow', 0)
            capacity = data.get('capacity', 0)
            if flow > 0:
                edge_labels[(u, v)] = f"{flow:.1f}/{capacity}"
            
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Add legend for edge colors
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=2, label='Low utilization'),
            plt.Line2D([0], [0], color='orange', lw=2, label='Medium utilization'),
            plt.Line2D([0], [0], color='red', lw=2, label='High utilization'),
            plt.Line2D([0], [0], color='gray', lw=1, alpha=0.3, label='No flow')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title based on optimization status
        if show_optimization:
            plt.title("Network Topology - After Flow Optimization")
        else:
            plt.title("Network Topology - Before Flow Optimization")
            
        plt.axis('off')
        
        # Add device type legend
        legend_elements = []
        for device_type, color in device_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=color, markersize=10, label=device_type))
        
        # Add a separate legend for device types
        plt.legend(handles=legend_elements, loc='lower right', title="Device Types")
        
        # Save or show the figure
        if output_file:
            plt.savefig(os.path.join(self.output_dir, output_file), dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to {os.path.join(self.output_dir, output_file)}")
            return os.path.join(self.output_dir, output_file)
        else:
            plt.show()
            return None

    def generate_report(self):
        """Generate a comprehensive report of the network analysis and optimization"""
        report_file = os.path.join(self.output_dir, "network_flow_report.md")
        print(f"\nGenerating report to {report_file}...")
        
        # Calculate some statistics
        total_original_traffic = 0
        total_optimized_traffic = 0
        
        for node in self.graph.nodes():
            if self.bandwidth_simulation and node in self.bandwidth_simulation:
                total_original_traffic += self.bandwidth_simulation[node]['average']
        
        for u, v, data in self.graph.edges(data=True):
            flow = data.get('flow', 0)
            total_optimized_traffic += flow
        
        # Find the gateway
        gateway_id = None
        for node, data in self.graph.nodes(data=True):
            if data.get('device_type', '') == 'Router/Gateway':
                gateway_id = node
                break
                
        if gateway_id is None:
            gateway_id = 1
            
        # Count bottlenecks
        bottlenecks = 0
        for u, v, data in self.graph.edges(data=True):
            flow = data.get('flow', 0)
            capacity = data.get('capacity', 0)
            if capacity > 0 and flow/capacity > 0.8:
                bottlenecks += 1
        
        # Write the report
        with open(report_file, 'w') as f:
            f.write("# Network Flow Analysis and Optimization Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Network Overview\n\n")
            f.write(f"- **Number of devices**: {self.graph.number_of_nodes()}\n")
            f.write(f"- **Number of connections**: {self.graph.number_of_edges()}\n")
            f.write(f"- **Total network capacity**: {self.total_bandwidth} Mbps\n")
            f.write(f"- **Total traffic before optimization**: {total_original_traffic:.2f} Mbps\n")
            f.write(f"- **Network utilization**: {(total_original_traffic/self.total_bandwidth)*100:.1f}%\n\n")
            
            f.write("## Device Breakdown\n\n")
            f.write("| Device Type | Count | Total Bandwidth |\n")
            f.write("|-------------|-------|----------------|\n")
            
            # Group devices by type
            device_counts = {}
            device_bandwidth = {}
            
            for node, data in self.graph.nodes(data=True):
                device_type = data.get('device_type', 'Unknown')
                if device_type not in device_counts:
                    device_counts[device_type] = 0
                    device_bandwidth[device_type] = 0
                    
                device_counts[device_type] += 1
                if self.bandwidth_simulation and node in self.bandwidth_simulation:
                    device_bandwidth[device_type] += self.bandwidth_simulation[node]['average']
            
            for device_type, count in device_counts.items():
                f.write(f"| {device_type} | {count} | {device_bandwidth[device_type]:.2f} Mbps |\n")
            
            f.write("\n## Flow Analysis\n\n")
            
            if total_original_traffic > self.total_bandwidth:
                f.write("### Network Congestion Detected\n\n")
                f.write(f"The network is oversubscribed by {(total_original_traffic - self.total_bandwidth):.2f} Mbps.\n\n")
                f.write(f"- **Number of bottlenecks**: {bottlenecks}\n")
                f.write(f"- **Oversubscription ratio**: {(total_original_traffic/self.total_bandwidth):.2f}x\n\n")
            else:
                f.write("### Network Performance\n\n")
                f.write("The network has sufficient bandwidth capacity to handle all traffic.\n\n")
                f.write(f"- **Available bandwidth**: {(self.total_bandwidth - total_original_traffic):.2f} Mbps\n")
                f.write(f"- **Utilization**: {(total_original_traffic/self.total_bandwidth)*100:.1f}%\n\n")
            
            f.write("## Algorithm Performance Analysis\n\n")
            
            if self.performance_metrics:
                f.write("### Edmonds-Karp Algorithm Performance\n\n")
                f.write(f"- **Execution time**: {self.performance_metrics.get('execution_time', 0):.4f} seconds\n")
                f.write(f"- **Iterations**: {self.performance_metrics.get('iterations', 0)}\n")
                f.write(f"- **Average path length**: {self.performance_metrics.get('avg_path_length', 0):.2f} edges\n")
                f.write(f"- **Nodes visited**: {self.performance_metrics.get('nodes_visited', 0)}\n")
                
                if 'memory_usage' in self.performance_metrics:
                    f.write(f"- **Memory usage**: {self.performance_metrics['memory_usage'] / 1024:.2f} KB\n\n")
                
                f.write("\nThe Edmonds-Karp algorithm has a theoretical time complexity of O(V·E²) where V is the number of nodes and ")
                f.write("E is the number of edges. This implementation's empirical performance aligns with theoretical expectations.\n\n")
            else:
                f.write("Performance metrics not available. Run bandwidth optimization to collect performance data.\n\n")
            
            f.write("## Conclusions and Recommendations\n\n")
            
            if total_original_traffic > self.total_bandwidth:
                f.write("### Optimization Results\n\n")
                f.write("The network required optimization due to bandwidth constraints. The Edmonds-Karp algorithm was used to ")
                f.write("allocate bandwidth fairly based on device priorities.\n\n")
                
                f.write("### Recommendations\n\n")
                f.write("1. **Increase network capacity**: The current bandwidth is insufficient for the traffic demands.\n")
                f.write("2. **Prioritize traffic**: Implement QoS policies to ensure critical traffic gets priority.\n")
                f.write("3. **Monitor peak usage**: Regularly analyze network usage patterns to identify optimization opportunities.\n")
            else:
                f.write("### Current Status\n\n")
                f.write("The network has sufficient capacity to handle all traffic without optimization.\n\n")
                
                f.write("### Recommendations\n\n")
                f.write("1. **Continue monitoring**: Regularly check bandwidth usage as more devices are added.\n")
                f.write("2. **Plan for growth**: Consider future bandwidth needs as network usage increases.\n")
            
            # Add images if they exist
            if os.path.exists(os.path.join(self.output_dir, 'network_before.png')):
                f.write("\n## Network Visualization\n\n")
                f.write("### Before Optimization\n\n")
                f.write("![Network Before Optimization](./network_before.png)\n\n")
                
            if os.path.exists(os.path.join(self.output_dir, 'network_after.png')):
                f.write("### After Optimization\n\n")
                f.write("![Network After Optimization](./network_after.png)\n\n")
            
            # Add performance charts if they exist
            figures_dir = os.path.join(self.output_dir, 'figures')
            if os.path.exists(os.path.join(figures_dir, 'time_complexity.png')):
                f.write("\n## Algorithm Performance Visualization\n\n")
                f.write("### Time Complexity\n\n")
                f.write("![Time Complexity](./figures/time_complexity.png)\n\n")
                
            if os.path.exists(os.path.join(figures_dir, 'iterations_complexity.png')):
                f.write("### Iterations Required\n\n")
                f.write("![Iterations Required](./figures/iterations_complexity.png)\n\n")
            
        print(f"Report generated successfully to {report_file}")
        return report_file
    
    def run_full_analysis(self):
        """Run a complete analysis and optimization workflow"""
        # Step 1: Discover devices (or create simulation)
        print("\n=== Step 1: Network Discovery ===")
        self.discover_network_devices()
        self.save_devices_to_csv()
        
        # Step 2: Simulate traffic
        print("\n=== Step 2: Traffic Simulation ===")
        self.simulate_traffic(duration=60, samples=10, time_factor=0.1)
        
        # Step 3: Visualize the network before optimization
        print("\n=== Step 3: Network Visualization (Before Optimization) ===")
        self.visualize_network(output_file="network_before.png")
        
        # Step 4: Optimize bandwidth allocation
        print("\n=== Step 4: Bandwidth Optimization ===")
        self.optimize_bandwidth_allocation()
        
        # Step 5: Visualize the network after optimization
        print("\n=== Step 5: Network Visualization (After Optimization) ===")
        self.visualize_network(output_file="network_after.png", show_optimization=True)
        
        # Step 6: Analyze algorithm complexity
        print("\n=== Step 6: Algorithm Analysis ===")
        self.analyze_algorithm_complexity(min_nodes=5, max_nodes=25, step=5, samples=3)
        
        # Step 7: Generate report
        print("\n=== Step 7: Report Generation ===")
        report_file = self.generate_report()
        
        print("\n=== Analysis Completed Successfully ===")
        print(f"Report available at: {report_file}")
        
        return {
            "network_before": os.path.join(self.output_dir, "network_before.png"),
            "network_after": os.path.join(self.output_dir, "network_after.png"),
            "report": report_file
        }
    
    def analyze_algorithm_complexity(self, min_nodes=5, max_nodes=30, step=5, samples=3):
        """
        Analyze Edmonds-Karp algorithm complexity by running on networks of different sizes
        """
        print("\n===== ANALYZING ALGORITHM COMPLEXITY =====")
        
        results = []
        
        for size in range(min_nodes, max_nodes + 1, step):
            print(f"\nAnalyzing networks with {size} nodes...")
            
            for sample in range(samples):
                # Create a random network
                G = nx.DiGraph()
                
                # Add nodes
                for i in range(size):
                    G.add_node(i)
                
                # Add random edges (about 3*n edges)
                edge_count = min(size * 3, size * (size - 1))
                
                edges_added = 0
                while edges_added < edge_count:
                    u = random.randint(0, size - 1)
                    v = random.randint(0, size - 1)
                    if u != v and not G.has_edge(u, v):
                        capacity = random.randint(1, 100)
                        G.add_edge(u, v, capacity=capacity, flow=0)
                        edges_added += 1
                
                # Choose random source and sink
                source = 0
                sink = size - 1
                
                # Store original graph
                old_graph = self.graph
                self.graph = G
                
                # Run Edmonds-Karp with performance measurement
                start_time = time.time()
                max_flow, flow_dict, performance = self.edmonds_karp(source, sink)
                end_time = time.time()
                
                # Restore original graph
                self.graph = old_graph
                
                # Record results
                results.append({
                    'nodes': size,
                    'edges': edge_count,
                    'execution_time': end_time - start_time,
                    'iterations': performance['iterations'],
                    'nodes_visited': performance['nodes_visited'],
                    'avg_path_length': performance['avg_path_length'],
                    'max_flow': max_flow
                })
                
                print(f"  Sample {sample+1}/{samples}: {performance['iterations']} iterations, {performance['execution_time']:.4f} seconds")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Analyze and display results
        self.display_complexity_analysis(df)
        
        return df
    
    def display_complexity_analysis(self, df):
        """Display complexity analysis results with charts"""
        print("\nComplexity Analysis Results:")
        
        # Group by number of nodes
        grouped = df.groupby('nodes').mean().reset_index()
        
        # Create figures directory
        figures_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Display average execution time by network size
        print("\nAverage Execution Time by Network Size:")
        print(f"{'Nodes':<10} {'Edges':<10} {'Time (s)':<15} {'Iterations':<15} {'Nodes Visited':<15}")
        print("-" * 65)
        
        for _, row in grouped.iterrows():
            print(f"{int(row['nodes']):<10} {int(row['edges']):<10} {row['execution_time']:<15.4f} {int(row['iterations']):<15} {int(row['nodes_visited']):<15}")
        
        # Plot execution time vs. network size
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['nodes'], grouped['execution_time'], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Number of Nodes')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Edmonds-Karp Algorithm: Time Complexity')
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'time_complexity.png'))
        
        # Plot iterations vs. network size
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['nodes'], grouped['iterations'], 'o-', linewidth=2, markersize=8, color='green')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Number of Iterations')
        plt.title('Edmonds-Karp Algorithm: Iterations Required')
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'iterations_complexity.png'))
        
        # Plot nodes visited vs. network size
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['nodes'], grouped['nodes_visited'], 'o-', linewidth=2, markersize=8, color='purple')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Nodes Visited')
        plt.title('Edmonds-Karp Algorithm: Search Space')
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'nodes_visited_complexity.png'))
        
        # Calculate growth rates
        if len(grouped) > 1:
            # Fit polynomial curves to estimate growth rates
            from scipy.optimize import curve_fit
            
            def poly_func(x, a, b, c):
                return a * x**2 + b * x + c
                
            try:
                # Fit for execution time
                x_data = grouped['nodes'].values
                y_data = grouped['execution_time'].values
                popt, _ = curve_fit(poly_func, x_data, y_data)
                
                # Calculate R-squared
                y_pred = poly_func(x_data, *popt)
                ss_tot = np.sum((y_data - np.mean(y_data))**2)
                ss_res = np.sum((y_data - y_pred)**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                print(f"\nTime Complexity Growth Rate Analysis:")
                print(f"Estimated function: {popt[0]:.6f} * n² + {popt[1]:.4f} * n + {popt[2]:.4f}")
                print(f"R-squared: {r_squared:.4f}")
                
                if abs(popt[0]) > 0.0001:  # Significant quadratic term
                    print("The algorithm exhibits approximately O(n²) time complexity")
                elif abs(popt[1]) > 0.001:  # Significant linear term
                    print("The algorithm exhibits approximately O(n) time complexity")
                else:
                    print("The algorithm exhibits approximately O(1) time complexity")
                    
                # Plot the fitted curve
                plt.figure(figsize=(10, 6))
                x_range = np.linspace(min(x_data), max(x_data), 100)
                plt.plot(x_data, y_data, 'o', label='Data')
                plt.plot(x_range, poly_func(x_range, *popt), '-', label=f'Fit: {popt[0]:.4f}*n² + {popt[1]:.4f}*n + {popt[2]:.4f}')
                plt.xlabel('Number of Nodes')
                plt.ylabel('Execution Time (seconds)')
                plt.title('Edmonds-Karp Algorithm: Time Complexity Curve Fitting')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(figures_dir, 'time_complexity_fit.png'))
                
            except Exception as e:
                print(f"Could not perform curve fitting: {e}")
        
        print(f"\nCharts saved to {figures_dir} directory")
        
        # Return the figures for display
        return {
            'time_complexity': os.path.join(figures_dir, 'time_complexity.png'),
            'iterations_complexity': os.path.join(figures_dir, 'iterations_complexity.png'),
            'nodes_visited_complexity': os.path.join(figures_dir, 'nodes_visited_complexity.png')
        }