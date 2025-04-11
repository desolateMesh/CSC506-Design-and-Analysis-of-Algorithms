# Network Flow Optimization Using Edmonds-Karp Algorithm

## CSC506-1: Analysis and Design of Algorithms

### Project Overview
This project focuses on simulating network traffic and optimizing bandwidth allocation using the Edmonds-Karp algorithm—a variant of the Ford-Fulkerson method leveraging breadth-first search (BFS) to efficiently calculate the maximum flow in a network. It demonstrates an applied understanding of graph theory, algorithm analysis, and practical network optimization.

### Objectives
- **Simulate network environments:** Discover devices, simulate network traffic, and create realistic scenarios for testing.
- **Optimize network flow:** Apply the Edmonds-Karp algorithm to maximize network bandwidth efficiency and allocation fairness.
- **Analyze algorithm performance:** Measure and report algorithm complexity through empirical testing across networks of varying sizes.
- **Provide comprehensive reporting:** Generate detailed reports summarizing network performance metrics, optimization results, and algorithm complexity analyses.

### Project Structure

- **Network Flow Simulator (`network_flow_simulator.py`):**
  - Core simulation engine for network traffic and flow optimization.
  - Includes functionalities for device discovery, traffic simulation, bandwidth optimization, and comprehensive reporting.

- **Simulation Runner (`run_network_flow_sim.py`):**
  - Command-line interface (CLI) to facilitate easy execution of simulations and analysis.
  - Supports customizable parameters such as network bandwidth, simulation duration, complexity analysis, and output options.

- **Database (`network_data.db`):**
  - SQLite database storing network topology, device information, and simulation results for analysis and reference.

- **Documentation:**
  - **Algorithm Flow:** Visual representation of the Edmonds-Karp algorithm logic.
  - **Simulation Flowchart:** Step-by-step workflow illustrating the overall process from network discovery to performance analysis.

### Key Features

#### Device Discovery and Traffic Simulation
- **ARP-based discovery simulation** generates realistic network devices categorized into servers, computers, IoT devices, and more.
- **Traffic patterns:** Simulated based on device types with configurable bandwidth demands to mimic realistic network conditions.

#### Edmonds-Karp Flow Optimization
- Creates a residual graph to calculate the maximum achievable flow efficiently.
- Performs optimization through iterative BFS to find augmenting paths.
- Updates flow allocations based on calculated maximum flows.

#### Algorithm Complexity Analysis
- Empirical performance analysis across multiple network sizes.
- Detailed metrics including execution time, iterations, nodes visited, and memory usage.
- Visual graphs depicting complexity and scalability.

#### Visualization and Reporting
- Generates clear network topology visualizations before and after optimization.
- Comprehensive Markdown-formatted reports summarizing network performance, device statistics, and optimization outcomes.

### Running the Simulation

Use the provided CLI tool for simulations:

```bash
python run_network_flow_sim.py --output-dir=output --bandwidth=1000 --simulation-duration=60 --simulation-samples=10 --full-analysis
```

**Options:**
- `--output-dir`: Directory to save output files.
- `--bandwidth`: Total available network bandwidth (in Mbps).
- `--simulation-duration`: Duration of traffic simulation.
- `--full-analysis`: Performs complete analysis including complexity and reporting.

### Requirements
- Python 3.x
- Libraries: `networkx`, `matplotlib`, `numpy`, `pandas`, `tqdm`

Install required packages via:
```bash
pip install networkx matplotlib numpy pandas tqdm
```

### Conclusion
This project exemplifies rigorous application of theoretical algorithmic principles to real-world network problems, demonstrating skills in algorithm design, complexity analysis, software engineering, and technical reporting.

