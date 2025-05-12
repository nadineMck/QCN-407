import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class QKDDemand:
    """Represents a quantum key distribution demand between nodes"""
    source: str
    destination: str
    priority: float  # 0-1, higher is more important
    min_rate: float  # Minimum required key rate (Mbps)
    min_fidelity: float  # Minimum required fidelity

class FullAUBQuantumNetworkMINLP:
    def __init__(self):
        # Network topology
        self.nodes = ['Bechtel', 'Physics', 'Oxy', 'College Hall', 'Van Dyck', 'Medical Center']
        self.node_indices = {node: i for i, node in enumerate(self.nodes)}
        
        # All possible links (not just star topology)
        self.links = [
            ('Bechtel', 'Physics'),
            ('Bechtel', 'Oxy'),
            ('Bechtel', 'College Hall'),
            ('Bechtel', 'Van Dyck'),
            ('Bechtel', 'Medical Center'),
            ('Van Dyck', 'College Hall')  # Backup link
            # Additional potential links for full mesh consideration
            #('Physics', 'Oxy'),
            #('College Hall', 'Medical Center'),
            #('Van Dyck', 'Medical Center')
        ]
        
        self.link_indices = {link: i for i, link in enumerate(self.links)}
        
        # Distances in km (calculate for new links)
        self.distances = {
            ('Bechtel', 'Physics'): 0.2186,
            ('Bechtel', 'Oxy'): 0.0606,
            ('Bechtel', 'College Hall'): 0.1369,
            ('Bechtel', 'Van Dyck'): 0.2723,
            ('Bechtel', 'Medical Center'): 0.4681,
            ('Van Dyck', 'College Hall'): 0.2129
            #uncomment the lines to get a full mesh:
            #('Physics', 'Oxy'): 0.1800,  # Estimated
            #('College Hall', 'Medical Center'): 0.3500,  # Estimated
            #('Van Dyck', 'Medical Center'): 0.2900  # Estimated
        }
        
        # Physical parameters
        self.alpha = 0.2  # dB/km fiber attenuation
        self.eta_det = 0.9  # Detector efficiency
        self.f_min_default = 0.85  # Default minimum fidelity
        
        # Cost parameters
        self.c_r = 1.0    # Cost per MHz resource allocation
        self.c_p = 0.5    # Cost per Watt power
        self.c_m = 2.0    # Cost per qubit memory
        self.c_link = 10.0  # Fixed cost per active link
        
        # Bell state parameters
        self.lambda_1 = 0.98  # |Φ+⟩ coefficient
        self.lambda_2 = 0.01  # |Φ-⟩ coefficient  
        self.lambda_3 = 0.005 # |Ψ+⟩ coefficient
        self.lambda_4 = 0.005 # |Ψ-⟩ coefficient
        
        # Power-resource relationship parameters
        self.alpha_p = 0.1  # Conversion efficiency
        self.beta = 2.0    # Power exponent
        
        # Define QKD demands
        self.demands = [
            QKDDemand('Bechtel', 'College Hall', 1.0, 2.0, 0.90),  # High priority
            QKDDemand('Bechtel', 'Medical Center', 0.9, 1.5, 0.87),  # High priority
            QKDDemand('Physics', 'Oxy', 0.5, 0.5, 0.85),  # Medium priority
            QKDDemand('Van Dyck', 'Medical Center', 0.7, 1.0, 0.86),  # Medium priority
            QKDDemand('College Hall', 'Medical Center', 0.6, 0.8, 0.85),  # Medium priority
        ]
    
    def fidelity_after_transmission(self, distance):
        """Calculate fidelity after transmission through fiber"""
        F = np.exp(-self.alpha * distance / 4.343)
        # Fidelity with respect to |Φ+⟩ state
        f = F * self.lambda_1 + (1 - F) / 4
        return f
    
    def quantum_bit_error_rate(self, fidelity):
        """Calculate QBER from fidelity"""
        return 0.5 * (1 - fidelity)
    
    def privacy_amplification_factor(self, qber):
        """Calculate privacy amplification factor h(e)"""
        if qber >= 0.11:  # Above threshold for BB84
            return 0
        if qber <= 0:
            return 1
        # Binary entropy function
        h_e = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber)
        return 1 - h_e
    
    def key_rate(self, distance, resource_allocation, power=None):
        """Calculate quantum key generation rate"""
        f = self.fidelity_after_transmission(distance)
        qber = self.quantum_bit_error_rate(f)
        h_e = self.privacy_amplification_factor(qber)
        
        # Power-dependent resource if power is specified
        if power is not None:
            resource_allocation = self.alpha_p * (power ** self.beta)
        
        # Key rate in Mbps
        return resource_allocation * self.eta_det * (f ** 2) * h_e
    
    def unpack_variables(self, x):
        """Unpack the optimization variable vector"""
        n_links = len(self.links)
        n_nodes = len(self.nodes)
        n_demands = len(self.demands)
        
        idx = 0
        # Resource allocations r_ij
        r = x[idx:idx+n_links]
        idx += n_links
        
        # Power allocations p_ij
        p = x[idx:idx+n_links]
        idx += n_links
        
        # Memory allocations m_i
        m = x[idx:idx+n_nodes]
        idx += n_nodes
        
        # Binary routing variables x_ij^k (relaxed to continuous)
        x_routing = x[idx:].reshape(n_demands, n_links)
        
        return r, p, m, x_routing
    
    def objective_function(self, x):
        """Complete objective function with routing"""
        r, p, m, x_routing = self.unpack_variables(x)
        
        # Calculate utility
        utility = 0
        for k, demand in enumerate(self.demands):
            path_utility = 0
            path_fidelity = 1  # Multiplicative along path
            
            for i, link in enumerate(self.links):
                if x_routing[k, i] > 0.01:  # If link is used
                    distance = self.distances[link]
                    f = self.fidelity_after_transmission(distance)
                    q = self.key_rate(distance, r[i])
                    
                    path_fidelity *= f
                    path_utility += x_routing[k, i] * q
            
            # Weight by priority and fidelity
            utility += demand.priority * path_fidelity * path_utility
        
        # Calculate cost
        resource_cost = np.sum(self.c_r * r)
        power_cost = np.sum(self.c_p * p)
        memory_cost = np.sum(self.c_m * m)
        
        # Add fixed cost for active links
        link_usage = np.sum(x_routing, axis=0) > 0.01
        link_cost = np.sum(link_usage * self.c_link)
        
        total_cost = resource_cost + power_cost + memory_cost + link_cost
        
        # Scalarized objective
        alpha_utility = 0.6
        alpha_cost = 0.4
        
        return -alpha_utility * utility + alpha_cost * total_cost
    
    def setup_constraints(self):
        """Setup all constraints for the optimization"""
        n_links = len(self.links)
        n_nodes = len(self.nodes)
        n_demands = len(self.demands)
        n_vars = n_links * 2 + n_nodes + n_demands * n_links
        
        constraints = []
        
        # 1. Fidelity constraints for each link
        for i, link in enumerate(self.links):
            distance = self.distances[link]
            f_min = self.fidelity_after_transmission(distance)
            
            if f_min < self.f_min_default:
                print(f"Warning: Link {link} cannot meet minimum fidelity {self.f_min_default:.3f} (max: {f_min:.3f})")
        
        # 2. Power-resource relationship constraints
        def power_resource_constraint(x):
            r, p, m, x_routing = self.unpack_variables(x)
            return r - self.alpha_p * (p ** self.beta)
        
        constraints.append({
            'type': 'eq',
            'fun': power_resource_constraint
        })
        
        # 3. Flow conservation constraints for each demand
        for k, demand in enumerate(self.demands):
            def flow_conservation(x, k=k, demand=demand):
                r, p, m, x_routing = self.unpack_variables(x)
                flow = np.zeros(n_nodes)
                
                for i, link in enumerate(self.links):
                    source_idx = self.node_indices[link[0]]
                    dest_idx = self.node_indices[link[1]]
                    
                    # Flow out of source
                    flow[source_idx] -= x_routing[k, i]
                    # Flow into destination
                    flow[dest_idx] += x_routing[k, i]
                
                # Set source/sink values
                source_idx = self.node_indices[demand.source]
                dest_idx = self.node_indices[demand.destination]
                
                flow[source_idx] += 1  # Source produces 1 unit
                flow[dest_idx] -= 1    # Sink consumes 1 unit
                
                return flow
            
            constraints.append({
                'type': 'eq',
                'fun': flow_conservation
            })
        
        # 4. Memory constraints at each node
        def memory_constraint(x):
            r, p, m, x_routing = self.unpack_variables(x)
            memory_usage = np.zeros(n_nodes)
            
            for i, link in enumerate(self.links):
                source_idx = self.node_indices[link[0]]
                dest_idx = self.node_indices[link[1]]
                
                # Memory usage proportional to key rate and routing
                q = self.key_rate(self.distances[link], r[i])
                total_routing = np.sum(x_routing[:, i])
                
                memory_usage[source_idx] += 0.1 * q * total_routing  # Storage time factor
                memory_usage[dest_idx] += 0.1 * q * total_routing
            
            return m - memory_usage
        
        constraints.append({
            'type': 'ineq',
            'fun': memory_constraint
        })
        
        # 5. Demand satisfaction constraints
        for k, demand in enumerate(self.demands):
            def rate_constraint(x, k=k, demand=demand):
                r, p, m, x_routing = self.unpack_variables(x)
                total_rate = 0
                
                for i, link in enumerate(self.links):
                    if x_routing[k, i] > 0.01:
                        distance = self.distances[link]
                        q = self.key_rate(distance, r[i])
                        total_rate += x_routing[k, i] * q
                
                return total_rate - demand.min_rate
            
            constraints.append({
                'type': 'ineq',
                'fun': rate_constraint
            })
        
        return constraints
    
    def solve_minlp(self):
        """Solve the complete MINLP problem with relaxation and rounding"""
        n_links = len(self.links)
        n_nodes = len(self.nodes)
        n_demands = len(self.demands)
        
        # Step 1: Continuous Relaxation
        print("Step 1: Solving continuous relaxation...")
        
        # Initial guess
        x0 = np.concatenate([
            np.ones(n_links) * 2,      # r_ij
            np.ones(n_links) * 5,      # p_ij
            np.ones(n_nodes) * 20,     # m_i
            np.ones(n_demands * n_links) * 0.5  # x_ij^k relaxed
        ])
        
        # Bounds
        bounds = []
        # Resource allocation bounds
        for i in range(n_links):
            bounds.append((0.1, 10))
        # Power bounds
        for i in range(n_links):
            bounds.append((1, 20))
        # Memory bounds
        for i in range(n_nodes):
            bounds.append((5, 100))
        # Routing variables bounds (relaxed to [0,1])
        for i in range(n_demands * n_links):
            bounds.append((0, 1))
        
        # Get constraints
        constraints = self.setup_constraints()
        
        # Solve relaxed problem
        relaxed_result = minimize(
            self.objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 1000}
        )
        
        print(f"Relaxed solution status: {relaxed_result.message}")
        print(f"Relaxed objective value: {relaxed_result.fun:.4f}")
        
        # Step 2: Intelligent Rounding
        print("\nStep 2: Applying intelligent rounding...")
        r_rel, p_rel, m_rel, x_routing_rel = self.unpack_variables(relaxed_result.x)
        
        # Round routing variables based on priority
        x_routing_binary = np.zeros_like(x_routing_rel)
        
        for k, demand in enumerate(self.demands):
            # Find best path based on relaxed solution
            path_scores = x_routing_rel[k, :] * demand.priority
            
            # Select links with highest scores until flow is satisfied
            sorted_links = np.argsort(path_scores)[::-1]
            
            for link_idx in sorted_links:
                if x_routing_rel[k, link_idx] > 0.3:  # Threshold for selection
                    x_routing_binary[k, link_idx] = 1
                    
                    # Check if path is complete (simplified)
                    if self.check_path_complete(k, x_routing_binary[k, :]):
                        break
        
        # Step 3: Local Refinement
        print("\nStep 3: Local refinement with fixed routing...")
        
        # Create new objective with fixed routing
        def fixed_routing_objective(x_continuous):
            x_full = np.concatenate([x_continuous, x_routing_binary.flatten()])
            return self.objective_function(x_full)
        
        # Optimize continuous variables with fixed routing
        x0_continuous = relaxed_result.x[:n_links*2 + n_nodes]
        bounds_continuous = bounds[:n_links*2 + n_nodes]
        
        final_result = minimize(
            fixed_routing_objective,
            x0_continuous,
            method='SLSQP',
            bounds=bounds_continuous,
            options={'disp': True}
        )
        
        print(f"Final solution status: {final_result.message}")
        print(f"Final objective value: {fixed_routing_objective(final_result.x):.4f}")
        
        # Combine results
        final_x = np.concatenate([final_result.x, x_routing_binary.flatten()])
        
        return final_x, relaxed_result, final_result
    
    def check_path_complete(self, demand_idx, routing_vector):
        """Check if routing vector forms complete path for demand"""
        demand = self.demands[demand_idx]
        
        # Simple check: at least one outgoing from source and incoming to destination
        source_has_outgoing = False
        dest_has_incoming = False
        
        for i, link in enumerate(self.links):
            if routing_vector[i] > 0.5:
                if link[0] == demand.source:
                    source_has_outgoing = True
                if link[1] == demand.destination:
                    dest_has_incoming = True
        
        return source_has_outgoing and dest_has_incoming
    
    def visualize_complete_solution(self, final_x):
        """Create comprehensive visualization of the solution"""
        r, p, m, x_routing = self.unpack_variables(final_x)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Network topology with routing
        ax1 = plt.subplot(2, 3, 1)
        self.plot_network_topology(x_routing, r, ax1)
        
        # 2. Resource allocation by link
        ax2 = plt.subplot(2, 3, 2)
        link_names = [f"{l[0][:3]}-{l[1][:3]}" for l in self.links]
        bars = ax2.bar(link_names, r, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Link')
        ax2.set_ylabel('Resource Allocation (MHz)')
        ax2.set_title('Resource Allocation by Link')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, r):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom')
        
        # 3. Fidelity and key rates
        ax3 = plt.subplot(2, 3, 3)
        fidelities = [self.fidelity_after_transmission(self.distances[l]) for l in self.links]
        key_rates = [self.key_rate(self.distances[l], r[i]) for i, l in enumerate(self.links)]
        
        x_pos = np.arange(len(link_names))
        ax3.bar(x_pos - 0.2, fidelities, 0.4, label='Fidelity', color='green', alpha=0.7)
        ax3.bar(x_pos + 0.2, np.array(key_rates)/max(key_rates), 0.4, 
                label='Normalized Key Rate', color='orange', alpha=0.7)
        ax3.set_xlabel('Link')
        ax3.set_ylabel('Value')
        ax3.set_title('Fidelity and Key Rates by Link')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(link_names, rotation=45)
        ax3.legend()
        ax3.axhline(y=self.f_min_default, color='r', linestyle='--', 
                   label=f'Min Fidelity = {self.f_min_default}')
        
        # 4. Power consumption
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(p, r, s=100, alpha=0.6, c=range(len(p)), cmap='viridis')
        ax4.set_xlabel('Power (W)')
        ax4.set_ylabel('Resource Allocation (MHz)')
        ax4.set_title('Power vs Resource Allocation')
        
        # Add power curve
        p_range = np.linspace(min(p), max(p), 100)
        r_theoretical = self.alpha_p * (p_range ** self.beta)
        ax4.plot(p_range, r_theoretical, 'r--', label='Theoretical: r = αp²')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Memory allocation
        ax5 = plt.subplot(2, 3, 5)
        # Use a single, valid name or list of names here:
        bars = ax5.bar(self.nodes, m, color='mediumpurple', edgecolor='indigo')
        ax5.set_xlabel('Node')
        ax5.set_ylabel('Memory Allocation (qubits)')
        ax5.set_title('Quantum Memory Allocation by Node')
        ax5.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, val in zip(bars, m):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom')

        
        # 6. Demand satisfaction
        ax6 = plt.subplot(2, 3, 6)
        demand_labels = [f"D{i}: {d.source[:3]}→{d.destination[:3]}" 
                        for i, d in enumerate(self.demands)]
        achieved_rates = []
        required_rates = []
        
        for k, demand in enumerate(self.demands):
            total_rate = 0
            for i, link in enumerate(self.links):
                if x_routing[k, i] > 0.5:
                    distance = self.distances[link]
                    q = self.key_rate(distance, r[i])
                    total_rate += q
            achieved_rates.append(total_rate)
            required_rates.append(demand.min_rate)
        
        x_pos = np.arange(len(demand_labels))
        ax6.bar(x_pos - 0.2, required_rates, 0.4, label='Required', color='red', alpha=0.7)
        ax6.bar(x_pos + 0.2, achieved_rates, 0.4, label='Achieved', color='green', alpha=0.7)
        ax6.set_xlabel('Demand')
        ax6.set_ylabel('Key Rate (Mbps)')
        ax6.set_title('Demand Satisfaction')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(demand_labels, rotation=45)
        ax6.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Cost breakdown pie chart
        fig2, ax = plt.subplots(figsize=(10, 8))
        resource_cost = np.sum(self.c_r * r)
        power_cost = np.sum(self.c_p * p)
        memory_cost = np.sum(self.c_m * m)
        link_usage = np.sum(x_routing, axis=0) > 0.5
        link_cost = np.sum(link_usage * self.c_link)
        
        costs = [resource_cost, power_cost, memory_cost, link_cost]
        labels = ['Resource', 'Power', 'Memory', 'Link Setup']
        colors = ['gold', 'lightcoral', 'lightblue', 'lightgreen']
        
        wedges, texts, autotexts = ax.pie(costs, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Cost Breakdown (Total: ${sum(costs):.2f})')
        
        plt.show()
        
        # Routing visualization matrix
        fig3, ax = plt.subplots(figsize=(12, 8))
        
        # Create routing matrix visualization
        routing_matrix = x_routing.copy()
        routing_matrix[routing_matrix < 0.5] = 0
        routing_matrix[routing_matrix >= 0.5] = 1
        
        sns.heatmap(routing_matrix, 
                   xticklabels=[f"{l[0][:3]}-{l[1][:3]}" for l in self.links],
                   yticklabels=[f"D{i}: {d.source[:3]}→{d.destination[:3]}" 
                               for i, d in enumerate(self.demands)],
                   cmap='Blues', cbar_kws={'label': 'Link Usage'},
                   annot=True, fmt='.0f', ax=ax)
        ax.set_title('Routing Decision Matrix')
        ax.set_xlabel('Links')
        ax.set_ylabel('Demands')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_network_topology(self, x_routing, resources, ax):
        """Plot network topology with active routes"""
        # Node positions (simplified 2D layout)
        pos = {
            'Bechtel': (0, 0),
            'Physics': (-1, 1),
            'Oxy': (-1, -1),
            'College Hall': (1, 1),
            'Van Dyck': (1, -1),
            'Medical Center': (2, 0)
        }
        
        # Draw nodes
        for node, (x, y) in pos.items():
            ax.scatter(x, y, s=300, c='lightblue', edgecolors='navy', linewidth=2, zorder=3)
            ax.text(x, y-0.3, node[:3], ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Draw links with varying thickness based on resource allocation
        for i, link in enumerate(self.links):
            start = pos[link[0]]
            end = pos[link[1]]
            
            # Check if link is actively used
            is_active = np.sum(x_routing[:, i]) > 0.5
            
            if is_active:
                width = 1 + resources[i] / 2  # Width proportional to resource
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'g-', linewidth=width, alpha=0.7)
                
                # Add resource label
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                ax.text(mid_x, mid_y, f'{resources[i]:.1f}', 
                       ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            else:
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'lightgray', linewidth=0.5, linestyle='--', alpha=0.5)
        
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title('Active Network Topology')
        ax.axis('off')
    
    def run_time_analysis(self, hours=24):
        """Analyze network performance over time"""
        print("\nRunning time-based analysis...")
        
        time_points = np.linspace(0, hours, 48)
        results_over_time = []
        costs_over_time = []
        
        for t in time_points:
            # Modify demands based on time of day
            time_multiplier = 1.0
            if 9 <= t % 24 <= 17:  # Business hours
                time_multiplier = 1.5 + 0.3 * np.sin(2 * np.pi * t / 24)
            else:
                time_multiplier = 0.7 + 0.2 * np.sin(2 * np.pi * t / 24)
            
            # Scale minimum rates
            original_rates = [d.min_rate for d in self.demands]
            for i, demand in enumerate(self.demands):
                demand.min_rate = original_rates[i] * time_multiplier
            
            # Solve for this time point
            try:
                final_x, _, _ = self.solve_minlp()
                r, p, m, x_routing = self.unpack_variables(final_x)
                
                # Calculate metrics
                total_rate = 0
                avg_fidelity = 0
                active_links = 0
                
                for i, link in enumerate(self.links):
                    if np.sum(x_routing[:, i]) > 0.5:
                        distance = self.distances[link]
                        total_rate += self.key_rate(distance, r[i])
                        avg_fidelity += self.fidelity_after_transmission(distance)
                        active_links += 1
                
                avg_fidelity /= active_links if active_links > 0 else 1
                
                # Calculate cost
                total_cost = (np.sum(self.c_r * r) + np.sum(self.c_p * p) + 
                             np.sum(self.c_m * m) + active_links * self.c_link)
                
                results_over_time.append({
                    'time': t,
                    'total_rate': total_rate,
                    'avg_fidelity': avg_fidelity,
                    'active_links': active_links
                })
                
                costs_over_time.append(total_cost)
                
            except Exception as e:
                print(f"Failed to solve at time {t:.1f}: {e}")
                continue
            
            # Restore original rates
            for i, demand in enumerate(self.demands):
                demand.min_rate = original_rates[i]
        
        # Plot time analysis
        results_df = pd.DataFrame(results_over_time)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total key rate over time
        ax1.plot(results_df['time'], results_df['total_rate'], 'b-', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Total Key Rate (Mbps)')
        ax1.set_title('Network Key Generation Rate Over 24 Hours')
        ax1.grid(True, alpha=0.3)
        ax1.axvspan(9, 17, alpha=0.2, color='yellow', label='Business Hours')
        ax1.legend()
        
        # Average fidelity over time
        ax2.plot(results_df['time'], results_df['avg_fidelity'], 'g-', linewidth=2)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Average Fidelity')
        ax2.set_title('Network Average Fidelity Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=self.f_min_default, color='r', linestyle='--', 
                   label=f'Min Required = {self.f_min_default}')
        ax2.axvspan(9, 17, alpha=0.2, color='yellow')
        ax2.legend()
        
        # Cost over time
        ax3.plot(results_df['time'], costs_over_time, 'r-', linewidth=2)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Total Cost ($)')
        ax3.set_title('Operational Cost Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.axvspan(9, 17, alpha=0.2, color='yellow')
        
        # Active links over time
        ax4.plot(results_df['time'], results_df['active_links'], 'purple', 
                linewidth=2, marker='o')
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Number of Active Links')
        ax4.set_title('Active Links Over Time')
        ax4.grid(True, alpha=0.3)
        ax4.axvspan(9, 17, alpha=0.2, color='yellow')
        ax4.set_ylim(0, len(self.links))
        
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def performance_3d_analysis(self, final_x):
        """3D visualization of performance metrics"""
        r, p, m, x_routing = self.unpack_variables(final_x)
        
        fig = plt.figure(figsize=(15, 12))
        
        # 3D scatter plot of resource, power, and key rate
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        key_rates = []
        for i, link in enumerate(self.links):
            distance = self.distances[link]
            q = self.key_rate(distance, r[i])
            key_rates.append(q)
        
        # Color by link utilization
        utilization = np.sum(x_routing, axis=0)
        
        scatter = ax1.scatter(r, p, key_rates, c=utilization, cmap='viridis', 
                             s=100, alpha=0.6, edgecolors='black')
        ax1.set_xlabel('Resource Allocation (MHz)')
        ax1.set_ylabel('Power (W)')
        ax1.set_zlabel('Key Rate (Mbps)')
        ax1.set_title('3D Performance Space')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1, pad=0.1)
        cbar.set_label('Link Utilization')
        
        # 3D surface for theoretical relationship
        r_range = np.linspace(0.1, 10, 20)
        p_range = np.linspace(1, 20, 20)
        R, P = np.meshgrid(r_range, p_range)
        
        # Average distance for surface
        avg_distance = np.mean(list(self.distances.values()))
        Q_theoretical = self.key_rate(avg_distance, R)
        
        ax1.plot_surface(R, P, Q_theoretical, alpha=0.3, cmap='coolwarm')
        
        # 2D projections
        # Resource vs Key Rate
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(r, key_rates, c=utilization, cmap='viridis', s=100, alpha=0.6)
        ax2.set_xlabel('Resource Allocation (MHz)')
        ax2.set_ylabel('Key Rate (Mbps)')
        ax2.set_title('Resource Allocation vs Key Rate')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(r, key_rates, 2)
        p_fit = np.poly1d(z)
        r_smooth = np.linspace(min(r), max(r), 100)
        ax2.plot(r_smooth, p_fit(r_smooth), "r--", alpha=0.8, linewidth=2)
        
        # Power vs Key Rate
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(p, key_rates, c=utilization, cmap='viridis', s=100, alpha=0.6)
        ax3.set_xlabel('Power (W)')
        ax3.set_ylabel('Key Rate (Mbps)')
        ax3.set_title('Power vs Key Rate')
        ax3.grid(True, alpha=0.3)
        
        # Efficiency plot
        ax4 = fig.add_subplot(2, 2, 4)
        efficiency = np.array(key_rates) / (r * p)  # Key rate per unit resource-power
        ax4.bar(range(len(self.links)), efficiency, color='orange', alpha=0.7)
        ax4.set_xlabel('Link Index')
        ax4.set_ylabel('Efficiency (Mbps/MHz·W)')
        ax4.set_title('Link Efficiency')
        ax4.set_xticks(range(len(self.links)))
        ax4.set_xticklabels([f"{l[0][:3]}-{l[1][:3]}" for l in self.links], 
                           rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self, final_x):
        """Generate detailed report of optimization results"""
        r, p, m, x_routing = self.unpack_variables(final_x)
        
        report = []
        report.append("=" * 60)
        report.append("AUB QUANTUM NETWORK OPTIMIZATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        
        total_cost = (np.sum(self.c_r * r) + np.sum(self.c_p * p) + 
                     np.sum(self.c_m * m) + np.sum(x_routing > 0.5) * self.c_link)
        total_key_rate = sum(self.key_rate(self.distances[l], r[i]) 
                           for i, l in enumerate(self.links) 
                           if np.sum(x_routing[:, i]) > 0.5)
        avg_fidelity = np.mean([self.fidelity_after_transmission(self.distances[l]) 
                               for i, l in enumerate(self.links) 
                               if np.sum(x_routing[:, i]) > 0.5])
        
        report.append(f"Total Network Cost: ${total_cost:.2f}")
        report.append(f"Total Key Generation Rate: {total_key_rate:.2f} Mbps")
        report.append(f"Average Link Fidelity: {avg_fidelity:.4f}")
        report.append(f"Active Links: {sum(np.sum(x_routing, axis=0) > 0.5)}/{len(self.links)}")
        report.append("")
        
        # Link Details
        report.append("LINK CONFIGURATION DETAILS")
        report.append("-" * 30)
        report.append(f"{'Link':<20} {'Resource':<10} {'Power':<10} {'Fidelity':<10} {'Key Rate':<10} {'Active':<8}")
        report.append("-" * 78)
        
        for i, link in enumerate(self.links):
            link_name = f"{link[0][:8]}-{link[1][:8]}"
            distance = self.distances[link]
            fidelity = self.fidelity_after_transmission(distance)
            key_rate = self.key_rate(distance, r[i])
            is_active = np.sum(x_routing[:, i]) > 0.5
            
            report.append(f"{link_name:<20} {r[i]:<10.2f} {p[i]:<10.2f} "
                         f"{fidelity:<10.4f} {key_rate:<10.2f} {'Yes' if is_active else 'No':<8}")
        
        report.append("")
        
        # Node Memory Allocation
        report.append("NODE MEMORY ALLOCATION")
        report.append("-" * 25)
        report.append(f"{'Node':<15} {'Memory (qubits)':<15}")
        report.append("-" * 30)
        
        for i, node in enumerate(self.nodes):
            report.append(f"{node:<15} {m[i]:<15.1f}")
        
        report.append("")
        
        # Demand Routing
        report.append("DEMAND ROUTING AND SATISFACTION")
        report.append("-" * 35)
        
        for k, demand in enumerate(self.demands):
            report.append(f"\nDemand {k}: {demand.source} → {demand.destination}")
            report.append(f"Priority: {demand.priority}, Required Rate: {demand.min_rate} Mbps")
            report.append("Route:")
            
            route_links = []
            total_rate = 0
            
            for i, link in enumerate(self.links):
                if x_routing[k, i] > 0.5:
                    route_links.append(link)
                    distance = self.distances[link]
                    q = self.key_rate(distance, r[i])
                    total_rate += q
                    report.append(f"  - {link[0]} → {link[1]} (Rate: {q:.2f} Mbps)")
            
            report.append(f"Total Achieved Rate: {total_rate:.2f} Mbps")
            report.append(f"Satisfaction: {'YES' if total_rate >= demand.min_rate else 'NO'}")
        
        report.append("")
        
        # Cost Breakdown
        report.append("COST BREAKDOWN")
        report.append("-" * 15)
        resource_cost = np.sum(self.c_r * r)
        power_cost = np.sum(self.c_p * p)
        memory_cost = np.sum(self.c_m * m)
        link_cost = sum(np.sum(x_routing[:, i]) > 0.5 for i in range(len(self.links))) * self.c_link
        
        report.append(f"Resource Cost: ${resource_cost:.2f} ({resource_cost/total_cost*100:.1f}%)")
        report.append(f"Power Cost: ${power_cost:.2f} ({power_cost/total_cost*100:.1f}%)")
        report.append(f"Memory Cost: ${memory_cost:.2f} ({memory_cost/total_cost*100:.1f}%)")
        report.append(f"Link Setup Cost: ${link_cost:.2f} ({link_cost/total_cost*100:.1f}%)")
        report.append(f"Total Cost: ${total_cost:.2f}")
        
        report.append("")
        
        # Performance Metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 20)
        report.append(f"Network Efficiency: {total_key_rate/total_cost:.4f} Mbps/$")
        report.append(f"Average Link Utilization: {np.mean(np.sum(x_routing, axis=0)):.2f}")
        report.append(f"Minimum Link Fidelity: {min(self.fidelity_after_transmission(self.distances[l]) for l in self.links):.4f}")
        report.append(f"Maximum Key Rate per Link: {max(self.key_rate(self.distances[l], r[i]) for i, l in enumerate(self.links)):.2f} Mbps")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        # Print report
        for line in report:
            print(line)
        
        # Save report to file
        with open('aub_quantum_network_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        return report

# Main execution function
def main():
    print("AUB Quantum Network MINLP Optimization")
    print("=" * 40)
    
    # Create network instance
    network = FullAUBQuantumNetworkMINLP()
    
    # Solve the complete MINLP problem
    print("\nSolving complete MINLP problem...")
    final_x, relaxed_result, refined_result = network.solve_minlp()
    
    # Visualize complete solution
    print("\nGenerating comprehensive visualizations...")
    network.visualize_complete_solution(final_x)
    
    # 3D performance analysis
    print("\nPerforming 3D analysis...")
    network.performance_3d_analysis(final_x)
    
    # Time-based analysis
    print("\nRunning 24-hour simulation...")
    time_results = network.run_time_analysis(hours=24)
    
    # Generate comprehensive report
    print("\nGenerating final report...")
    report = network.generate_comprehensive_report(final_x)
    
    # Additional analysis: Pareto frontier
    print("\nComputing Pareto frontier...")
    alphas = np.linspace(0.1, 0.9, 9)
    pareto_costs = []
    pareto_utilities = []
    
    for alpha in alphas:
        # Temporarily modify objective weights
        def temp_objective(x):
            r, p, m, x_routing = network.unpack_variables(x)
            # Calculate utility and cost separately
            utility = 0
            for k, demand in enumerate(network.demands):
                for i, link in enumerate(network.links):
                    if x_routing[k, i] > 0.01:
                        distance = network.distances[link]
                        f = network.fidelity_after_transmission(distance)
                        q = network.key_rate(distance, r[i])
                        utility += demand.priority * f * q * x_routing[k, i]
            
            cost = (np.sum(network.c_r * r) + np.sum(network.c_p * p) + 
                   np.sum(network.c_m * m) + np.sum(x_routing > 0.5) * network.c_link)
            
            return -alpha * utility + (1-alpha) * cost
        
        # Solve with modified objective
        n_vars = len(network.links) * 2 + len(network.nodes) + len(network.demands) * len(network.links)
        x0 = np.random.rand(n_vars)
        bounds = [(0, 10)] * (len(network.links) * 2) + [(5, 100)] * len(network.nodes) + [(0, 1)] * (len(network.demands) * len(network.links))
        
        try:
            result = minimize(temp_objective, x0, method='SLSQP', bounds=bounds)
            r, p, m, x_routing = network.unpack_variables(result.x)
            
            # Calculate actual utility and cost
            utility = 0
            for k, demand in enumerate(network.demands):
                for i, link in enumerate(network.links):
                    if x_routing[k, i] > 0.01:
                        distance = network.distances[link]
                        f = network.fidelity_after_transmission(distance)
                        q = network.key_rate(distance, r[i])
                        utility += demand.priority * f * q * x_routing[k, i]
            
            cost = (np.sum(network.c_r * r) + np.sum(network.c_p * p) + 
                   np.sum(network.c_m * m) + np.sum(x_routing > 0.5) * network.c_link)
            
            pareto_costs.append(cost)
            pareto_utilities.append(utility)
            
        except Exception as e:
            print(f"Failed for alpha={alpha}: {e}")
            continue
    
    # Plot Pareto frontier
    plt.figure(figsize=(10, 8))
    plt.scatter(pareto_costs, pareto_utilities, s=100, c=alphas, cmap='coolwarm')
    plt.xlabel('Total Cost ($)')
    plt.ylabel('Network Utility')
    plt.title('Pareto Frontier: Cost vs Utility Trade-off')
    plt.colorbar(label='Alpha (Cost Weight)')
    plt.grid(True, alpha=0.3)
    
    # Mark the chosen solution
    r, p, m, x_routing = network.unpack_variables(final_x)
    chosen_cost = (np.sum(network.c_r * r) + np.sum(network.c_p * p) + 
                  np.sum(network.c_m * m) + sum(np.sum(x_routing[:, i]) > 0.5 
                  for i in range(len(network.links))) * network.c_link)
    
    chosen_utility = 0
    for k, demand in enumerate(network.demands):
        for i, link in enumerate(network.links):
            if x_routing[k, i] > 0.5:
                distance = network.distances[link]
                f = network.fidelity_after_transmission(distance)
                q = network.key_rate(distance, r[i])
                chosen_utility += demand.priority * f * q
    
    plt.scatter(chosen_cost, chosen_utility, s=200, c='red', marker='*', 
               label='Chosen Solution', edgecolors='black', linewidth=2)
    plt.legend()
    plt.show()
    
    print("\nOptimization complete!")
    print("Results saved to 'aub_quantum_network_report.txt'")
    
    return network, final_x, report

if __name__ == "__main__":
    network, solution, report = main()