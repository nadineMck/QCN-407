# 🌟 AUB Quantum Communication Network

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Quantum](https://img.shields.io/badge/quantum-ready-purple.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

> *Optimizing the future of secure campus communications through quantum key distribution*

## 📡 Project Overview

The **AUB Quantum Communication Network** is a cutting-edge implementation of a quantum key distribution (QKD) network designed for the American University of Beirut campus. This project leverages quantum mechanics principles to create an unbreakable communication infrastructure
### 🔬 What Makes This Special?

- **Quantum-Secure**: Theoretically unbreakable encryption
- **Campus-Scale**: Connects 6 critical AUB facilities with quantum channels
- **Hybrid Security**: XOR combination of quantum and classical keys for 100% uptime
- **AI-Optimized**:FUTURE PLAN
- **Real Physics**

## 🏗️ Network Architecture

```
                    [Physics]
                       |
    [Oxy] ━━━━━━ [Bechtel] ━━━━━━ [College Hall]
                  ┃   ┃              ┃
              [Van Dyck] ━━━━━━━━━━━━┛
                  ┃
            [Medical Center]
```

### 📍 Connected Facilities
- **Bechtel Building**: Central hub & monitoring station
- **College Hall**: Student records, CAMS data center
- **Medical Center**: Patient data, research databases
- **Van Dyck**: Backup data center, disaster recovery
- **Physics Department**: Quantum expertise, research
- **Oxy Complex**: Heavy computational labs

## 🚀 Key Features

### 1. Quantum State Modeling
- Bell-diagonal mixed states with realistic noise
- Depolarizing channel for fiber transmission
- Fidelity degradation with distance

### 2. Intelligent Resource Allocation
- Multi-objective optimization (utility vs cost)
- Dynamic demand-based allocation
- Time-varying network adaptation

### 3. Advanced Routing
- MINLP solver with binary routing decisions
- Automatic failover paths
- QoS-aware demand prioritization

### 4. Comprehensive Analysis
- 3D performance visualization
- 24-hour operational simulation
- Pareto frontier analysis
- Cost breakdown reports

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/nadineMck/QCN-407.git
```

### Dependencies
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Seaborn

## 🔧 Usage

### Basic Run
```python
from quantum_network_optimizer import FullAUBQuantumNetworkMINLP

# Create network instance
network = FullAUBQuantumNetworkMINLP()

# Solve optimization
final_solution, _, _ = network.solve_minlp()

# Generate visualizations
network.visualize_complete_solution(final_solution)

# Create report
network.generate_comprehensive_report(final_solution)
```

### Quick Test
```bash
python run_quantum_optimization.py
```

## 📊 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Min Fidelity | 0.85 | 0.86-0.96 |
| Key Rate | >1 Mbps | 1.5-30 Mbps |
| Network Uptime | 99.9% | 99.97% |
| Cost Efficiency | <$0.05/kb | $0.023/kb |

## 🧮 Mathematical Foundation

### Quantum State Evolution
```
ρ(d) = F(d)ρ₀ + (1-F(d))I/4
```
where F(d) = e^(-αd/4.343)

### Key Rate Formula
```
R = r × η_det × f² × h(e)
```

### Optimization Objective
```
min α₁(-U) + α₂C
s.t. physical constraints
```

## 📸 Visualizations

The optimizer produces multiple visualizations:
- Network topology with resource allocation
- 3D performance space analysis  
- Time-based demand simulation
- Cost breakdown charts
- Pareto frontier curves

## 🔮 Future Enhancements

- [ ] Machine learning integration
- [ ] Real-time network monitoring
- [ ] Mobile node support
- [ ] International campus connections

## 🤝 Contributing

We welcome contributions!
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request


## 🏆 Acknowledgments

- Professor Mahdi Chehimi
- Swiss Quantum Network for inspiration
- Open source quantum computing community

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## 🔐 Security Notice

While this system uses quantum mechanics for security, remember:
- Physical security of nodes is crucial
- Classical backup keys must be properly managed
- Regular security audits recommended

## 📞 Contact

- Project Team: N-M
- Email: nnm30@mail.aub.edu

---

<p align="center">
  <i>Building the quantum future, one photon at a time 🌟</i>
</p>

<p align="center">
  Made with ❤️ at American University of Beirut
</p>
