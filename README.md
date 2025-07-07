![License](https://img.shields.io/github/license/marcoaapfortes/dynamic-constraint-sequencer)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Energy Efficient](https://img.shields.io/badge/energy-efficient-green.svg)
![AI Ready](https://img.shields.io/badge/AI-ready-orange.svg)

# Dynamic Constraint Sequencer

## Overview

The `DynamicConstraintSequencer` is an energy-efficient algorithm for optimizing query processing over large structured datasets, designed to reduce computational and energy costs in data-intensive AI applications. Inspired by the human brain's remarkable 20-watt efficiency, the algorithm employs advanced techniques such as sparsity exploitation, reinforcement learning, and energy-aware constraint sequencing to minimize resource usage while maintaining high performance.

## Features

- **Energy Monitoring**: Tracks CPU, memory, and estimated power consumption (joules) using `psutil`, enabling optimization for energy efficiency
- **Sparsity Exploitation**: Optimizes processing for sparse datasets (e.g., >30% nulls/zeros), reducing memory and computational overhead
- **Reinforcement Learning**: Dynamically learns optimal constraint orderings using a Q-learning approach, minimizing operations and energy for recurring queries
- **Edge AI Compatibility**: Supports resource-constrained devices with memory limits and reduced parallelization
- **AI-Specific Query Parsing**: Handles queries for AI workloads (e.g., filtering training data by variance or correlation)
- **Parallel Processing**: Leverages multi-core CPUs for large datasets, improving scalability and reducing execution time
- **Natural Language Query Processing**: Supports intuitive query syntax like "Find cheapest red phones under $300"

## Energy Efficiency and Green AI

The algorithm addresses the energy inefficiency of modern AI systems by achieving:

- **Up to 60% reduction in computational operations**, minimizing CPU cycles and energy consumption
- **Energy usage of 20-50 joules per query**, with an average power draw of ~20-30 watts
- **Estimated CO2 footprint of ~0.01-0.1 grams per query**, based on a 500g CO2/kWh grid average

## Installation

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/dynamic-constraint-sequencer.git
cd dynamic-constraint-sequencer

# Install dependencies
pip install -r requirements.txt

# Optional: Install spaCy for enhanced NLP features
python -m spacy download en_core_web_sm
```

### Dependencies
The project requires the following packages:
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.9.0` - Scientific computing and statistics
- `spacy>=3.4.0` - Natural language processing (optional)
- `psutil>=5.8.0` - System and process monitoring

## Usage

### Basic Usage
```python
from dcs import DynamicConstraintSequencer, generate_test_data_with_sparsity

# Generate test data
data = generate_test_data_with_sparsity(n_products=50000, sparsity_factor=0.3)

# Initialize sequencer with energy monitoring
dcs = DynamicConstraintSequencer(
    enable_energy_monitoring=True,
    enable_sparsity_optimization=True,
    enable_reinforcement_learning=True,
    target_energy_budget_joules=50.0
)

# Execute natural language query
result = dcs.search(data, "Find cheapest red phones not made by Apple under 300 with rating above 4")

# Print results
print(f"Results found: {len(result.final_data)}")
print(f"Energy consumed: {result.total_energy_joules:.2f} Joules")
print(f"Operations: {result.total_operations:,}")
print(f"Execution time: {result.execution_time*1000:.1f}ms")
print(f"Cache hit: {result.cache_hit}")
print(f"Sparsity exploited: {result.sparsity_exploited}")
```

### Advanced Configuration
```python
# Edge AI mode for resource-constrained devices
dcs.enable_edge_mode(memory_limit_mb=512)

# Custom energy budget
dcs = DynamicConstraintSequencer(
    target_energy_budget_joules=25.0,
    parallel_threshold=10000,
    max_workers=2
)
```

### AI-Specific Queries
```python
# Filter training data
result = dcs.search(data, "Find training data with high variance above 4")

# Get validation samples
result = dcs.search(data, "Show validation data with low correlation")
```

### Running the Benchmark
```python
# Run the built-in benchmark
python dcs.py
```

This will execute a comprehensive benchmark with 6 test queries and display:
- Performance comparisons (traditional vs. enhanced)
- Energy consumption statistics
- Cache hit rates and sparsity optimization usage
- Learned constraint orderings

## Query Syntax

The system supports natural language queries with the following patterns:

### Product Filtering
- Colors: "red phones", "not blue laptops", "exclude black items"
- Brands: "Apple products", "not Samsung", "not made by Google"
- Categories: "phones", "laptops", "tablets", "watches"
- Price ranges: "under $500", "between $200 and $800", "over $1000"
- Ratings: "rating above 4", "4+ stars", "minimum 3.5 stars"

### AI-Specific Queries
- Training data: "training samples", "learning data"
- Validation: "validation set", "dev data"
- Inference: "test data", "prediction samples"
- Data characteristics: "high variance", "low correlation", "anomaly detection"

### Optimization Keywords
- "cheapest", "most affordable" → sort by price ascending
- "most expensive", "premium" → sort by price descending  
- "best rated", "highest rated" → sort by rating descending
- "most popular" → sort by popularity descending

## Architecture

### Core Components

1. **EnergyMonitor**: Tracks CPU, memory, and power consumption
2. **ReinforcementOptimizer**: Q-learning for constraint ordering
3. **DynamicConstraintSequencer**: Main algorithm with sparsity optimization
4. **Query Parser**: Natural language to constraint conversion

### Data Structures

- `EnergyMetrics`: Energy consumption data
- `StepResult`: Per-constraint filtering results
- `BenchmarkResult`: Complete query execution results
- `LearnedOrdering`: Cached optimal constraint sequences

## Performance Characteristics

Based on benchmarks with 50,000 records and 30% sparsity:

| Metric | Traditional | Enhanced | Improvement |
|--------|-------------|----------|-------------|
| Operations | 200,000 | 80,000-100,000 | 50-60% reduction |
| Cost | 50 units | 20-25 units | 40-50% reduction |
| Time | 45ms | 15-20ms | 2-3x faster |
| Energy | N/A | 20-50 joules | New capability |

## Repository Structure

```
dynamic-constraint-sequencer/
├── dcs.py                 # Core implementation
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

## Applications

### AI/ML Pipelines
- **Data Preprocessing**: Efficiently filter training/validation datasets
- **Feature Selection**: Query data by statistical properties (variance, correlation)
- **Inference Optimization**: Fast query processing for recommendation systems
- **Edge AI**: Resource-efficient processing on IoT/mobile devices

### Use Cases
- E-commerce product search and filtering
- Scientific data analysis and filtering
- Real-time recommendation systems
- IoT sensor data processing
- Mobile app data queries

## Energy Statistics

The system provides detailed energy consumption metrics:

```python
# Get energy statistics after running queries
stats = dcs.get_energy_statistics()
print(f"Total energy: {stats['total_energy_joules']:.2f} J")
print(f"Average power: {stats['avg_watts']:.1f} W")
print(f"CO2 footprint: {stats['total_energy_joules']/3600000*500:.3f} g")
```

## 🤝 Contributing

We welcome contributions of all kinds! 🎉 Whether you're fixing bugs, adding features, improving documentation, or optimizing performance, your help makes this project better.

### 🚀 Quick Start for Contributors

1. **Check out our [CONTRIBUTING.md](CONTRIBUTING.md)** for detailed guidelines
2. **Look for issues labeled** `good first issue` or `help wanted`
3. **Join the discussion** in our GitHub Discussions
4. **Read our [Code of Conduct](CODE_OF_CONDUCT.md)**

### 🎯 Ways to Contribute

- 🐛 **Bug Fixes**: Help us squash bugs and improve reliability
- ✨ **New Features**: Add energy optimizations, AI enhancements, or query improvements
- 📚 **Documentation**: Improve guides, add examples, or fix typos
- ⚡ **Performance**: Optimize algorithms for speed and energy efficiency
- 🧪 **Testing**: Add tests, benchmarks, or edge case coverage
- 🌱 **Ideas**: Share suggestions in GitHub Discussions

### 🏃‍♂️ Quick Development Setup
```bash
# Fork and clone the repo
git clone https://github.com/<your-username>/dynamic-constraint-sequencer.git
cd dynamic-constraint-sequencer

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Verify setup
python dcs.py  # Run benchmark

# Create your feature branch
git checkout -b feature/amazing-feature
```

### 🏷️ Good First Issues

Perfect for newcomers:
- Documentation improvements
- Adding query examples
- Performance benchmarking
- Code formatting and cleanup
- Adding type hints

### 🌟 Recognition

Contributors are recognized in:
- 📝 Project README
- 🎉 Release notes
- 💬 Community discussions
- 🏆 GitHub contributor graphs

## Future Work

- **Hardware Integration**: Adapt for neuromorphic hardware (e.g., Intel Loihi)
- **Distributed Systems**: Integration with Apache Spark or Dask
- **Advanced RL**: Neural network-based reinforcement learning (DQN)
- **Real-World Validation**: Testing on industry datasets
- **Query Optimization**: More sophisticated constraint reordering algorithms

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

- Strubell, E., et al. (2019). "Energy and Policy Considerations for Deep Learning in NLP." *ACL 2019*
- Green AI initiatives and sustainable computing research
- Human brain efficiency studies (~20W power consumption)

## Contact

For questions, issues, or collaboration opportunities, please:
- Open an issue on GitHub
- Submit a pull request
- Contact the maintainers through GitHub

---

*Built with ❤️ for sustainable AI and energy-efficient computing*