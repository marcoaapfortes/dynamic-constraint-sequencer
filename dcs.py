import pandas as pd
import numpy as np
import time
import re
import hashlib
import pickle
import os
import psutil
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import threading
from scipy import sparse

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Install with: pip install scipy")

@dataclass
class EnergyMetrics:
    """Energy consumption metrics for a single operation."""
    cpu_percent: float
    memory_mb: float
    estimated_watts: float
    duration_seconds: float
    total_energy_joules: float

@dataclass
class StepResult:
    """Results from a single filtering step."""
    remaining: int
    eliminated_pct: float
    operations: int
    cost: float
    energy_metrics: Optional[EnergyMetrics] = None
    sparsity_ratio: float = 0.0

@dataclass
class BenchmarkResult:
    """Results from a benchmark execution."""
    final_data: pd.DataFrame
    total_operations: int
    total_cost: float
    total_energy_joules: float
    execution_time: float
    constraint_sequence: List[str]
    step_results: Dict[str, StepResult]
    cache_hit: bool = False
    sparsity_exploited: bool = False
    parallel_chunks_used: int = 0

@dataclass
class LearnedOrdering:
    """Cached constraint ordering with performance metrics."""
    sequence: List[str]
    avg_operations: float
    avg_cost: float
    avg_energy: float
    usage_count: int
    last_updated: float
    success_rate: float = 1.0

class EnergyMonitor:
    """Monitor energy consumption during algorithm execution."""
    
    def __init__(self):
        self.start_time = None
        self.start_cpu_percent = None
        self.start_memory = None
        self.monitoring = False
        self.base_power_watts = 10.0  # Idle system power
        self.cpu_power_per_percent = 1.5  # Watts per CPU percent
        self.memory_power_per_gb = 3.0  # Watts per GB of RAM
    
    def start_monitoring(self):
        """Start energy monitoring."""
        self.start_time = time.perf_counter()
        self.start_cpu_percent = psutil.cpu_percent(interval=0.1)
        self.start_memory = psutil.virtual_memory().used / (1024**3)
        self.monitoring = True
    
    def stop_monitoring(self) -> EnergyMetrics:
        """Stop monitoring and calculate energy consumption."""
        if not self.monitoring or self.start_time is None or self.start_cpu_percent is None or self.start_memory is None:
            return EnergyMetrics(0, 0, 0, 0, 0)
        
        end_time = time.perf_counter()
        end_cpu_percent = psutil.cpu_percent(interval=0.1)
        end_memory = psutil.virtual_memory().used / (1024**3)
        
        duration = end_time - self.start_time
        avg_cpu_percent = (self.start_cpu_percent + end_cpu_percent) / 2
        avg_memory_gb = (self.start_memory + end_memory) / 2
        
        estimated_watts = (
            self.base_power_watts + 
            (avg_cpu_percent * self.cpu_power_per_percent) +
            (avg_memory_gb * self.memory_power_per_gb)
        )
        
        total_energy_joules = estimated_watts * duration
        
        self.monitoring = False
        
        return EnergyMetrics(
            cpu_percent=avg_cpu_percent,
            memory_mb=avg_memory_gb * 1024,
            estimated_watts=estimated_watts,
            duration_seconds=duration,
            total_energy_joules=total_energy_joules
        )

class ReinforcementOptimizer:
    """Reinforcement learning for constraint ordering optimization."""
    
    def __init__(self, learning_rate: float = 0.1, exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.state_visits = {}
    
    def get_state_signature(self, constraints: Dict, data_stats: Dict) -> str:
        """Generate state signature for reinforcement learning."""
        state_features = {
            'constraint_types': sorted([c['type'] for c in constraints.values()]),
            'data_size_bucket': self._get_size_bucket(data_stats.get('size', 0)),
            'sparsity_bucket': self._get_sparsity_bucket(data_stats.get('sparsity', 0))
        }
        return hashlib.md5(str(state_features).encode()).hexdigest()[:8]
    
    def _get_size_bucket(self, size: int) -> str:
        """Bucket data size for state representation."""
        if size < 1000: return "small"
        elif size < 10000: return "medium"
        elif size < 100000: return "large"
        else: return "xlarge"
    
    def _get_sparsity_bucket(self, sparsity: float) -> str:
        """Bucket sparsity ratio for state representation."""
        if sparsity < 0.1: return "dense"
        elif sparsity < 0.3: return "medium_sparse"
        else: return "very_sparse"
    
    def select_action(self, state: str, available_actions: List[str]) -> str:
        """Select constraint ordering action using epsilon-greedy policy."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}
        
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_actions)
        else:
            return max(available_actions, key=lambda a: self.q_table[state].get(a, 0.0))
    
    def update_q_value(self, state: str, action: str, reward: float):
        """Update Q-value based on performance."""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        self.q_table[state][action] += self.learning_rate * (reward - self.q_table[state][action])

class DynamicConstraintSequencer:
    """
    Dynamic Constraint Sequencing algorithm optimized for energy efficiency.
    
    Features:
    - Energy consumption monitoring
    - Sparsity exploitation
    - Reinforcement learning for constraint ordering
    - Edge AI compatibility
    - AI-specific query parsing
    """
    
    def __init__(self, 
                 sample_size: int = 1000, 
                 enable_histograms: bool = True, 
                 missing_value_strategy: str = 'exclude',
                 enable_learning: bool = True,
                 enable_energy_monitoring: bool = True,
                 enable_sparsity_optimization: bool = True,
                 enable_reinforcement_learning: bool = True,
                 cache_file: str = 'dcs_cache.pkl',
                 parallel_threshold: int = 100000,
                 max_workers: Optional[int] = None,
                 target_energy_budget_joules: Optional[float] = None):
        """
        Initialize the sequencer with energy and performance optimizations.
        
        Args:
            sample_size: Size of sample for selectivity estimation.
            enable_histograms: Use pre-computed histograms for selectivity estimation.
            missing_value_strategy: Strategy for handling missing values ('exclude', 'impute_mode', 'impute_median').
            enable_learning: Enable caching of successful constraint orderings.
            enable_energy_monitoring: Track energy consumption.
            enable_sparsity_optimization: Exploit sparse data structures.
            enable_reinforcement_learning: Use RL for constraint ordering.
            cache_file: File to persist learned orderings.
            parallel_threshold: Dataset size threshold for parallel processing.
            max_workers: Number of worker processes for parallel execution.
            target_energy_budget_joules: Optional energy budget for queries.
        """
        self.sample_size = sample_size
        self.enable_histograms = enable_histograms
        self.missing_value_strategy = missing_value_strategy
        self.enable_learning = enable_learning
        self.enable_energy_monitoring = enable_energy_monitoring
        self.enable_sparsity_optimization = enable_sparsity_optimization
        self.enable_reinforcement_learning = enable_reinforcement_learning
        self.cache_file = cache_file
        self.parallel_threshold = parallel_threshold
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.target_energy_budget = target_energy_budget_joules
        
        self.column_histograms = {}
        self.column_correlations = {}
        self.sparse_columns = set()
        self.small_dataset_threshold = 1000
        
        self.learned_orderings: Dict[str, LearnedOrdering] = {}
        self.rl_optimizer = ReinforcementOptimizer() if enable_reinforcement_learning else None
        if self.enable_learning:
            self._load_cache()
        
        self.energy_monitor = EnergyMonitor() if enable_energy_monitoring else None
        self.energy_history = []
        
        self.constraint_costs = {
            'equality': 1.0,
            'inequality': 1.2,
            'comparison': 0.5,
            'string_contains': 2.0,
            'regex_match': 10.0,
            'range': 1.5,
            'sparse_equality': 0.3,
            'sparse_comparison': 0.2
        }
        
        self.edge_mode = False
        self.memory_limit_mb = None
        
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    
    def enable_edge_mode(self, memory_limit_mb: int = 512):
        """Enable optimizations for resource-constrained devices."""
        self.edge_mode = True
        self.memory_limit_mb = memory_limit_mb
        self.parallel_threshold = 10000
        self.max_workers = 2
        self.sample_size = 500
    
    def _detect_sparsity(self, data: pd.DataFrame) -> Dict[str, float]:
        """Detect sparse columns and calculate sparsity ratios."""
        sparsity_ratios = {}
        
        for column in data.columns:
            if data[column].dtype in ['object', 'category']:
                null_ratio = data[column].isnull().sum() / len(data)
                if null_ratio > 0.3:
                    self.sparse_columns.add(column)
                    sparsity_ratios[column] = null_ratio
            elif data[column].dtype in ['int64', 'float64']:
                zero_null_ratio = ((data[column] == 0) | data[column].isnull()).sum() / len(data)
                if zero_null_ratio > 0.3:
                    self.sparse_columns.add(column)
                    sparsity_ratios[column] = zero_null_ratio
        
        return sparsity_ratios
    
    def _optimize_for_sparsity(self, data: pd.DataFrame, constraint: Dict) -> bool:
        """Check if constraint can use sparsity optimizations."""
        column = constraint['column']
        return self.enable_sparsity_optimization and column in self.sparse_columns
    
    def _build_histograms(self, data: pd.DataFrame) -> None:
        """Build histograms and detect sparsity for selectivity estimation."""
        if not self.enable_histograms:
            return
        
        sparsity_ratios = self._detect_sparsity(data)
        
        for column in data.columns:
            col_data = data[column]
            
            if self.missing_value_strategy == 'impute_mode' and col_data.dtype == 'object':
                col_data = col_data.fillna(col_data.mode().iloc[0] if not col_data.mode().empty else 'unknown')
            elif self.missing_value_strategy == 'impute_median' and col_data.dtype in ['int64', 'float64']:
                col_data = col_data.fillna(col_data.median())
            
            if col_data.dtype == 'object' or col_data.dtype.name == 'category':
                histogram = Counter(col_data)
                # Store sparse ratio separately in the histogram dict
                histogram_dict: Dict[str, float] = dict(histogram)
                if column in self.sparse_columns:
                    histogram_dict['_sparse_ratio'] = sparsity_ratios.get(column, 0)
                self.column_histograms[column] = histogram_dict
            elif col_data.dtype in ['int64', 'float64']:
                self.column_histograms[column] = {
                    'percentiles': np.percentile(col_data.dropna(), [0, 25, 50, 75, 100]),
                    'total_count': len(col_data),
                    'sparse_ratio': sparsity_ratios.get(column, 0)
                }
    
    def _calculate_constraint_cost_with_energy(self, constraint: Dict, data_size: int, 
                                             current_energy: float = 0) -> float:
        """Calculate computational cost with energy and sparsity factors."""
        base_cost = self.constraint_costs.get(constraint['type'], 1.0)
        
        if self._optimize_for_sparsity(None, constraint):
            constraint_type = f"sparse_{constraint['type']}"
            base_cost = self.constraint_costs.get(constraint_type, base_cost * 0.3)
        
        size_factor = data_size / 1000.0
        
        if self.target_energy_budget and current_energy > 0:
            energy_factor = min(2.0, current_energy / self.target_energy_budget)
            base_cost *= energy_factor
        
        if self.edge_mode:
            base_cost *= 0.7
        
        return base_cost * size_factor
    
    def _generate_query_signature(self, constraints: Dict, optimization: Dict) -> str:
        """Generate a hash signature for a query pattern."""
        signature_data = {
            'constraints': {name: {k: v for k, v in constraint.items() if k != 'value'} 
                          for name, constraint in constraints.items()},
            'optimization': optimization,
            'sparse_columns': sorted(list(self.sparse_columns))
        }
        signature_str = str(sorted(signature_data.items()))
        return hashlib.md5(signature_str.encode()).hexdigest()[:12]
    
    def _load_cache(self):
        """Load learned orderings from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.learned_orderings = cache_data.get('orderings', {})
                    if 'rl_q_table' in cache_data and self.rl_optimizer:
                        self.rl_optimizer.q_table = cache_data['rl_q_table']
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                self.learned_orderings = {}
    
    def _save_cache(self):
        """Save learned orderings and RL state to disk."""
        if not self.enable_learning:
            return
        try:
            cache_data = {
                'orderings': self.learned_orderings,
                'rl_q_table': self.rl_optimizer.q_table if self.rl_optimizer else {}
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _update_learned_ordering(self, query_signature: str, sequence: List[str], 
                                operations: int, cost: float, energy: float, success: bool):
        """Update learned ordering with performance and energy data."""
        current_time = time.time()
        
        if query_signature in self.learned_orderings:
            learned = self.learned_orderings[query_signature]
            alpha = 0.3
            learned.avg_operations = (1 - alpha) * learned.avg_operations + alpha * operations
            learned.avg_cost = (1 - alpha) * learned.avg_cost + alpha * cost
            learned.avg_energy = (1 - alpha) * learned.avg_energy + alpha * energy
            learned.usage_count += 1
            learned.last_updated = current_time
            learned.success_rate = (learned.success_rate * (learned.usage_count - 1) + (1.0 if success else 0.0)) / learned.usage_count
        else:
            self.learned_orderings[query_signature] = LearnedOrdering(
                sequence=sequence,
                avg_operations=operations,
                avg_cost=cost,
                avg_energy=energy,
                usage_count=1,
                last_updated=current_time,
                success_rate=1.0 if success else 0.0
            )
    
    def parse_ai_specific_query(self, query: str) -> Dict[str, Any]:
        """Parse queries with AI-specific terms."""
        query = query.lower()
        constraints = {}
        
        ai_patterns = {
            'training': ['training', 'train', 'learning'],
            'inference': ['inference', 'predict', 'test'],
            'validation': ['validation', 'valid', 'dev'],
            'high_variance': ['high variance', 'diverse', 'varied'],
            'low_correlation': ['low correlation', 'uncorrelated', 'independent'],
            'anomaly': ['anomaly', 'outlier', 'unusual']
        }
        
        parsed = self.parse_problem_statement(query)
        constraints.update(parsed['constraints'])
        
        for ai_type, keywords in ai_patterns.items():
            if any(keyword in query for keyword in keywords):
                constraints[f'ai_{ai_type}'] = {
                    'type': 'ai_specific', 
                    'value': ai_type, 
                    'column': 'metadata'
                }
        
        return {
            'task_type': parsed['task_type'],
            'constraints': constraints,
            'optimization': parsed['optimization'],
            'original_query': query,
            'ai_specific': True
        }
    
    def parse_problem_statement(self, query: str) -> Dict[str, Any]:
        """Parse natural language query to extract constraints and optimization."""
        query = query.lower()
        constraints = {}
        
        colors = ['red', 'blue', 'green', 'black', 'white', 'silver', 'gold', 'yellow', 'purple']
        categories = ['electronics', 'phones', 'laptops', 'cars', 'books', 'clothing', 'tablets', 'watches', 'headphones']
        brands = ['apple', 'samsung', 'google', 'sony', 'microsoft', 'dell', 'hp', 'nvidia', 'amd']
        
        if self.nlp:
            doc = self.nlp(query)
            for token in doc:
                if token.text in colors:
                    constraints['color'] = {'type': 'equality', 'value': token.text, 'column': 'color'}
                elif token.text in categories:
                    constraints['category'] = {'type': 'equality', 'value': token.text, 'column': 'category'}
                elif token.text in brands:
                    constraints['brand'] = {'type': 'equality', 'value': token.text, 'column': 'brand'}
            
            for token in doc:
                if token.dep_ == 'neg' or token.text in ['not', 'exclude', 'without']:
                    for child in token.head.children:
                        if child.text in colors:
                            constraints['color'] = {'type': 'inequality', 'value': child.text, 'column': 'color'}
                        elif child.text in brands:
                            constraints['brand'] = {'type': 'inequality', 'value': child.text, 'column': 'brand'}
        else:
            for color in colors:
                if f'not {color}' in query or f'exclude {color}' in query:
                    constraints['color'] = {'type': 'inequality', 'value': color, 'column': 'color'}
                elif color in query:
                    constraints['color'] = {'type': 'equality', 'value': color, 'column': 'color'}
            
            for category in categories:
                if category in query:
                    constraints['category'] = {'type': 'equality', 'value': category, 'column': 'category'}
            
            for brand in brands:
                if f'not {brand}' in query or f'not made by {brand}' in query:
                    constraints['brand'] = {'type': 'inequality', 'value': brand, 'column': 'brand'}
                elif brand in query:
                    constraints['brand'] = {'type': 'equality', 'value': brand, 'column': 'brand'}
        
        price_patterns = [
            (r'under \$?(\d+)', '<='),
            (r'less than \$?(\d+)', '<'),
            (r'below \$?(\d+)', '<='),
            (r'over \$?(\d+)', '>'),
            (r'above \$?(\d+)', '>'),
            (r'between \$?(\d+) and \$?(\d+)', 'range'),
            (r'from \$?(\d+) to \$?(\d+)', 'range')
        ]
        
        for pattern, operator in price_patterns:
            match = re.search(pattern, query)
            if match:
                if operator == 'range':
                    min_price = int(match.group(1))
                    max_price = int(match.group(2))
                    constraints['price'] = {
                        'type': 'range',
                        'value': (min_price, max_price),
                        'column': 'price'
                    }
                else:
                    price = int(match.group(1))
                    constraints['price'] = {
                        'type': 'comparison',
                        'value': price,
                        'column': 'price',
                        'operator': operator
                    }
                break
        
        rating_patterns = [
            (r'rating\s+(?:above|over|greater than)\s+(\d+(?:\.\d+)?)', '>='),
            (r'(\d+(?:\.\d+)?)\+?\s*stars?\s+(?:or|and)\s+(?:above|over|higher)', '>='),
            (r'(?:at least|minimum)\s+(\d+(?:\.\d+)?)\s*stars?', '>='),
            (r'(\d+(?:\.\d+)?)\s*stars?\s*(?:minimum|or better)', '>=')
        ]
        
        for pattern, operator in rating_patterns:
            match = re.search(pattern, query)
            if match:
                rating = float(match.group(1))
                constraints['rating'] = {
                    'type': 'comparison',
                    'value': rating,
                    'column': 'rating',
                    'operator': operator
                }
                break
        
        optimization = None
        optimization_patterns = [
            (['cheapest', 'lowest price', 'most affordable'], 'price', 'asc'),
            (['most expensive', 'highest price', 'premium'], 'price', 'desc'),
            (['best rated', 'highest rated', 'top rated'], 'rating', 'desc'),
            (['most popular', 'bestselling'], 'popularity', 'desc'),
            (['newest', 'latest', 'most recent'], 'date', 'desc')
        ]
        
        for keywords, column, direction in optimization_patterns:
            if any(keyword in query for keyword in keywords):
                optimization = {'column': column, 'direction': direction}
                break
        
        return {
            'task_type': 'search',
            'constraints': constraints,
            'optimization': optimization,
            'original_query': query
        }
    
    def estimate_selectivity_with_sparsity(self, data: pd.DataFrame, constraint: Dict) -> float:
        """Estimate constraint selectivity with sparsity considerations."""
        column = constraint['column']
        if column not in data.columns:
            return 0.0
        
        if column in self.sparse_columns:
            if constraint['type'] == 'equality' and constraint.get('value') in [None, 0, '', 'null']:
                return 0.9
        
        col_data = data[column]
        if self.missing_value_strategy == 'exclude':
            col_data = col_data.dropna()
        
        total_count = len(col_data)
        if total_count == 0:
            return 0.0
        
        if self.enable_histograms and column in self.column_histograms:
            hist = self.column_histograms[column]
            
            if constraint['type'] == 'equality':
                if isinstance(hist, dict) and 'percentiles' not in hist:
                    # This is a categorical histogram (converted from Counter)
                    matching_count = hist.get(constraint['value'], 0)
                    remaining_ratio = matching_count / total_count
                    if '_sparse_ratio' in hist:
                        remaining_ratio *= (1 - hist['_sparse_ratio'] * 0.5)
                else:
                    remaining_ratio = 0.1
            elif constraint['type'] == 'inequality':
                if isinstance(hist, dict) and 'percentiles' not in hist:
                    # This is a categorical histogram (converted from Counter)
                    matching_count = sum(v for k, v in hist.items() if k != constraint['value'] and not k.startswith('_'))
                    remaining_ratio = matching_count / total_count
                else:
                    remaining_ratio = 0.9
            elif constraint['type'] == 'range':
                if isinstance(hist, dict) and 'percentiles' in hist:
                    min_val, max_val = constraint['value']
                    percentiles = hist['percentiles']
                    min_percentile = np.searchsorted(percentiles, min_val) / len(percentiles)
                    max_percentile = np.searchsorted(percentiles, max_val) / len(percentiles)
                    remaining_ratio = max_percentile - min_percentile
                else:
                    remaining_ratio = 0.3
            elif constraint['type'] == 'comparison' and isinstance(hist, dict):
                percentiles = hist['percentiles']
                value = constraint['value']
                
                if constraint['operator'] in ['<=', '<']:
                    remaining_ratio = np.searchsorted(percentiles, value) / len(percentiles)
                elif constraint['operator'] in ['>=', '>']:
                    remaining_ratio = 1.0 - (np.searchsorted(percentiles, value) / len(percentiles))
                else:
                    remaining_ratio = 0.5
            else:
                remaining_ratio = 0.5
        else:
            sample_size = min(self.sample_size, total_count)
            if self.edge_mode:
                sample_size = min(sample_size, 200)
                
            sample_data = col_data.sample(sample_size, random_state=0)
            
            if constraint['type'] == 'equality':
                remaining_ratio = (sample_data == constraint['value']).mean()
            elif constraint['type'] == 'inequality':
                remaining_ratio = (sample_data != constraint['value']).mean()
            elif constraint['type'] == 'range':
                min_val, max_val = constraint['value']
                remaining_ratio = ((sample_data >= min_val) & (sample_data <= max_val)).mean()
            elif constraint['type'] == 'comparison':
                if constraint['operator'] == '<=':
                    remaining_ratio = (sample_data <= constraint['value']).mean()
                elif constraint['operator'] == '>=':
                    remaining_ratio = (sample_data >= constraint['value']).mean()
                elif constraint['operator'] == '<':
                    remaining_ratio = (sample_data < constraint['value']).mean()
                elif constraint['operator'] == '>':
                    remaining_ratio = (sample_data > constraint['value']).mean()
                else:
                    remaining_ratio = (sample_data == constraint['value']).mean()
            else:
                remaining_ratio = 0.5
        
        return 1.0 - remaining_ratio
    
    def sequence_constraints_with_reinforcement_learning(self, data: pd.DataFrame, 
                                                       constraints: Dict) -> List[Tuple[str, Dict]]:
        """Sequence constraints using reinforcement learning."""
        if not self.rl_optimizer or len(constraints) <= 1:
            return self.sequence_constraints_with_cost_and_energy(data, constraints)
        
        data_stats = {
            'size': len(data),
            'sparsity': len(self.sparse_columns) / len(data.columns) if len(data.columns) > 0 else 0
        }
        state = self.rl_optimizer.get_state_signature(constraints, data_stats)
        
        cost_ordered = self.sequence_constraints_with_cost_and_energy(data, constraints)
        available_actions = [f"{c[0]}_first" for c in cost_ordered[:3]]
        
        if not available_actions:
            return cost_ordered
        
        selected_action = self.rl_optimizer.select_action(state, available_actions)
        
        if selected_action.endswith('_first'):
            first_constraint = selected_action.replace('_first', '')
            reordered = [(first_constraint, constraints[first_constraint])]
            reordered.extend([(name, constraint) for name, constraint in cost_ordered 
                            if name != first_constraint])
            return reordered
        
        return cost_ordered
    
    def sequence_constraints_with_cost_and_energy(self, data: pd.DataFrame, 
                                                constraints: Dict, 
                                                current_energy: float = 0) -> List[Tuple[str, Dict]]:
        """Sequence constraints by cost-benefit ratio with energy considerations."""
        if not self.column_correlations:
            self._compute_column_correlations(data, constraints)
        
        constraint_scores = {}
        data_size = len(data)
        
        for name, constraint in constraints.items():
            selectivity = self.estimate_selectivity_with_sparsity(data, constraint)
            cost = self._calculate_constraint_cost_with_energy(constraint, data_size, current_energy)
            
            base_score = selectivity / (cost + 0.1)
            
            if constraint['column'] in self.sparse_columns:
                base_score *= 1.3
            
            if self.edge_mode:
                if constraint['type'] in ['equality', 'comparison']:
                    base_score *= 1.2
                elif constraint['type'] in ['regex_match', 'string_contains']:
                    base_score *= 0.7
            
            constraint_scores[name] = base_score
        
        for (col1, col2), p_value in self.column_correlations.items():
            if p_value < 0.05:
                name1 = next((n for n, c in constraints.items() if c['column'] == col1), None)
                name2 = next((n for n, c in constraints.items() if c['column'] == col2), None)
                if name1 and name2 and name1 in constraint_scores and name2 in constraint_scores:
                    if constraint_scores[name1] > constraint_scores[name2]:
                        constraint_scores[name2] *= 0.8
                    else:
                        constraint_scores[name1] *= 0.8
        
        return sorted(
            constraints.items(),
            key=lambda x: constraint_scores[x[0]],
            reverse=True
        )
    
    def apply_constraint_with_sparsity(self, data: pd.DataFrame, constraint: Dict) -> Tuple[pd.DataFrame, int, bool]:
        """Apply constraint with sparsity optimization."""
        column = constraint['column']
        if column not in data.columns:
            return data, 0, False
        
        initial_size = len(data)
        sparsity_exploited = False
        
        if self._optimize_for_sparsity(data, constraint):
            sparsity_exploited = True
            if constraint['type'] == 'equality':
                if constraint['value'] in [None, 0, '', 'null']:
                    if constraint['value'] in [None, 'null']:
                        mask = data[column].isnull()
                    else:
                        mask = data[column] == constraint['value']
                else:
                    mask = data[column] == constraint['value']
            elif constraint['type'] == 'inequality':
                if constraint['value'] in [None, 0, '', 'null']:
                    if constraint['value'] in [None, 'null']:
                        mask = ~data[column].isnull()
                    else:
                        mask = data[column] != constraint['value']
                else:
                    mask = data[column] != constraint['value']
            else:
                filtered_data, operations = self.apply_constraint_vectorized(data, constraint)
                return filtered_data, operations, False
        else:
            filtered_data, operations = self.apply_constraint_vectorized(data, constraint)
            return filtered_data, operations, False
        
        if self.missing_value_strategy == 'exclude' and not sparsity_exploited:
            mask = mask & ~data[column].isnull()
        
        filtered_data = data[mask]
        operations = initial_size
        
        return filtered_data, operations, sparsity_exploited
    
    def apply_constraint_vectorized(self, data: pd.DataFrame, constraint: Dict) -> Tuple[pd.DataFrame, int]:
        """Apply constraint using vectorized operations."""
        column = constraint['column']
        if column not in data.columns:
            return data, 0
        
        initial_size = len(data)
        
        if constraint['type'] == 'equality':
            mask = data[column] == constraint['value']
        elif constraint['type'] == 'inequality':
            mask = data[column] != constraint['value']
        elif constraint['type'] == 'range':
            min_val, max_val = constraint['value']
            mask = (data[column] >= min_val) & (data[column] <= max_val)
        elif constraint['type'] == 'comparison':
            if constraint['operator'] == '<=':
                mask = data[column] <= constraint['value']
            elif constraint['operator'] == '>=':
                mask = data[column] >= constraint['value']
            elif constraint['operator'] == '<':
                mask = data[column] < constraint['value']
            elif constraint['operator'] == '>':
                mask = data[column] > constraint['value']
            else:
                mask = data[column] == constraint['value']
        else:
            mask = pd.Series([True] * len(data), index=data.index)
        
        if self.missing_value_strategy == 'exclude':
            mask = mask & ~data[column].isnull()
        
        filtered_data = data[mask]
        operations = initial_size
        
        return filtered_data, operations
    
    def search(self, data: pd.DataFrame, query: str) -> BenchmarkResult:
        """Execute query with energy and performance optimizations."""
        start_time = time.perf_counter()
        cache_hit = False
        total_energy = 0.0
        sparsity_exploited = False
        parallel_chunks_used = 0
        
        if self.energy_monitor:
            self.energy_monitor.start_monitoring()
        
        if len(data) < self.small_dataset_threshold:
            result = self._traditional_search(data, query)
            if self.energy_monitor:
                energy_metrics = self.energy_monitor.stop_monitoring()
                result.total_energy_joules = energy_metrics.total_energy_joules
            return result
        
        if self.edge_mode and self.memory_limit_mb:
            current_memory = psutil.virtual_memory().used / (1024**2)
            if current_memory > self.memory_limit_mb * 0.8:
                self.sample_size = min(self.sample_size, 200)
                self.parallel_threshold *= 2
        
        if self.enable_histograms and not self.column_histograms:
            self._build_histograms(data)
        
        if 'training' in query or 'inference' in query or 'variance' in query:
            parsed = self.parse_ai_specific_query(query)
        else:
            parsed = self.parse_problem_statement(query)
        
        constraints = parsed['constraints']
        optimization = parsed['optimization']
        
        if not constraints:
            if optimization and len(data) > 0:
                final_data = data.sort_values(optimization['column'], 
                                            ascending=(optimization['direction'] == 'asc'))
            else:
                final_data = data
            
            execution_time = time.perf_counter() - start_time
            if self.energy_monitor:
                energy_metrics = self.energy_monitor.stop_monitoring()
                total_energy = energy_metrics.total_energy_joules
            
            return BenchmarkResult(
                final_data=final_data,
                total_operations=0,
                total_cost=0.0,
                total_energy_joules=total_energy,
                execution_time=execution_time,
                constraint_sequence=[],
                step_results={},
                cache_hit=False,
                sparsity_exploited=False,
                parallel_chunks_used=0
            )
        
        query_signature = self._generate_query_signature(constraints, optimization)
        ordered_constraints = None
        
        if self.enable_learning and query_signature in self.learned_orderings:
            learned = self.learned_orderings[query_signature]
            if learned.usage_count >= 2 and learned.success_rate > 0.7:
                constraint_names = learned.sequence
                ordered_constraints = [(name, constraints[name]) for name in constraint_names if name in constraints]
                cache_hit = True
        
        if not ordered_constraints:
            if self.enable_reinforcement_learning:
                ordered_constraints = self.sequence_constraints_with_reinforcement_learning(data, constraints)
            else:
                ordered_constraints = self.sequence_constraints_with_cost_and_energy(data, constraints, total_energy)
        
        current_data = data.copy()
        total_operations = 0
        total_cost = 0.0
        constraint_sequence = []
        step_results = {}
        
        for constraint_name, constraint_def in ordered_constraints:
            initial_count = len(current_data)
            step_energy_start = total_energy
            
            if len(current_data) >= self.parallel_threshold and not self.edge_mode:
                current_data, operations = self.apply_constraint_parallel(current_data, constraint_def)
                parallel_chunks_used += self.max_workers
            elif self.enable_sparsity_optimization:
                current_data, operations, step_sparsity = self.apply_constraint_with_sparsity(current_data, constraint_def)
                if step_sparsity:
                    sparsity_exploited = True
            else:
                current_data, operations = self.apply_constraint_vectorized(current_data, constraint_def)
            
            final_count = len(current_data)
            constraint_cost = self._calculate_constraint_cost_with_energy(constraint_def, initial_count, total_energy)
            
            total_operations += operations
            total_cost += constraint_cost
            constraint_sequence.append(constraint_name)
            
            eliminated_pct = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
            
            sparsity_ratio = 0.0
            if constraint_def['column'] in self.sparse_columns:
                if constraint_def['column'] in self.column_histograms:
                    hist = self.column_histograms[constraint_def['column']]
                    if isinstance(hist, dict) and '_sparse_ratio' in hist:
                        sparsity_ratio = hist['_sparse_ratio']
                    elif isinstance(hist, dict) and 'sparse_ratio' in hist:
                        sparsity_ratio = hist['sparse_ratio']
            
            step_results[constraint_name] = StepResult(
                remaining=final_count,
                eliminated_pct=eliminated_pct,
                operations=operations,
                cost=constraint_cost,
                sparsity_ratio=sparsity_ratio
            )
            
            if self.target_energy_budget and total_energy > self.target_energy_budget:
                break
        
        if optimization and len(current_data) > 0:
            current_data = current_data.sort_values(
                optimization['column'],
                ascending=(optimization['direction'] == 'asc')
            )
            sort_operations = int(len(current_data) * math.log2(max(len(current_data), 2)))
            sort_cost = sort_operations * 0.1
            total_operations += sort_operations
            total_cost += sort_cost
        
        execution_time = time.perf_counter() - start_time
        
        if self.energy_monitor:
            energy_metrics = self.energy_monitor.stop_monitoring()
            total_energy = energy_metrics.total_energy_joules
            self.energy_history.append(energy_metrics)
        
        success = len(current_data) > 0
        
        if self.enable_learning and not cache_hit:
            self._update_learned_ordering(query_signature, constraint_sequence, 
                                        total_operations, total_cost, total_energy, success)
            self._save_cache()
        
        if self.rl_optimizer and not cache_hit:
            baseline_operations = len(data) * len(constraints)
            efficiency_reward = (baseline_operations - total_operations) / baseline_operations
            energy_reward = -total_energy / 100.0
            reward = efficiency_reward + energy_reward + (0.5 if success else -0.5)
            
            data_stats = {
                'size': len(data),
                'sparsity': len(self.sparse_columns) / len(data.columns) if len(data.columns) > 0 else 0
            }
            state = self.rl_optimizer.get_state_signature(constraints, data_stats)
            action = f"{constraint_sequence[0]}_first" if constraint_sequence else "default"
            self.rl_optimizer.update_q_value(state, action, reward)
        
        return BenchmarkResult(
            final_data=current_data,
            total_operations=total_operations,
            total_cost=total_cost,
            total_energy_joules=total_energy,
            execution_time=execution_time,
            constraint_sequence=constraint_sequence,
            step_results=step_results,
            cache_hit=cache_hit,
            sparsity_exploited=sparsity_exploited,
            parallel_chunks_used=parallel_chunks_used
        )
    
    def apply_constraint_parallel(self, data: pd.DataFrame, constraint: Dict) -> Tuple[pd.DataFrame, int]:
        """Apply constraint using parallel processing for large datasets."""
        if len(data) < self.parallel_threshold:
            return self.apply_constraint_vectorized(data, constraint)
        
        chunk_size = len(data) // self.max_workers
        chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            def apply_to_chunk(chunk):
                return self.apply_constraint_vectorized(chunk, constraint)
            
            futures = [executor.submit(apply_to_chunk, chunk) for chunk in chunks]
            
            results = []
            total_operations = 0
            
            for future in as_completed(futures):
                chunk_result, chunk_ops = future.result()
                results.append(chunk_result)
                total_operations += chunk_ops
        
        if results:
            combined_data = pd.concat(results, ignore_index=True)
        else:
            combined_data = pd.DataFrame(columns=data.columns)
        
        return combined_data, total_operations
    
    def _traditional_search(self, data: pd.DataFrame, query: str) -> BenchmarkResult:
        """Execute traditional search for baseline comparison."""
        start_time = time.perf_counter()
        
        parsed = self.parse_problem_statement(query)
        constraints = parsed['constraints']
        optimization = parsed['optimization']
        
        current_data = data.copy()
        total_operations = len(data) * len(constraints) if constraints else 0
        total_cost = sum(self._calculate_constraint_cost_with_energy(c, len(data), 0) 
                        for c in constraints.values()) if constraints else 0
        
        step_results = {}
        constraint_sequence = list(constraints.keys()) if constraints else []
        
        for constraint_name, constraint_def in constraints.items():
            initial_count = len(current_data)
            current_data, _ = self.apply_constraint_vectorized(current_data, constraint_def)
            final_count = len(current_data)
            
            eliminated_pct = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
            constraint_cost = self._calculate_constraint_cost_with_energy(constraint_def, len(data), 0)
            
            step_results[constraint_name] = StepResult(
                remaining=final_count,
                eliminated_pct=eliminated_pct,
                operations=len(data),
                cost=constraint_cost
            )
        
        if optimization and len(current_data) > 0:
            current_data = current_data.sort_values(
                optimization['column'],
                ascending=(optimization['direction'] == 'asc')
            )
            sort_operations = int(len(current_data) * math.log2(max(len(current_data), 2)))
            total_operations += sort_operations
            total_cost += sort_operations * 0.1
        
        execution_time = time.perf_counter() - start_time
        
        return BenchmarkResult(
            final_data=current_data,
            total_operations=total_operations,
            total_cost=total_cost,
            total_energy_joules=0.0,
            execution_time=execution_time,
            constraint_sequence=constraint_sequence,
            step_results=step_results,
            cache_hit=False,
            sparsity_exploited=False,
            parallel_chunks_used=0
        )
    
    def _compute_column_correlations(self, data: pd.DataFrame, constraints: Dict) -> None:
        """Compute correlations between constraint columns."""
        if not SCIPY_AVAILABLE:
            return
            
        self.column_correlations = {}
        constraint_columns = [c['column'] for c in constraints.values() if c['column'] in data.columns]
        
        for i, col1 in enumerate(constraint_columns):
            for col2 in constraint_columns[i+1:]:
                if col1 == col2:
                    continue
                if data[col1].dtype == 'object' and data[col2].dtype == 'object':
                    try:
                        contingency = pd.crosstab(data[col1], data[col2])
                        chi2, p, _, _ = chi2_contingency(contingency)
                        self.column_correlations[(col1, col2)] = p
                    except:
                        pass
    
    def get_energy_statistics(self) -> Dict[str, float]:
        """Retrieve energy consumption statistics."""
        if not self.energy_history:
            return {}
        
        energies = [e.total_energy_joules for e in self.energy_history]
        return {
            'total_energy_joules': sum(energies),
            'avg_energy_per_query': np.mean(energies),
            'min_energy': min(energies),
            'max_energy': max(energies),
            'energy_std': np.std(energies),
            'avg_watts': np.mean([e.estimated_watts for e in self.energy_history])
        }

def generate_test_data_with_sparsity(n_products: int = 50000, sparsity_factor: float = 0.3) -> pd.DataFrame:
    """Generate test dataset with configurable sparsity."""
    np.random.seed(42)
    
    colors = ['red', 'blue', 'green', 'black', 'white', 'silver']
    categories = ['phones', 'laptops', 'tablets', 'watches', 'headphones']
    brands = ['Apple', 'Samsung', 'Google', 'Sony', 'Microsoft']
    
    color_data = np.random.choice(colors + [np.nan], n_products, 
                                 p=[0.14, 0.14, 0.14, 0.14, 0.14, 0.14, sparsity_factor])
    category_data = np.random.choice(categories + [np.nan], n_products,
                                   p=[0.16, 0.16, 0.16, 0.16, 0.16, sparsity_factor])
    brand_data = np.random.choice(brands + [np.nan], n_products,
                                p=[0.16, 0.16, 0.16, 0.16, 0.16, sparsity_factor])
    
    price_data = np.random.randint(50, 1000, n_products)
    zero_indices = np.random.choice(n_products, int(n_products * sparsity_factor * 0.5), replace=False)
    price_data[zero_indices] = 0
    
    data = pd.DataFrame({
        'id': range(n_products),
        'name': [f'Product {i}' for i in range(n_products)],
        'color': color_data,
        'category': category_data,
        'brand': brand_data,
        'price': price_data,
        'rating': np.round(np.random.uniform(1, 5, n_products), 1),
        'stock': np.random.randint(0, 100, n_products),
        'metadata': np.random.choice(['training', 'validation', 'test', None], n_products,
                                   p=[0.6, 0.2, 0.1, 0.1])
    })
    
    return data

def run_benchmark():
    """Run benchmark to evaluate energy and performance optimizations."""
    print("Benchmark: Dynamic Constraint Sequencing with Energy Optimization")
    print("=" * 70)
    
    data = generate_test_data_with_sparsity(50000, sparsity_factor=0.3)
    print(f"Generated dataset: {len(data):,} records")
    
    dcs = DynamicConstraintSequencer(
        enable_histograms=True,
        enable_learning=True,
        enable_energy_monitoring=True,
        enable_sparsity_optimization=True,
        enable_reinforcement_learning=True,
        parallel_threshold=25000,
        max_workers=4,
        target_energy_budget_joules=50.0
    )
    
    test_queries = [
        "Find the cheapest red phones not made by Apple under 300 with rating above 4",
        "Show blue laptops under 800 with good ratings",
        "Find expensive Samsung phones with rating above 4.5",
        "Get black tablets between 200 and 500 from Google",
        "Find training data with high ratings above 4",
        "Find the cheapest red phones not made by Apple under 300 with rating above 4"
    ]
    
    print(f"Executing {len(test_queries)} test queries")
    all_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)
        
        traditional_result = dcs._traditional_search(data, query)
        enhanced_result = dcs.search(data, query)
        
        all_results.append((traditional_result, enhanced_result))
        
        ops_reduction = ((traditional_result.total_operations - enhanced_result.total_operations) 
                        / traditional_result.total_operations * 100) if traditional_result.total_operations > 0 else 0
        cost_reduction = ((traditional_result.total_cost - enhanced_result.total_cost) 
                         / traditional_result.total_cost * 100) if traditional_result.total_cost > 0 else 0
        speed_improvement = traditional_result.execution_time / enhanced_result.execution_time if enhanced_result.execution_time > 0 else 1
        
        print(f"Results found: {len(enhanced_result.final_data)}")
        print(f"Operations: {traditional_result.total_operations:,} -> {enhanced_result.total_operations:,} ({ops_reduction:.1f}% reduction)")
        print(f"Cost: {traditional_result.total_cost:.2f} -> {enhanced_result.total_cost:.2f} ({cost_reduction:.1f}% reduction)")
        print(f"Time: {traditional_result.execution_time*1000:.1f}ms -> {enhanced_result.execution_time*1000:.1f}ms ({speed_improvement:.1f}x)")
        print(f"Energy: {enhanced_result.total_energy_joules:.2f} Joules")
        print(f"Features: Cache hit: {enhanced_result.cache_hit}, Sparsity: {enhanced_result.sparsity_exploited}, Parallel chunks: {enhanced_result.parallel_chunks_used}")
        print(f"Constraint sequence: {' -> '.join(enhanced_result.constraint_sequence)}")
    
    print("\nBenchmark Summary")
    print("=" * 70)
    
    total_traditional_ops = sum(r[0].total_operations for r in all_results)
    total_enhanced_ops = sum(r[1].total_operations for r in all_results)
    total_traditional_cost = sum(r[0].total_cost for r in all_results)
    total_enhanced_cost = sum(r[1].total_cost for r in all_results)
    total_enhanced_energy = sum(r[1].total_energy_joules for r in all_results)
    
    overall_ops_reduction = ((total_traditional_ops - total_enhanced_ops) / total_traditional_ops * 100) if total_traditional_ops > 0 else 0
    overall_cost_reduction = ((total_traditional_cost - total_enhanced_cost) / total_traditional_cost * 100) if total_traditional_cost > 0 else 0
    cache_hits = sum(1 for r in all_results if r[1].cache_hit)
    sparsity_exploited = sum(1 for r in all_results if r[1].sparsity_exploited)
    
    print(f"Overall operations reduction: {overall_ops_reduction:.1f}%")
    print(f"Overall cost reduction: {overall_cost_reduction:.1f}%")
    print(f"Total energy consumed: {total_enhanced_energy:.2f} Joules")
    print(f"Cache hits: {cache_hits}/{len(all_results)} queries ({cache_hits/len(all_results)*100:.1f}%)")
    print(f"Sparsity optimizations: {sparsity_exploited}/{len(all_results)} queries")
    print(f"Learned orderings: {len(dcs.learned_orderings)}")
    
    stats = dcs.get_energy_statistics()
    if stats:
        print("\nEnergy Statistics")
        print("-" * 40)
        print(f"Total energy consumed: {stats['total_energy_joules']:.2f} Joules")
        print(f"Average power draw: {stats['avg_watts']:.1f} Watts")
        print(f"Queries processed: {len(dcs.energy_history)}")
        print(f"Average energy per query: {stats['avg_energy_per_query']:.2f} Joules")
        kwh_consumed = stats['total_energy_joules'] / 3600000
        co2_grams = kwh_consumed * 500
        print(f"Estimated CO2 footprint: {co2_grams:.3f} grams")
    
    return all_results

if __name__ == "__main__":
    results = run_benchmark()
    if os.path.exists('dcs_cache.pkl'):
        os.remove('dcs_cache.pkl')