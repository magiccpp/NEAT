import getopt
from pathlib import Path
import sys
from tracemalloc import start
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
import copy
from typing import Dict, List, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import re
import time
import yfinance as yfin
import os
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import glob

DAILY_COST = 0


# Add this standalone worker function OUTSIDE your class (replace the existing one)
def evaluate_creature_batch_worker_process(creature_data_batch, returns_price_vec, features_vec, commission_rate, current_step, num_features, num_stocks, long_only):
    """Process-based worker function that evaluates a batch of creatures"""
    import os
    process_id = os.getpid()

    results = []

    for creature_data in creature_data_batch:
        try:
            # Reconstruct creature from data
            creature_id, nodes_data, connections_data, portfolio, energy, birth_step = creature_data

            # Create creature
            creature = NEATCreature(
                num_features=num_features,
                num_portfolio_positions=num_stocks + 1,
                creature_id=creature_id,
                long_only=long_only
            )

            # Restore nodes
            creature.nodes = {}
            creature.next_node_id = 0
            for node_id, node_data in nodes_data.items():
                node = NodeGene(node_id, node_data['type'], node_data['activation'])
                node.input_index = node_data['input_index']
                node.output_index = node_data['output_index']
                creature.nodes[node_id] = node
                creature.next_node_id = max(creature.next_node_id, node_id + 1)

            # Restore connections
            creature.connections = {}
            for innovation, conn_data in connections_data.items():
                conn = ConnectionGene(
                    conn_data['from_node'],
                    conn_data['to_node'],
                    conn_data['weight'],
                    conn_data['enabled'],
                    innovation
                )
                creature.connections[innovation] = conn

            # Restore state
            creature.portfolio = portfolio
            creature.energy = energy
            creature.birth_step = birth_step

            # Forward pass
            new_portfolio = creature.forward(features_vec)

            # Calculate portfolio returns (simplified calculation)
            portfolio_change = np.abs(new_portfolio - creature.portfolio)
            turnover = np.sum(portfolio_change[:-1])
            cost_frac = commission_rate * turnover

            gross_simple = np.dot(creature.portfolio[:-1], np.expm1(returns_price_vec))
            gross_log = np.log(max(1.0 + gross_simple, 1e-12))
            net_log_return = gross_log + np.log(max(1.0 - cost_frac, 1e-12))

            new_energy = creature.energy * np.exp(net_log_return)
            age = current_step - creature.birth_step
            alive = new_energy > 0.0

            # Return minimal data needed
            results.append((creature_id, new_portfolio, new_energy, net_log_return, age, alive))

        except Exception as e:
            print(f"Process {process_id} error for creature {creature_data[0]}: {e}")
            results.append((creature_data[0], creature_data[3], 0.0, -10.0, 0, False))

    return results


# Add this standalone worker function OUTSIDE your class
def evaluate_creature_batch_worker_thread(creatures_batch, returns_price_vec, features_vec, commission_rate, current_step):
    """Thread-based worker function that evaluates a batch of creatures"""
    import threading
    thread_id = threading.get_ident()

    results = []

    for creature in creatures_batch:
        try:
            # Forward pass
            new_portfolio = creature.forward(features_vec)

            # Calculate portfolio returns
            portfolio_change = np.abs(new_portfolio - creature.portfolio)
            turnover = np.sum(portfolio_change[:-1])
            cost_frac = commission_rate * turnover

            gross_simple = np.dot(creature.portfolio[:-1], np.expm1(returns_price_vec))
            gross_log = np.log(max(1.0 + gross_simple, 1e-12))
            net_log_return = gross_log + np.log(max(1.0 - cost_frac, 1e-12))

            new_energy = creature.energy * np.exp(net_log_return)
            age = current_step - creature.birth_step
            alive = new_energy > 0.0

            results.append((creature, new_portfolio, new_energy, net_log_return, age, alive))

        except Exception as e:
            print(f"Thread {thread_id} error for creature {creature.creature_id}: {e}")
            results.append((creature, creature.portfolio, 0.0, -10.0, 0, False))

    return results


# Global innovation tracking for NEAT
class InnovationTracker:
    def __init__(self):
        self.innovation_number = 0
        self.connection_innovations = {}  # (from_node, to_node) -> innovation_number

    def get_innovation_number(self, from_node: int, to_node: int) -> int:
        """Get innovation number for a connection, creating one if it doesn't exist"""
        key = (from_node, to_node)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.innovation_number
            self.innovation_number += 1
        return self.connection_innovations[key]

# Global tracker
innovation_tracker = InnovationTracker()

class Species:
    """Species class for NEAT speciation"""

    def __init__(self, representative, species_id: int, complexity_panelty: float):
        self.species_id = species_id
        self.representative = representative  # Representative genome
        self.members = [representative]
        self.best_fitness = representative.energy
        self.average_fitness = representative.energy
        self.generations_since_improvement = 0
        self.offspring_count = 0
        self.age = 0  # Age in reproduction cycles, not steps

        # Track species complexity metrics
        self.max_nodes = len(representative.nodes)
        self.max_connections = len([c for c in representative.connections.values() if c.enabled])
        self.has_recurrent = self._has_recurrent_connections(representative)
        self.complexity_panelty = complexity_panelty

    def _has_recurrent_connections(self, creature):
        """Check if creature has recurrent connections"""
        for conn in creature.connections.values():
            if not conn.enabled:
                continue
            # Simple check: if to_node <= from_node, it might be recurrent
            if conn.to_node <= conn.from_node:
                return True
        return False

    def add_member(self, creature):
        """Add a creature to this species"""
        self.members.append(creature)
        creature.species_id = self.species_id

        # Update complexity metrics
        node_count = len(creature.nodes)
        conn_count = len([c for c in creature.connections.values() if c.enabled])

        if node_count > self.max_nodes:
            self.max_nodes = node_count
        if conn_count > self.max_connections:
            self.max_connections = conn_count
        if not self.has_recurrent and self._has_recurrent_connections(creature):
            self.has_recurrent = True

    def update_fitness_stats(self):
        """Update species fitness statistics"""
        if not self.members:
            return

        fitnesses = [member.energy for member in self.members]
        self.average_fitness = np.mean(fitnesses)
        current_best = max(fitnesses)

        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.generations_since_improvement = 0
        else:
            self.generations_since_improvement += 1

        # Age is incremented in reproduction, not here

    def calculate_complexity_penalty(self):
        """Calculate penalty for species complexity - encourages simpler networks"""
        penalty = 0.0

        # Mild penalty for network size
        if self.max_nodes > 10:
            penalty += 0.05 * (self.max_nodes - 10) / 20.0  # Max 5% penalty for very large networks

        if self.max_connections > 15:
            penalty += 0.05 * (self.max_connections - 15) / 30.0  # Max 5% penalty for many connections

        # Small penalty for recurrent connections (they add complexity)
        if self.has_recurrent:
            penalty += 0.1  # 10% penalty for recurrent networks

        # Penalty for species that have been stagnant (complexity without improvement)
        if self.generations_since_improvement > 5:
            penalty += 0.05 * (self.generations_since_improvement - 5) / 10.0  # Max 5% stagnation penalty

        return min(penalty, self.complexity_panelty)  # Cap penalty at 30% to avoid killing complex networks entirely


    def get_champion(self):
        """Get the best creature in this species"""
        if not self.members:
            return None
        return max(self.members, key=lambda c: c.energy)

    def clear_members(self):
        """Clear all members except representative"""
        self.members = []

class NodeGene:
    """Represents a node in the neural network"""
    def __init__(self, node_id: int, node_type: str, activation_func: str = 'tanh'):
        self.node_id = node_id
        self.node_type = node_type  # 'input', 'output', 'hidden'
        self.activation_func = activation_func
        self.value = 0.0
        # For input nodes: which feature index they connect to
        self.input_index = None  # Only used for input nodes
        # For output nodes: which portfolio position they control
        self.output_index = None  # Only used for output nodes

    def activate(self, x: float) -> float:
        """Apply activation function"""
        if self.activation_func == 'tanh':
            return np.tanh(x)
        elif self.activation_func == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_func == 'relu':
            return max(0, x)
        elif self.activation_func == 'linear':
            return x
        else:
            return np.tanh(x)

class ConnectionGene:
    """Represents a connection in the neural network"""
    def __init__(self, from_node: int, to_node: int, weight: float, enabled: bool = True, innovation: int = None):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation if innovation is not None else innovation_tracker.get_innovation_number(from_node, to_node)


class MultiFileDataManager:
    """Manages multiple input files with different start dates"""
    
    def __init__(self, input_dir: str, reload: bool = False):
        self.input_dir = input_dir
        self.reload = reload
        self.file_configs = []  # List of (filename, start_date, df, features, returns)
        self.unified_dates = []  # All unique dates sorted
        self.current_features = None  # Current combined features
        self.current_returns = None   # Current combined returns
        self.feature_names = []
        self.tickers = []
        self.position_names = []
        
    def discover_and_load_files(self):
        """Discover and load all CSV files in input directory"""
        self.write_log("Discovering CSV files in input directory...")
        
        csv_files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.input_dir}")
            
        self.write_log(f"Found {len(csv_files)} CSV files")
        
        for csv_file in sorted(csv_files):
            self._load_single_file(csv_file)
            
        self._create_unified_timeline()
        
    def _load_single_file(self, filepath: str):
        """Load a single CSV file and extract metadata"""
        filename = os.path.basename(filepath)
        self.write_log(f"Loading file: {filename}")
        
        if self.reload:
            self._reload_data(filepath)
            
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        if df.shape[1] % 2 != 0:
            raise ValueError(f"Expected even columns in {filename}. Got {df.shape[1]}.")
            
        # Extract metadata
        num_stocks = df.shape[1] // 2
        columns = df.columns[:num_stocks]
        columns = [col.replace('_price', '').replace('_vol', '') for col in columns]
        tickers = [ticker.replace(' ', '-').strip() for ticker in columns]
        
        # Get returns and volumes
        ret_cols = df.columns[:num_stocks]
        vol_cols = df.columns[num_stocks:]
        returns = df[ret_cols].to_numpy(dtype=float)
        volumes = df[vol_cols].to_numpy(dtype=float)
        features = np.concatenate([returns, volumes], axis=1)
        
        start_date = df.index[0]
        end_date = df.index[-1]
        
        file_config = {
            'filename': filename,
            'filepath': filepath,
            'start_date': start_date,
            'end_date': end_date,
            'df': df,
            'features': features,
            'returns': returns,
            'tickers': tickers,
            'num_stocks': num_stocks,
            'num_features': 2 * num_stocks
        }
        
        self.file_configs.append(file_config)
        
        self.write_log(f"  File: {filename}")
        self.write_log(f"  Date range: {start_date.date()} to {end_date.date()}")
        self.write_log(f"  Stocks: {num_stocks}, Features: {2 * num_stocks}")
        self.write_log(f"  Tickers: {tickers[:5]}{'...' if len(tickers) > 5 else ''}")
        
    def _reload_data(self, filepath):
        """Reload data from external sources (same as original implementation)"""
        # Same implementation as original reload_data method
        data_dir = './data/prices'
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        columns = [col.strip() for col in df.columns]
        num_stocks = len(columns) // 2
        columns = columns[:num_stocks]
        columns = [col.replace('_price', '') for col in columns]
        columns = [col.replace('_vol', '') for col in columns]
        tickers = [ticker.replace(' ', '-').strip() for ticker in columns]

        start = '1950-01-01'
        self._download_price_data(tickers, start, None, data_dir)
        
        # Create new input data (same logic as original)
        dataframes_vol = []
        for ticker in tickers:
            filename = f"{ticker}.csv"
            file_path = Path(data_dir) / filename
            df_vol = pd.read_csv(file_path, usecols=['Date', 'Adj Close', 'Volume'], parse_dates=['Date'])
            df_vol.set_index('Date', inplace=True)
            df_vol.rename(columns={'Adj Close': f"{ticker}_price", 'Volume': f"{ticker}_vol"}, inplace=True)
            dataframes_vol.append(df_vol)

        combined_df_vol = pd.concat(dataframes_vol, axis=1)
        combined_df_vol.fillna(method='ffill', inplace=True)
        combined_df_vol.fillna(method='bfill', inplace=True)
        log_returns_df_vol = np.log(combined_df_vol / (combined_df_vol.shift(1) + 1e-9) + 1e-9)
        log_returns_df_vol = log_returns_df_vol.dropna()

        price_columns = [col for col in log_returns_df_vol.columns if col.endswith('_price')]
        volume_columns = [col for col in log_returns_df_vol.columns if col.endswith('_vol')]
        log_returns_df_vol = log_returns_df_vol[price_columns + volume_columns]

        backup_path = filepath + ".backup"
        os.rename(filepath, backup_path)
        log_returns_df_vol.to_csv(filepath, index_label='Date')
        
    def _download_price_data(self, tickers, start_date, end_date, dest_dir):
        """Download price data using yfinance (same as original)"""
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for ticker in tickers:
            try:
                data = yfin.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False, timeout=40)
                data.index = data.index.date.astype(str)
                data.to_csv(f'{dest_dir}/{ticker}.csv', index_label='Date')
                print(f'Downloaded data for {ticker}')
            except Exception as e:
                print(f'Failed to download data for {ticker}: {e}')
                continue
                
    def _create_unified_timeline(self):
        """Create unified timeline from all files"""
        all_dates = set()
        
        # Collect all dates
        for config in self.file_configs:
            all_dates.update(config['df'].index)
            
        # Sort dates
        self.unified_dates = sorted(list(all_dates))
        
        self.write_log(f"Created unified timeline: {self.unified_dates[0].date()} to {self.unified_dates[-1].date()}")
        self.write_log(f"Total trading days: {len(self.unified_dates)}")
        
        # Sort file configs by start date
        self.file_configs.sort(key=lambda x: x['start_date'])
        
        # Create initial feature names and tickers from earliest file
        earliest_config = self.file_configs[0]
        self.tickers = earliest_config['tickers'].copy()
        self.feature_names = []
        for ticker in self.tickers:
            self.feature_names.append(f"{ticker}_price_return")
        for ticker in self.tickers:
            self.feature_names.append(f"{ticker}_volume_logdiff")
            
        self.position_names = self.tickers.copy()
        self.position_names.append("CASH")
        
    def get_available_files_for_date(self, current_date):
        """Get all files that should be active for the given date"""
        available_files = []
        
        for config in self.file_configs:
            if config['start_date'] <= current_date <= config['end_date']:
                available_files.append(config)
                
        return available_files
        
    def get_features_for_date(self, current_date):
        """Get combined features for a specific date"""
        available_files = self.get_available_files_for_date(current_date)
        
        if not available_files:
            return None, None, None
            
        # Find data for this date in available files
        combined_features = []
        combined_returns = []
        all_tickers = []
        
        for config in available_files:
            if current_date in config['df'].index:
                date_idx = config['df'].index.get_loc(current_date)
                combined_features.append(config['features'][date_idx])
                combined_returns.append(config['returns'][date_idx])
                all_tickers.extend(config['tickers'])
                
        if not combined_features:
            return None, None, None
            
        # Concatenate all features and returns
        final_features = np.concatenate(combined_features) if len(combined_features) > 1 else combined_features[0]
        final_returns = np.concatenate(combined_returns) if len(combined_returns) > 1 else combined_returns[0]
        
        return final_features, final_returns, all_tickers
        
    def get_date_by_index(self, day_index):
        """Get date by index in unified timeline"""
        if day_index >= len(self.unified_dates):
            return None
        return self.unified_dates[day_index]
        
    def get_total_days(self):
        """Get total number of days in unified timeline"""
        return len(self.unified_dates)
        
    def should_trigger_expansion_mutation(self, current_date, prev_date=None):
        """Check if new files became available and should trigger expansion mutations"""
        if prev_date is None:
            return False, 0
            
        prev_files = set(config['filename'] for config in self.get_available_files_for_date(prev_date))
        current_files = set(config['filename'] for config in self.get_available_files_for_date(current_date))
        
        new_files = current_files - prev_files
        
        if new_files:
            self.write_log(f"New files became available at {current_date.date()}: {list(new_files)}")
            return True, len(new_files)
            
        return False, 0
        
    def get_current_feature_dimensions(self, current_date):
        """Get current total feature dimensions for the date"""
        available_files = self.get_available_files_for_date(current_date)
        total_features = sum(config['num_features'] for config in available_files)
        total_stocks = sum(config['num_stocks'] for config in available_files)
        return total_features, total_stocks
        
    def write_log(self, message):
        """Placeholder for logging - will be set by main system"""
        print(message)


class NEATCreature:
    """NEAT-based creature for portfolio prediction with adaptive input handling"""

    def __init__(self, num_features: int, num_portfolio_positions: int, creature_id: int = None, long_only: bool = False):
        self.creature_id = creature_id or random.randint(0, 100000)
        self.num_features = num_features
        self.num_portfolio_positions = num_portfolio_positions
        self.long_only = long_only

        # NEAT genome
        self.nodes = {}  # node_id -> NodeGene
        self.connections = {}  # innovation -> ConnectionGene
        self.next_node_id = 0

        # Network evaluation
        self.evaluation_cache = {}

        # Creature properties
        self.energy = 1.0
        self.portfolio = np.zeros(num_portfolio_positions)
        self.portfolio[-1] = 1.0  # Start with 100% cash
        self.birth_step = 0
        self.fitness_history = []

        # Species information
        self.species_id = None

        # Initialize minimal network
        self._initialize_minimal_network()

    def _initialize_minimal_network(self):
        """Initialize with 1 input and 1 output as per requirement"""
        # Create 1 random input node
        input_node = NodeGene(self.next_node_id, 'input', 'linear')
        input_node.input_index = random.randint(0, max(0, self.num_features - 1))
        self.nodes[self.next_node_id] = input_node
        self.next_node_id += 1

        # Create 1 random output node
        output_node = NodeGene(self.next_node_id, 'output', 'sigmoid')
        output_node.output_index = random.randint(0, self.num_portfolio_positions - 1)
        self.nodes[self.next_node_id] = output_node
        self.next_node_id += 1

        # Connect them
        weight = random.uniform(-2.0, 2.0)
        connection = ConnectionGene(0, 1, weight, True)
        self.connections[connection.innovation] = connection

    def adapt_to_new_features(self, new_num_features: int, new_num_positions: int):
        """Adapt creature to handle new feature dimensions"""
        old_num_features = self.num_features
        old_num_positions = self.num_portfolio_positions
        
        self.num_features = new_num_features
        self.num_portfolio_positions = new_num_positions
        
        # Expand portfolio if needed
        if new_num_positions > old_num_positions:
            old_portfolio = self.portfolio.copy()
            self.portfolio = np.zeros(new_num_positions)
            self.portfolio[:old_num_positions] = old_portfolio
            # Redistribute weights proportionally
            if old_num_positions > 0:
                total_weight = np.sum(old_portfolio)
                if total_weight > 0:
                    self.portfolio[:old_num_positions] = old_portfolio / total_weight
                    
        # Update input node indices to be within valid range
        for node in self.nodes.values():
            if node.node_type == 'input' and node.input_index is not None:
                if node.input_index >= new_num_features:
                    node.input_index = node.input_index % new_num_features if new_num_features > 0 else 0
                    
        # Update output node indices
        for node in self.nodes.values():
            if node.node_type == 'output' and node.output_index is not None:
                if node.output_index >= new_num_positions:
                    node.output_index = node.output_index % new_num_positions if new_num_positions > 0 else 0

    def expansion_mutation(self, new_num_features: int, new_num_positions: int):
        """Special mutation to handle new input/output dimensions"""
        old_num_features = self.num_features
        old_num_positions = self.num_portfolio_positions
        
        # Update dimensions first
        self.adapt_to_new_features(new_num_features, new_num_positions)
        
        # Add new input nodes for new features with higher probability
        if new_num_features > old_num_features:
            # Add 1-3 new input nodes targeting new feature ranges
            num_new_inputs = min(3, random.randint(1, max(1, (new_num_features - old_num_features) // 10)))
            
            for _ in range(num_new_inputs):
                new_input_node = NodeGene(self.next_node_id, 'input', 'linear')
                # Bias towards new feature indices
                if random.random() < 0.7:  # 70% chance to use new features
                    new_input_node.input_index = random.randint(old_num_features, new_num_features - 1)
                else:
                    new_input_node.input_index = random.randint(0, new_num_features - 1)
                    
                self.nodes[self.next_node_id] = new_input_node
                
                # Connect to existing nodes
                existing_non_input = [nid for nid, node in self.nodes.items() 
                                    if node.node_type != 'input' and nid != self.next_node_id]
                if existing_non_input:
                    target_count = min(len(existing_non_input), random.randint(1, 3))
                    targets = random.sample(existing_non_input, target_count)
                    
                    for target in targets:
                        weight = random.uniform(-1.5, 1.5)
                        conn = ConnectionGene(self.next_node_id, target, weight, True)
                        self.connections[conn.innovation] = conn
                        
                self.next_node_id += 1
                
        # Add new output nodes for new positions
        if new_num_positions > old_num_positions:
            num_new_outputs = min(2, random.randint(1, max(1, (new_num_positions - old_num_positions) // 5)))
            
            for _ in range(num_new_outputs):
                new_output_node = NodeGene(self.next_node_id, 'output', 'sigmoid')
                # Bias towards new position indices
                if random.random() < 0.7:  # 70% chance to use new positions
                    new_output_node.output_index = random.randint(old_num_positions, new_num_positions - 1)
                else:
                    new_output_node.output_index = random.randint(0, new_num_positions - 1)
                    
                self.nodes[self.next_node_id] = new_output_node
                
                # Connect from existing nodes
                existing_non_output = [nid for nid, node in self.nodes.items() 
                                     if node.node_type != 'output' and nid != self.next_node_id]
                if existing_non_output:
                    source_count = min(len(existing_non_output), random.randint(1, 3))
                    sources = random.sample(existing_non_output, source_count)
                    
                    for source in sources:
                        weight = random.uniform(-1.5, 1.5)
                        conn = ConnectionGene(source, self.next_node_id, weight, True)
                        self.connections[conn.innovation] = conn
                        
                self.next_node_id += 1

    def get_input_feature_name(self, input_index: int, feature_names: List[str]) -> str:
        """Get human readable name for input feature"""
        if input_index < len(feature_names):
            return feature_names[input_index]
        return f"Feature_{input_index}"

    def get_output_position_name(self, output_index: int, position_names: List[str]) -> str:
        """Get human readable name for output position"""
        if output_index < len(position_names):
            return position_names[output_index]
        return f"Position_{output_index}"

    def get_network_structure_detailed(self, feature_names: List[str], position_names: List[str]) -> str:
        """Get detailed network structure with feature and position names"""
        structure = []

        # Input nodes
        input_nodes = [(nid, node) for nid, node in self.nodes.items() if node.node_type == 'input']
        if input_nodes:
            structure.append("INPUTS:")
            for nid, node in input_nodes:
                feature_name = self.get_input_feature_name(node.input_index, feature_names)
                structure.append(f"  Node{nid}: {feature_name}")

        # Hidden nodes
        hidden_nodes = [(nid, node) for nid, node in self.nodes.items() if node.node_type == 'hidden']
        if hidden_nodes:
            structure.append("HIDDEN:")
            for nid, node in hidden_nodes:
                structure.append(f"  Node{nid}: {node.activation_func}")

        # Output nodes
        output_nodes = [(nid, node) for nid, node in self.nodes.items() if node.node_type == 'output']
        if output_nodes:
            structure.append("OUTPUTS:")
            for nid, node in output_nodes:
                position_name = self.get_output_position_name(node.output_index, position_names)
                structure.append(f"  Node{nid}: {position_name}")

        # Connections
        enabled_connections = [conn for conn in self.connections.values() if conn.enabled]
        if enabled_connections:
            structure.append("CONNECTIONS:")
            for conn in enabled_connections:
                conn_type = "(RECURRENT)" if conn.to_node <= conn.from_node else ""
                structure.append(f"  Node{conn.from_node}→Node{conn.to_node}: w={conn.weight:.3f} {conn_type}")

        return "\n".join(structure)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through the NEAT network with safe feature access"""
        # Reset node values
        for node in self.nodes.values():
            node.value = 0.0

        # Set input values with bounds checking
        for node_id, node in self.nodes.items():
            if node.node_type == 'input' and node.input_index is not None:
                if node.input_index < len(features):
                    node.value = features[node.input_index]
                else:
                    # Use a default value or wrap around
                    node.value = 0.0

        # Topological sort for network evaluation (handles recurrent connections with max iterations)
        max_iterations = 15  # Increased for more complex recurrent networks

        for iteration in range(max_iterations):
            prev_values = {nid: node.value for nid, node in self.nodes.items()}

            # Update all non-input nodes
            for node_id, node in self.nodes.items():
                if node.node_type == 'input':
                    continue

                # Sum inputs from all enabled connections
                input_sum = 0.0
                for connection in self.connections.values():
                    if connection.enabled and connection.to_node == node_id:
                        if connection.from_node in self.nodes:
                            input_sum += self.nodes[connection.from_node].value * connection.weight

                # Apply activation
                node.value = node.activate(input_sum)

            # Check convergence for recurrent networks
            if iteration > 0:
                converged = True
                for nid in self.nodes:
                    if abs(self.nodes[nid].value - prev_values[nid]) > 1e-6:
                        converged = False
                        break
                if converged:
                    break

        # Collect outputs with bounds checking
        portfolio = np.copy(self.portfolio)  # Start with current portfolio

        for node_id, node in self.nodes.items():
            if node.node_type == 'output' and node.output_index is not None:
                if node.output_index < len(portfolio):
                    if self.long_only:
                        # For long-only, output is the target weight (0 to 1)
                        portfolio[node.output_index] = max(0, min(1, node.value))
                    else:
                        # For long/short, output can be negative
                        portfolio[node.output_index] = np.clip(node.value, -1, 1)

        # Normalize portfolio weights
        if self.long_only:
            total_weight = np.sum(portfolio)
            if total_weight > 0:
                portfolio = portfolio / total_weight
            else:
                portfolio = np.zeros_like(portfolio)
                portfolio[-1] = 1.0  # All cash if no positive weights
        else:
            # For long/short, normalize by sum of absolute weights
            total_abs_weight = np.sum(np.abs(portfolio))
            if total_abs_weight > 0:
                portfolio = portfolio / total_abs_weight
            else:
                portfolio = np.zeros_like(portfolio)
                portfolio[-1] = 1.0  # All cash

        return portfolio

    def add_node_mutation(self):
        """Add a new node by splitting an existing connection"""
        if not self.connections:
            return

        # Choose a random enabled connection
        enabled_connections = [c for c in self.connections.values() if c.enabled]
        if not enabled_connections:
            return

        connection = random.choice(enabled_connections)

        # Disable the connection
        connection.enabled = False

        # Create new hidden node
        activation_funcs = ['tanh', 'sigmoid', 'relu']
        # Bias towards more expressive activations for complex structures
        activation_weights = [0.5, 0.3, 0.2]
        new_node = NodeGene(self.next_node_id, 'hidden',
                          np.random.choice(activation_funcs, p=activation_weights))
        self.nodes[self.next_node_id] = new_node

        # Create two new connections
        conn1 = ConnectionGene(connection.from_node, self.next_node_id, 1.0, True)
        conn2 = ConnectionGene(self.next_node_id, connection.to_node, connection.weight, True)

        self.connections[conn1.innovation] = conn1
        self.connections[conn2.innovation] = conn2

        self.next_node_id += 1

    def add_connection_mutation(self):
        """Add a new connection between existing nodes - encourages recurrent connections"""
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return

        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            from_node = random.choice(node_ids)
            to_node = random.choice(node_ids)

            # Check if connection already exists
            connection_exists = any(
                c.from_node == from_node and c.to_node == to_node
                for c in self.connections.values()
            )

            # Prevent self-connection but allow recurrent connections
            if from_node != to_node and not connection_exists:
                # Bias towards creating recurrent connections (innovation protection)
                weight_scale = 1.5 if to_node <= from_node else 1.0  # Slightly stronger weights for recurrent
                weight = random.uniform(-2.0 * weight_scale, 2.0 * weight_scale)
                connection = ConnectionGene(from_node, to_node, weight, True)
                self.connections[connection.innovation] = connection
                break

            attempts += 1

    def add_input_mutation(self):
        """Add a new input node"""
        new_node = NodeGene(self.next_node_id, 'input', 'linear')
        new_node.input_index = random.randint(0, max(0, self.num_features - 1))
        self.nodes[self.next_node_id] = new_node

        # Connect to multiple existing non-input nodes for richer connectivity
        non_input_nodes = [nid for nid, node in self.nodes.items() if node.node_type != 'input']
        if non_input_nodes:
            # Connect to 1-3 nodes
            num_connections = min(len(non_input_nodes), random.randint(1, 3))
            target_nodes = random.sample(non_input_nodes, num_connections)

            for target_node in target_nodes:
                weight = random.uniform(-2.0, 2.0)
                connection = ConnectionGene(self.next_node_id, target_node, weight, True)
                self.connections[connection.innovation] = connection

        self.next_node_id += 1

    def add_output_mutation(self):
        """Add a new output node"""
        new_node = NodeGene(self.next_node_id, 'output', 'sigmoid')
        new_node.output_index = random.randint(0, self.num_portfolio_positions - 1)
        self.nodes[self.next_node_id] = new_node

        # Connect from multiple existing non-output nodes for richer connectivity
        non_output_nodes = [nid for nid, node in self.nodes.items() if node.node_type != 'output']
        if non_output_nodes:
            # Connect from 1-3 nodes
            num_connections = min(len(non_output_nodes), random.randint(1, 3))
            source_nodes = random.sample(non_output_nodes, num_connections)

            for source_node in source_nodes:
                weight = random.uniform(-2.0, 2.0)
                connection = ConnectionGene(source_node, self.next_node_id, weight, True)
                self.connections[connection.innovation] = connection

        self.next_node_id += 1

    def add_recurrent_connection_mutation(self):
        """Specifically add recurrent connections to encourage complex behavior"""
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return

        attempts = 0
        max_attempts = 50

        while attempts < max_attempts:
            from_node = random.choice(node_ids)
            to_node = random.choice(node_ids)

            # Focus on recurrent connections (to_node <= from_node)
            if to_node <= from_node and from_node != to_node:
                # Check if connection already exists
                connection_exists = any(
                    c.from_node == from_node and c.to_node == to_node
                    for c in self.connections.values()
                )

                if not connection_exists:
                    # Recurrent connections get special weight initialization
                    weight = random.uniform(-1.0, 1.0)  # Smaller weights for stability
                    connection = ConnectionGene(from_node, to_node, weight, True)
                    self.connections[connection.innovation] = connection
                    break

            attempts += 1

    def mutate_weights(self, mutation_rate: float = 0.8, mutation_strength: float = 0.1):
        """Mutate connection weights"""
        for connection in self.connections.values():
            if random.random() < mutation_rate:
                if random.random() < 0.1:  # 10% chance of complete random reset
                    connection.weight = random.uniform(-2.0, 2.0)
                else:  # 90% chance of small perturbation
                    connection.weight += random.uniform(-mutation_strength, mutation_strength)

    def mutate(self, mutation_rates: dict = None):
        """Apply various mutations with emphasis on complex structures"""
        if mutation_rates is None:
            mutation_rates = {
                'add_node': 0.05,  # Increased for more complexity
                'add_connection': 0.08,  # Increased
                'add_recurrent_connection': 0.04,  # New mutation type
                'add_input': 0.03,
                'add_output': 0.03,
                'weight_mutation': 0.8
            }

        if random.random() < mutation_rates['add_node']:
            self.add_node_mutation()

        if random.random() < mutation_rates['add_connection']:
            self.add_connection_mutation()

        if random.random() < mutation_rates['add_recurrent_connection']:
            self.add_recurrent_connection_mutation()

        if random.random() < mutation_rates['add_input']:
            self.add_input_mutation()

        if random.random() < mutation_rates['add_output']:
            self.add_output_mutation()

        if random.random() < mutation_rates['weight_mutation']:
            self.mutate_weights()

    def get_network_size(self):
        """Get network complexity metrics"""
        recurrent_count = sum(1 for c in self.connections.values()
                            if c.enabled and c.to_node <= c.from_node and c.to_node != c.from_node)

        return {
            'nodes': len(self.nodes),
            'connections': len([c for c in self.connections.values() if c.enabled]),
            'inputs': len([n for n in self.nodes.values() if n.node_type == 'input']),
            'outputs': len([n for n in self.nodes.values() if n.node_type == 'output']),
            'hidden': len([n for n in self.nodes.values() if n.node_type == 'hidden']),
            'recurrent': recurrent_count
        }

    def calculate_compatibility_distance(self, other, c1=1.0, c2=1.0, c3=0.4):
        """Calculate compatibility distance for speciation"""
        # Get all innovations from both genomes
        innovations1 = set(self.connections.keys())
        innovations2 = set(other.connections.keys())

        all_innovations = innovations1.union(innovations2)
        if not all_innovations:
            return 0.0

        max_innovation = max(all_innovations)
        min_innovation = min(all_innovations)

        # Count disjoint and excess genes
        disjoint = 0
        excess = 0
        matching = 0
        weight_diff_sum = 0.0

        for innovation in all_innovations:
            in_1 = innovation in innovations1
            in_2 = innovation in innovations2

            if in_1 and in_2:
                # Matching gene
                matching += 1
                weight_diff_sum += abs(self.connections[innovation].weight -
                                     other.connections[innovation].weight)
            else:
                # Non-matching gene
                if innovation == max_innovation:
                    excess += 1
                else:
                    disjoint += 1

        # Calculate distance
        N = max(len(innovations1), len(innovations2), 1)  # Normalizing factor

        avg_weight_diff = weight_diff_sum / max(matching, 1)

        distance = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_weight_diff)

        return distance


class NEATTradingSystem:
    """NEAT-based trading system with multi-file support and elite-based reproduction"""

    def __init__(self, commission_rate=0.001, max_population=100000, log_file="output.log",
                 output_json_dir=None, long_only=False, checkpoint_dir=None, checkpoint_interval=None, 
                 complexity_panelty=1.0, input_dir=None, reload=False, expansion_increment=2000,
                 elite_reproduction_rate=0.15, max_offspring_per_parent=3):
        self.commission_rate = commission_rate
        self.max_population = max_population
        self.initial_max_population = max_population  # Store original max population
        self.expansion_increment = expansion_increment  # How much to increase population per new file
        self.elite_reproduction_rate = elite_reproduction_rate  # Top % that can reproduce
        self.max_offspring_per_parent = max_offspring_per_parent  # Max offspring per elite parent
        self.population = []

        self.dead_count = 0
        self.dead_meta = []

        self.current_step = 0

        # Species management
        self.species = {}  # species_id -> Species
        self.next_species_id = 0
        self.compatibility_threshold = 3.0

        # Innovation protection parameters
        self.young_age_threshold = 5  # Young species in reproduction cycles
        self.old_age_threshold = 20   # Old species threshold
        self.young_age_bonus = 0.3
        self.stagnation_threshold = 10  # Stagnation in reproduction cycles

        # Multi-file data management
        self.data_manager = MultiFileDataManager(input_dir, reload) if input_dir else None
        self.files_seen = set()  # Track which files we've already processed expansions for
        
        # Dynamic dimensions tracking
        self.current_num_features = 0
        self.current_num_stocks = 0
        self.prev_date = None
        
        # Data holders (maintained for compatibility)
        self.stock_data = None
        self.features = None
        self.returns = None
        self.num_stocks = None
        self.num_features = None
        self.tickers = []

        # Feature and position names for logging
        self.feature_names = []
        self.position_names = []

        self.log_file = log_file
        self.long_only = long_only
        self.output_json_dir = output_json_dir

        self.selected_creature = None
        self.selected_rank = None
        self.asset = 1.0
        self.asset_history = [self.asset]

        self.output_records = []
        self.output_file = None
        self._output_file_initialized = False

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.output_json_dir:
            os.makedirs(self.output_json_dir, exist_ok=True)

        self.complexity_panelty = complexity_panelty

        # PERSISTENT PROCESS POOL - Initialize once
        self.process_pool = None
        self.max_workers = min(8, os.cpu_count() or 4)
        self.batch_size = None  # Will be set based on population size

    def __enter__(self):
        """Context manager entry - initialize process pool"""
        self._initialize_process_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup process pool"""
        self._cleanup_process_pool()

    def _initialize_process_pool(self):
        """Initialize persistent process pool"""
        if self.process_pool is None:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            print(f"Initialized persistent process pool with {self.max_workers} workers")

    def _cleanup_process_pool(self):
        """Cleanup persistent process pool"""
        if self.process_pool is not None:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
            print("Process pool shut down")

    def prepare_creature_for_process(self, creature):
        """Prepare creature data for process-based evaluation"""
        # Serialize node data
        nodes_data = {}
        for node_id, node in creature.nodes.items():
            nodes_data[node_id] = {
                'type': node.node_type,
                'activation': node.activation_func,
                'input_index': node.input_index,
                'output_index': node.output_index
            }

        # Serialize connection data
        connections_data = {}
        for innovation, conn in creature.connections.items():
            connections_data[innovation] = {
                'from_node': conn.from_node,
                'to_node': conn.to_node,
                'weight': conn.weight,
                'enabled': conn.enabled
            }

        return (creature.creature_id, nodes_data, connections_data,
                creature.portfolio.copy(), creature.energy, creature.birth_step)

    def assign_to_species(self, creature):
        """Assign a creature to a species based on compatibility"""
        best_species = None
        min_distance = float('inf')

        # Try to assign to existing species
        for species in self.species.values():
            if not species.representative:
                continue

            distance = creature.calculate_compatibility_distance(species.representative)
            if distance < self.compatibility_threshold and distance < min_distance:
                min_distance = distance
                best_species = species

        if best_species:
            best_species.add_member(creature)
        else:
            # Create new species
            new_species = Species(creature, self.next_species_id, self.complexity_panelty)
            self.species[self.next_species_id] = new_species
            creature.species_id = self.next_species_id
            self.next_species_id += 1
            self.write_log(f"Created new species {new_species.species_id} with creature {creature.creature_id}")

    def speciate_population(self):
        """Assign all creatures to species"""
        # Clear existing species memberships
        for species in self.species.values():
            species.clear_members()

        # Assign each creature to a species
        for creature in self.population:
            self.assign_to_species(creature)

        # Remove empty species and update representatives
        empty_species = []
        for species_id, species in self.species.items():
            if not species.members:
                empty_species.append(species_id)
            else:
                # Update representative (choose best member or keep current if still in species)
                if species.representative not in species.members:
                    species.representative = species.get_champion()

                # Update species statistics
                species.update_fitness_stats()

        # Remove empty species
        for species_id in empty_species:
            del self.species[species_id]

    def calculate_adjusted_fitness(self):
        """Calculate adjusted fitness for each creature based on species"""
        for species in self.species.values():
            if not species.members:
                continue

            species_size = len(species.members)
            complexity_penalty = species.calculate_complexity_penalty()

            # Age-based bonuses and penalties (based on reproduction cycles)
            age_modifier = 1.0
            if species.age < self.young_age_threshold:
                age_modifier += self.young_age_bonus  # Protect young species
            elif species.generations_since_improvement > self.stagnation_threshold:
                age_modifier *= 0.7  # Less aggressive penalty

            for creature in species.members:
                # Base adjusted fitness (sharing)
                adjusted_fitness = creature.energy / species_size

                # Apply complexity penalty
                adjusted_fitness *= (1.0 - complexity_penalty)

                # Apply age modifier
                adjusted_fitness *= age_modifier

                # Store adjusted fitness
                creature.adjusted_fitness = adjusted_fitness

    def elite_based_reproduction(self):
        """FIXED Elite-based reproduction system - only adds offspring, doesn't remove unless over capacity"""
        if len(self.population) < 10:  # Need minimum population
            self.write_log("Population too small for elite reproduction")
            return

        self.write_log(f"\n=== ELITE-BASED REPRODUCTION EVENT (Step {self.current_step}) ===")
        self.write_log(f"Pre-reproduction: {len(self.species)} species, {len(self.population)} creatures, max_pop: {self.max_population}")
        
        # Store selected creature
        selected_creature_before = self.selected_creature
        selected_creature_id = selected_creature_before.creature_id if selected_creature_before else None
        self.write_log(f"Selected creature before reproduction: {selected_creature_id}")

        # Increment species ages
        for species in self.species.values():
            species.age += 1

        # Get elite creatures from each species
        elite_parents = []
        species_elites = {}
        
        for species_id, species in self.species.items():
            if not species.members:
                continue
                
            # Sort species members by adjusted fitness
            sorted_members = sorted(species.members, key=lambda c: getattr(c, 'adjusted_fitness', c.energy), reverse=True)
            
            # Select top elite percentage from each species
            elite_count = max(1, int(len(sorted_members) * self.elite_reproduction_rate))
            species_elite = sorted_members[:elite_count]
            
            elite_parents.extend(species_elite)
            species_elites[species_id] = species_elite
            
            self.write_log(f"  Species {species_id}: {len(species_elite)} elites from {len(sorted_members)} members")

        total_elites = len(elite_parents)
        self.write_log(f"  Total elite parents: {total_elites}")

        # FIXED: Calculate offspring allocation more reasonably
        # Target much smaller reproduction - only 10-20% of current population as new offspring
        current_pop_size = len(self.population)
        target_offspring_ratio = 0.15  # 15% of current population as new offspring
        target_offspring_count = max(10, int(current_pop_size * target_offspring_ratio))
        
        # Limit offspring per elite to prevent explosion
        if total_elites > 0:
            offspring_per_elite = min(self.max_offspring_per_parent, max(1, target_offspring_count // total_elites))
        else:
            offspring_per_elite = 0
        
        actual_offspring_count = total_elites * offspring_per_elite
        
        self.write_log(f"  Target offspring: {target_offspring_count}, actual: {actual_offspring_count}")
        self.write_log(f"  Offspring per elite: {offspring_per_elite}")

        # Preserve selected creature
        preserved_selected = None
        if selected_creature_before and selected_creature_before in self.population:
            preserved_selected = self.clone_creature(selected_creature_before)
            preserved_selected.creature_id = selected_creature_before.creature_id
            preserved_selected.energy = selected_creature_before.energy
            preserved_selected.portfolio = selected_creature_before.portfolio.copy()
            preserved_selected.birth_step = selected_creature_before.birth_step
            preserved_selected.fitness_history = selected_creature_before.fitness_history.copy()
            preserved_selected.adapt_to_new_features(self.current_num_features, self.current_num_stocks + 1)
            self.write_log(f"  Preserved selected creature {selected_creature_id}")

        # Generate offspring from elite parents
        all_offspring = []
        
        for parent in elite_parents:
            for _ in range(offspring_per_elite):
                # 70% chance of crossover with another elite, 30% chance of cloning with mutation
                if len(elite_parents) > 1 and random.random() < 0.7:
                    # Crossover with another elite parent
                    other_parent = random.choice([p for p in elite_parents if p != parent])
                    child = self.crossover(parent, other_parent)
                else:
                    # Clone and mutate
                    child = self.clone_creature(parent)
                
                # Apply mutation
                child.mutate()
                child.birth_step = self.current_step
                child.energy = parent.energy * random.uniform(0.9, 1.1)  # Small energy variation
                child.adapt_to_new_features(self.current_num_features, self.current_num_stocks + 1)
                
                all_offspring.append(child)

        self.write_log(f"  Created {len(all_offspring)} offspring from elite reproduction")

        # FIXED: Population management - only remove if we exceed max_population
        total_after_addition = len(self.population) + len(all_offspring)
        if preserved_selected:
            # Remove any duplicate of preserved creature first
            self.population = [c for c in self.population if c.creature_id != selected_creature_id]
            total_after_addition = len(self.population) + len(all_offspring) + 1

        # Only remove creatures if we exceed max_population
        creatures_removed = 0
        if total_after_addition > self.max_population:
            # Remove the worst performers to make room
            excess = total_after_addition - self.max_population
            
            # Sort by adjusted fitness and remove worst
            sorted_pop = sorted(self.population, key=lambda c: getattr(c, 'adjusted_fitness', c.energy), reverse=True)
            
            # Keep the best, remove the worst
            survivors = sorted_pop[:-excess] if excess < len(sorted_pop) else [sorted_pop[0]]  # Keep at least 1
            creatures_removed = len(self.population) - len(survivors)
            
            # Update population
            for creature in self.population:
                if creature not in survivors:
                    self.dead_count += 1
                    
            self.population = survivors
            self.write_log(f"  Removed {creatures_removed} worst performers to make room for offspring")
        else:
            self.write_log(f"  No creatures removed - sufficient capacity (pop: {len(self.population)}, adding: {len(all_offspring)}, max: {self.max_population})")

        # Add all offspring to population
        self.population.extend(all_offspring)
        
        # Add preserved selected creature back
        if preserved_selected:
            self.population.append(preserved_selected)
            self.selected_creature = preserved_selected

        # Re-speciate
        self.speciate_population()
        self.calculate_adjusted_fitness()

        # Verify selected creature still exists
        selected_still_exists = any(c.creature_id == selected_creature_id for c in self.population) if selected_creature_id else False
        self.write_log(f"Post-reproduction: {len(self.species)} species, {len(self.population)} creatures")
        self.write_log(f"Added {len(all_offspring)} offspring, removed {creatures_removed} creatures")
        self.write_log(f"Selected creature {selected_creature_id} still exists: {selected_still_exists}")
        self.write_log("========================================\n")

    def clone_creature(self, creature):
        """Create an exact clone of a creature"""
        clone = NEATCreature(
            num_features=creature.num_features,
            num_portfolio_positions=creature.num_portfolio_positions,
            long_only=self.long_only
        )

        # Copy genome
        clone.nodes = {}
        for node_id, node in creature.nodes.items():
            new_node = NodeGene(node.node_id, node.node_type, node.activation_func)
            new_node.input_index = node.input_index
            new_node.output_index = node.output_index
            clone.nodes[node_id] = new_node

        clone.connections = {}
        for innovation, conn in creature.connections.items():
            new_conn = ConnectionGene(
                conn.from_node, conn.to_node, conn.weight, conn.enabled, innovation
            )
            clone.connections[innovation] = new_conn

        clone.next_node_id = creature.next_node_id
        clone.portfolio = creature.portfolio.copy()

        return clone

    def save_checkpoint(self):
        """Save checkpoint with species information"""
        if not self.checkpoint_dir:
            return

        # Save species information
        species_data = {}
        for species_id, species in self.species.items():
            species_data[species_id] = {
                "species_id": species.species_id,
                "representative_id": species.representative.creature_id if species.representative else None,
                "best_fitness": species.best_fitness,
                "average_fitness": species.average_fitness,
                "generations_since_improvement": species.generations_since_improvement,
                "age": species.age,
                "max_nodes": species.max_nodes,
                "max_connections": species.max_connections,
                "has_recurrent": species.has_recurrent
            }

        data = {
            "current_step": self.current_step,
            "selected_rank": self.selected_rank,
            "selected_creature_id": (
                self.selected_creature.creature_id
                if self.selected_creature is not None else None
            ),
            "global_asset": self.asset,
            "compatibility_threshold": self.compatibility_threshold,
            "next_species_id": self.next_species_id,
            "species_data": species_data,
            "current_num_features": self.current_num_features,
            "current_num_stocks": self.current_num_stocks,
            "max_population": self.max_population,
            "initial_max_population": self.initial_max_population,
            "files_seen": list(self.files_seen),
            "elite_reproduction_rate": self.elite_reproduction_rate,
            "max_offspring_per_parent": self.max_offspring_per_parent,
            "creatures": []
        }

        for c in self.population:
            c_dict = {
                "creature_id": c.creature_id,
                "energy": c.energy,
                "birth_step": c.birth_step,
                "portfolio": c.portfolio.tolist(),
                "fitness_history": c.fitness_history,
                "species_id": getattr(c, 'species_id', None),
                "adjusted_fitness": getattr(c, 'adjusted_fitness', 0.0),
                "num_features": c.num_features,
                "num_portfolio_positions": c.num_portfolio_positions,
                "nodes": {nid: {
                    "node_type": node.node_type,
                    "activation_func": node.activation_func,
                    "input_index": node.input_index,
                    "output_index": node.output_index
                } for nid, node in c.nodes.items()},
                "connections": {innov: {
                    "from_node": conn.from_node,
                    "to_node": conn.to_node,
                    "weight": conn.weight,
                    "enabled": conn.enabled
                } for innov, conn in c.connections.items()},
                "next_node_id": c.next_node_id
            }
            data["creatures"].append(c_dict)

        filename = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.current_step}.pt")
        torch.save(data, filename)
        self.write_log(f"[Checkpoint saved at step {self.current_step} → {filename}]")

    def load_checkpoint(self):
        """Load checkpoint with species information and multi-file support"""
        if not self.checkpoint_dir:
            return

        files = os.listdir(self.checkpoint_dir)
        pattern = re.compile(r"^checkpoint_step_(\d+)\.pt$")
        best_step = -1
        best_file = None

        for fn in files:
            m = pattern.match(fn)
            if not m:
                continue
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_file = os.path.join(self.checkpoint_dir, fn)

        if best_file is None:
            self.write_log(f"[No checkpoint files found in {self.checkpoint_dir}]")
            return

        self.write_log(f"[Loading checkpoint from {best_file}]")
        data = torch.load(best_file, map_location="cpu", weights_only=False)

        self.current_step = data.get("current_step", 0)
        self.selected_rank = data.get("selected_rank", None)
        self.asset = data.get("global_asset", 1.0)
        self.compatibility_threshold = data.get("compatibility_threshold", 3.0)
        self.next_species_id = data.get("next_species_id", 0)
        self.current_num_features = data.get("current_num_features", 0)
        self.current_num_stocks = data.get("current_num_stocks", 0)
        self.max_population = data.get("max_population", self.max_population)
        self.initial_max_population = data.get("initial_max_population", self.initial_max_population)
        self.files_seen = set(data.get("files_seen", []))
        self.elite_reproduction_rate = data.get("elite_reproduction_rate", self.elite_reproduction_rate)
        self.max_offspring_per_parent = data.get("max_offspring_per_parent", self.max_offspring_per_parent)
        sel_id = data.get("selected_creature_id", None)

        # Rebuild population
        new_pop = []
        for cdata in data["creatures"]:
            c = NEATCreature(
                num_features=cdata.get("num_features", self.current_num_features),
                num_portfolio_positions=cdata.get("num_portfolio_positions", self.current_num_stocks + 1),
                creature_id=cdata["creature_id"],
                long_only=self.long_only
            )

            # Restore genome
            c.nodes = {}
            for nid_str, node_data in cdata["nodes"].items():
                nid = int(nid_str)
                node = NodeGene(nid, node_data["node_type"], node_data["activation_func"])
                node.input_index = node_data["input_index"]
                node.output_index = node_data["output_index"]
                c.nodes[nid] = node

            c.connections = {}
            for innov_str, conn_data in cdata["connections"].items():
                innov = int(innov_str)
                conn = ConnectionGene(
                    conn_data["from_node"],
                    conn_data["to_node"],
                    conn_data["weight"],
                    conn_data["enabled"],
                    innov
                )
                c.connections[innov] = conn

            c.next_node_id = cdata["next_node_id"]
            c.energy = cdata["energy"]
            c.birth_step = cdata["birth_step"]
            c.portfolio = np.array(cdata["portfolio"])
            c.fitness_history = cdata["fitness_history"]
            c.species_id = cdata.get("species_id", None)
            c.adjusted_fitness = cdata.get("adjusted_fitness", 0.0)
            new_pop.append(c)

        self.population = new_pop

        # Restore species
        self.species = {}
        species_data = data.get("species_data", {})
        for species_id_str, sdata in species_data.items():
            species_id = int(species_id_str)

            # Find representative
            representative = None
            rep_id = sdata.get("representative_id")
            if rep_id:
                for c in self.population:
                    if c.creature_id == rep_id:
                        representative = c
                        break

            if representative:
                species = Species(representative, species_id, self.complexity_panelty)
                species.best_fitness = sdata.get("best_fitness", 0.0)
                species.average_fitness = sdata.get("average_fitness", 0.0)
                species.generations_since_improvement = sdata.get("generations_since_improvement", 0)
                species.age = sdata.get("age", 0)
                species.max_nodes = sdata.get("max_nodes", 2)
                species.max_connections = sdata.get("max_connections", 1)
                species.has_recurrent = sdata.get("has_recurrent", False)

                self.species[species_id] = species

        # Restore selected creature
        self.selected_creature = None
        if sel_id is not None:
            for c in self.population:
                if c.creature_id == sel_id:
                    self.selected_creature = c
                    break

        # Re-speciate to ensure consistency
        self.speciate_population()

        self.write_log(f"[Loaded checkpoint at step {self.current_step}, "
                       f"population size {len(self.population)}, "
                       f"species count {len(self.species)}, "
                       f"max_population: {self.max_population}, "
                       f"elite_rate: {self.elite_reproduction_rate}, "
                       f"max_offspring: {self.max_offspring_per_parent}, "
                       f"current dimensions: {self.current_num_features}F x {self.current_num_stocks}S, "
                       f"selected_creature_id={sel_id}]")

    def update_selected_creature(self):
        """Update selected creature from top 1%"""
        if not self.population:
            self.selected_creature = None
            self.selected_rank = None
            return

        sorted_pop = sorted(self.population, key=lambda c: c.energy, reverse=True)
        top_n = max(1, int(len(sorted_pop) * 0.01))
        top_group = sorted_pop[:top_n]

        if self.selected_creature in top_group:
            self.selected_rank = sorted_pop.index(self.selected_creature)
            return

        oldest_in_top = max(
            top_group,
            key=lambda c: self.current_step - c.birth_step
        )

        self.selected_creature = oldest_in_top
        self.selected_rank = sorted_pop.index(oldest_in_top)

    def load_data(self, input_path, reload=False):
        """Load data from single file or multiple files"""
        if self.data_manager is None:
            # Single file mode (backward compatibility)
            self._load_single_file_data(input_path, reload)
        else:
            # Multi-file mode
            self.data_manager.write_log = self.write_log
            self.data_manager.discover_and_load_files()
            
            # Set initial dimensions from earliest file
            if self.data_manager.file_configs:
                earliest_config = self.data_manager.file_configs[0]
                self.current_num_features = earliest_config['num_features']
                self.current_num_stocks = earliest_config['num_stocks']
                self.feature_names = self.data_manager.feature_names
                self.position_names = self.data_manager.position_names
                self.tickers = self.data_manager.tickers
                
                # Initialize files_seen with the earliest file
                self.files_seen.add(earliest_config['filename'])
                
                self.write_log(f"Multi-file mode initialized with {len(self.data_manager.file_configs)} files")
                self.write_log(f"Initial dimensions: {self.current_num_features} features, {self.current_num_stocks} stocks")
                self.write_log(f"Unified timeline: {self.data_manager.get_total_days()} days")
                self.write_log(f"Max population will expand from {self.initial_max_population} by {self.expansion_increment} per new file")
                self.write_log(f"Elite reproduction: {self.elite_reproduction_rate*100:.1f}% elites, max {self.max_offspring_per_parent} offspring per parent")
            
        # Set batch size based on population
        if self.batch_size is None and len(self.population) > 0:
            self.batch_size = max(25, len(self.population) // self.max_workers)
        elif self.batch_size is None:
            self.batch_size = 50  # Default batch size

    def _load_single_file_data(self, filepath, reload):
        """Load single file data (original implementation)"""
        if reload:
            self.reload_data(filepath)

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if df.shape[1] % 2 != 0:
            raise ValueError(
                f"Expected an even number of columns (returns + vol_logdiff). Got {df.shape[1]}."
            )

        total_cols = df.shape[1]
        num_stocks = total_cols // 2
        columns = df.columns[:num_stocks]
        columns = [col.replace('_price', '') for col in columns]
        columns = [col.replace('_vol', '') for col in columns]
        self.tickers = [ticker.replace(' ', '-').strip() for ticker in columns]

        ret_cols = df.columns[:num_stocks]
        vol_cols = df.columns[num_stocks:]

        returns = df[ret_cols].to_numpy(dtype=float)
        volumes = df[vol_cols].to_numpy(dtype=float)
        features = np.concatenate([returns, volumes], axis=1)

        self.stock_data = df
        self.num_stocks = num_stocks
        self.num_features = 2 * num_stocks  # price + volume features
        self.returns = returns
        self.features = features
        
        # Set current dimensions
        self.current_num_features = self.num_features
        self.current_num_stocks = self.num_stocks

        # Create feature names and position names for logging
        self.feature_names = []
        for ticker in self.tickers:
            self.feature_names.append(f"{ticker}_price_return")
        for ticker in self.tickers:
            self.feature_names.append(f"{ticker}_volume_logdiff")

        self.position_names = self.tickers.copy()
        self.position_names.append("CASH")

        self.write_log(
            f"Loaded data with {self.num_stocks} stocks (features per day: {self.num_features}) "
            f"from {df.index[0].date()} to {df.index[-1].date()}"
        )
        self.write_log(f"Total trading days: {len(df)}")

    def reload_data(self, filepath):
        """Reload and update data from external sources (original implementation)"""
        data_dir = './data/prices'
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        columns = [col.strip() for col in df.columns]
        num_stocks = len(columns) // 2
        columns = columns[:num_stocks]
        columns = [col.replace('_price', '') for col in columns]
        columns = [col.replace('_vol', '') for col in columns]
        tickers = [ticker.replace(' ', '-').strip() for ticker in columns]

        start = '1950-01-01'
        self.download_price_data(tickers, start, None, data_dir)
        self.write_log(f"Downloaded price data for {len(columns)} stocks")

        # Create new input data
        dataframes_vol = []
        for ticker in tickers:
            filename = f"{ticker}.csv"
            file_path = Path(data_dir) / filename
            df_vol = pd.read_csv(file_path, usecols=['Date', 'Adj Close', 'Volume'], parse_dates=['Date'])
            df_vol.set_index('Date', inplace=True)
            df_vol.rename(columns={'Adj Close': f"{ticker}_price", 'Volume': f"{ticker}_vol"}, inplace=True)
            dataframes_vol.append(df_vol)

        combined_df_vol = pd.concat(dataframes_vol, axis=1)
        combined_df_vol.fillna(method='ffill', inplace=True)
        combined_df_vol.fillna(method='bfill', inplace=True)
        log_returns_df_vol = np.log(combined_df_vol / (combined_df_vol.shift(1) + 1e-9) + 1e-9)
        log_returns_df_vol = log_returns_df_vol.dropna()

        price_columns = [col for col in log_returns_df_vol.columns if col.endswith('_price')]
        volume_columns = [col for col in log_returns_df_vol.columns if col.endswith('_vol')]
        log_returns_df_vol = log_returns_df_vol[price_columns + volume_columns]

        backup_path = filepath + ".backup"
        os.rename(filepath, backup_path)
        log_returns_df_vol.to_csv(filepath, index_label='Date')
        self.write_log(f"Backed up original file to {backup_path} and wrote new data to {filepath}")

    def download_price_data(self, tickers, start_date, end_date, dest_dir):
        """Download price data using yfinance"""
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for ticker in tickers:
            try:
                data = yfin.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False, timeout=40)
                data.index = data.index.date.astype(str)
                data.to_csv(f'{dest_dir}/{ticker}.csv', index_label='Date')
                print(f'Downloaded data for {ticker}')
            except Exception as e:
                print(f'Failed to download data for {ticker}: {e}')
                continue

    def initialize_population(self, initial_size=100):
        """Initialize population with minimal NEAT creatures"""
        for _ in range(initial_size):
            creature = NEATCreature(
                num_features=max(1, self.current_num_features),
                num_portfolio_positions=max(2, self.current_num_stocks + 1),  # +1 for cash
                long_only=self.long_only
            )
            creature.birth_step = self.current_step
            self.population.append(creature)

        # Set batch size based on population size
        self.batch_size = max(25, len(self.population) // self.max_workers)

        # Initial speciation
        self.speciate_population()
        self.write_log(f"Initialized population with {initial_size} creatures in {len(self.species)} species")
        self.write_log(f"Initial dimensions: {self.current_num_features} features, {self.current_num_stocks + 1} positions")
        self.write_log(f"Elite reproduction: {self.elite_reproduction_rate*100:.1f}% can reproduce, max {self.max_offspring_per_parent} offspring each")

        # Initialize process pool after population is created
        self._initialize_process_pool()

    def expand_population_for_new_files(self, num_new_files):
        """Expand population when new files become available"""
        expansion_amount = num_new_files * self.expansion_increment
        old_max_population = self.max_population
        self.max_population += expansion_amount
        
        self.write_log(f"Expanding max population: {old_max_population} → {self.max_population} (+{expansion_amount})")
        
        # Add new creatures by cloning and mutating existing top performers
        if self.population:
            # Get top 20% of population to clone from
            sorted_pop = sorted(self.population, key=lambda c: c.energy, reverse=True)
            top_performers = sorted_pop[:max(1, int(len(sorted_pop) * 0.2))]
            
            new_creatures = []
            for _ in range(expansion_amount):
                # Clone a random top performer
                parent = random.choice(top_performers)
                child = self.clone_creature(parent)
                
                # Adapt to new dimensions
                child.adapt_to_new_features(self.current_num_features, self.current_num_stocks + 1)
                
                # Apply expansion mutation
                child.expansion_mutation(self.current_num_features, self.current_num_stocks + 1)
                
                # Reset some properties
                child.birth_step = self.current_step
                child.energy = parent.energy * random.uniform(0.8, 1.2)  # Add some variation
                
                new_creatures.append(child)
            
            self.population.extend(new_creatures)
            self.write_log(f"Added {len(new_creatures)} new creatures through expansion cloning")
        
        # Re-speciate with expanded population
        self.speciate_population()

    def calculate_portfolio_return(self, creature, returns_log, previous_portfolio, new_portfolio, eps=1e-12):
        """Calculate portfolio returns with commission costs"""
        portfolio_change = np.abs(new_portfolio - previous_portfolio)
        turnover = np.sum(portfolio_change[:-1])  # No turnover on cash
        cost_frac = self.commission_rate * turnover

        gross_simple = np.dot(previous_portfolio[:-1], np.expm1(returns_log))
        gross_log = np.log(max(1.0 + gross_simple, eps))

        net_log_return = gross_log + np.log(max(1.0 - cost_frac, eps))
        return net_log_return

    def write_log(self, message):
        """Write message to log file"""
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def step(self, day_idx):
        """Execute one trading day with multi-file support"""
        if self.data_manager is None:
            # Single file mode
            return self._step_single_file(day_idx)
        else:
            # Multi-file mode
            return self._step_multi_file(day_idx)

    def _step_single_file(self, day_idx):
        """Single file step (original implementation)"""
        self.write_log(f"Entering step function with day_idx: {day_idx}, stock_data length: {len(self.stock_data)}")
        
        if day_idx >= len(self.stock_data):
            self.write_log(f"day_idx {day_idx} >= stock_data length {len(self.stock_data)}, returning False")
            return False

        current_date = self.stock_data.index[day_idx]
        x_features = self.features[day_idx]
        r_price = self.returns[day_idx]

        return self._execute_step_common(current_date, x_features, r_price, day_idx)

    def _step_multi_file(self, day_idx):
        """Multi-file step with adaptive features"""
        if day_idx >= self.data_manager.get_total_days():
            self.write_log(f"day_idx {day_idx} >= total days {self.data_manager.get_total_days()}, returning False")
            return False

        current_date = self.data_manager.get_date_by_index(day_idx)
        if current_date is None:
            return False

        # Get features for this date
        x_features, r_price, current_tickers = self.data_manager.get_features_for_date(current_date)
        
        if x_features is None:
            self.write_log(f"No data available for date {current_date.date()}, skipping")
            self.current_step += 1
            return True

        # Check for dimension changes
        new_num_features, new_num_stocks = self.data_manager.get_current_feature_dimensions(current_date)
        
        # Check if we need to trigger expansion mutations
        expansion_triggered = False
        if self.prev_date:
            should_expand, num_new_files = self.data_manager.should_trigger_expansion_mutation(current_date, self.prev_date)
            if should_expand:
                # Check which files are actually new
                current_files = set(config['filename'] for config in self.data_manager.get_available_files_for_date(current_date))
                truly_new_files = current_files - self.files_seen
                
                if truly_new_files:
                    self.write_log(f"=== EXPANSION EVENT DETECTED at {current_date.date()} ===")
                    self.write_log(f"New files: {list(truly_new_files)}")
                    self.write_log(f"Features: {self.current_num_features} → {new_num_features}")
                    self.write_log(f"Stocks: {self.current_num_stocks} → {new_num_stocks}")
                    
                    # Update files_seen
                    self.files_seen.update(truly_new_files)
                    
                    # Expand population capacity
                    self.expand_population_for_new_files(len(truly_new_files))
                    
                    # Update population to handle new dimensions
                    for creature in self.population:
                        creature.adapt_to_new_features(new_num_features, new_num_stocks + 1)
                        
                        # Apply expansion mutations to 50% of population
                        if random.random() < 0.5:
                            creature.expansion_mutation(new_num_features, new_num_stocks + 1)
                            
                    # Update global dimensions
                    self.current_num_features = new_num_features
                    self.current_num_stocks = new_num_stocks
                    
                    # Update feature and position names
                    all_tickers = []
                    for config in self.data_manager.get_available_files_for_date(current_date):
                        all_tickers.extend(config['tickers'])
                    
                    self.tickers = all_tickers
                    self.feature_names = []
                    for ticker in all_tickers:
                        self.feature_names.append(f"{ticker}_price_return")
                    for ticker in all_tickers:
                        self.feature_names.append(f"{ticker}_volume_logdiff")
                    
                    self.position_names = all_tickers.copy()
                    self.position_names.append("CASH")
                    
                    expansion_triggered = True
                    self.write_log(f"Population adapted to new dimensions: {new_num_features}F x {new_num_stocks + 1}P")
                    self.write_log(f"Population expanded to {len(self.population)} creatures (max: {self.max_population})")
                    self.write_log("=" * 50)

        result = self._execute_step_common(current_date, x_features, r_price, day_idx, expansion_triggered)
        
        self.prev_date = current_date
        return result

    def _execute_step_common(self, current_date, x_features, r_price, day_idx, expansion_triggered=False):
        """Common step execution logic"""
        self.write_log(f"Processing step {self.current_step}, date: {current_date}")

        if not self.population:
            self.write_log(
                f"Step {self.current_step}: Alive: 0, Species: 0, Died: 0, Total dead: {self.dead_count}"
            )
            self.selected_creature = None
            self.selected_rank = None
            self.current_step += 1
            return True

        # Update batch size if population changed significantly
        new_batch_size = max(25, len(self.population) // self.max_workers)
        if abs(new_batch_size - self.batch_size) > 10:
            self.batch_size = new_batch_size

        self.write_log(f"Evaluating {len(self.population)} creatures with {self.max_workers} processes (batch size: {self.batch_size})")
        if expansion_triggered:
            self.write_log(f"*** EXPANSION EVENT: New dimensions active, max population: {self.max_population} ***")

        start_time = time.time()
        results = []

        try:
            # Prepare creature data for process serialization
            creature_data_batches = []
            for i in range(0, len(self.population), self.batch_size):
                batch = self.population[i:i + self.batch_size]
                creature_data_batch = []
                for creature in batch:
                    creature_data = self.prepare_creature_for_process(creature)
                    creature_data_batch.append(creature_data)
                creature_data_batches.append(creature_data_batch)

            # Submit batches to persistent process pool
            futures = []
            for creature_data_batch in creature_data_batches: 
                future = self.process_pool.submit(
                    evaluate_creature_batch_worker_process,
                    creature_data_batch,
                    r_price,
                    x_features,
                    self.commission_rate,
                    self.current_step,
                    self.current_num_features,
                    self.current_num_stocks,
                    self.long_only
                )
                futures.append(future)

            # Collect results from all batches
            for future in as_completed(futures, timeout=600):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    self.write_log(f"Process batch failed: {e}")

        except Exception as e:
            self.write_log(f"ProcessPoolExecutor failed: {e}, falling back to sequential")
            # Fallback to sequential
            for creature in self.population:
                result = self._evaluate_creature(creature, r_price, x_features)
                creature_result = (result[0].creature_id, result[1], result[2], result[3], result[4], result[5])
                results.append(creature_result)

        end_time = time.time()
        self.write_log(f"Evaluation took {end_time - start_time:.2f} seconds, got {len(results)} results")

        # Process results
        creature_map = {c.creature_id: c for c in self.population}
        energies = []
        ages = []
        creatures_to_remove = []
        portfolio_return_map = {}

        for creature_id, new_portfolio, new_energy, portfolio_log_return, age, alive in results:
            if creature_id in creature_map:
                creature = creature_map[creature_id]
                portfolio_return_map[creature_id] = portfolio_log_return
                creature.fitness_history.append(portfolio_log_return)

                if alive:
                    creature.energy = new_energy
                    creature.portfolio = new_portfolio
                    energies.append(creature.energy)
                    ages.append(age)
                else:
                    creatures_to_remove.append(creature)

        # Remove dead creatures
        for creature in creatures_to_remove:
            if creature in self.population:
                self.population.remove(creature)
            self.dead_count += 1
            self.dead_meta.append({
                "id": creature.creature_id,
                "birth_step": creature.birth_step,
                "death_step": self.current_step,
                "age": self.current_step - creature.birth_step,
                "energy_at_death": float(getattr(creature, "energy", 0.0)),
            })

        # Re-speciate population
        self.speciate_population()

        # Calculate adjusted fitness
        self.calculate_adjusted_fitness()

        # Update selected creature
        self.update_selected_creature()

        # Get return for selected creature
        selected_creature_return = None
        if self.selected_creature is not None:
            selected_creature_id = self.selected_creature.creature_id
            if selected_creature_id in portfolio_return_map:
                selected_creature_return = portfolio_return_map[selected_creature_id]
            else:
                self.write_log(f"WARNING: Selected creature {selected_creature_id} not in portfolio_return_map!")

        # Compute stats
        if energies:
            avg_energy = float(np.mean(energies))
            max_energy = float(np.max(energies))
            avg_age = float(np.mean(ages))
            max_age = int(np.max(ages))

            best_creature = max(self.population, key=lambda x: x.energy)
            best_id = best_creature.creature_id
            best_age = self.current_step - best_creature.birth_step
        else:
            avg_energy = 0.0
            max_energy = 0.0
            avg_age = 0.0
            max_age = 0
            best_id = "N/A"
            best_age = 0

        dead_count = len(creatures_to_remove)
        total_dead = self.dead_count

        # Log step summary
        species_info = f"Species: {len(self.species)}"
        species_sizes = [len(s.members) for s in self.species.values()]
        if species_sizes:
            species_info += f" (avg size: {np.mean(species_sizes):.1f})"

        dimension_info = f"Dims: {self.current_num_features}F x {self.current_num_stocks}S"
        if expansion_triggered:
            dimension_info += " [EXPANDED]"

        population_info = f"Pop: {len(self.population)}/{self.max_population}"

        self.write_log(
            f"Step {self.current_step}: "
            f"Date: {current_date}, "
            f"{population_info}, "
            f"{species_info}, "
            f"{dimension_info}, "
            f"Died: {dead_count}, "
            f"Total dead: {total_dead}, "
            f"Avg Energy: {avg_energy:.3f}, "
            f"Max Energy: {max_energy:.3f}, "
            f"Best ID: {best_id} (age: {best_age}), "
            f"Max Age: {max_age}, "
            f"Avg Age: {avg_age:.1f}"
        )

        # Log selected creature info
        if self.selected_creature is not None:
            sel_age = self.current_step - self.selected_creature.birth_step
            rank_str = (
                f"{self.selected_rank + 1}/{len(self.population)}"
                if self.selected_rank is not None else f"?/{len(self.population)}"
            )

            net_info = self.selected_creature.get_network_size()
            species_id = getattr(self.selected_creature, 'species_id', 'None')
            self.write_log(
                f"  Selected Creature: ID {self.selected_creature.creature_id}, "
                f"Energy: {self.selected_creature.energy:.3f}, "
                f"Age: {sel_age}, Rank: {rank_str}, Species: {species_id}, "
                f"Dims: {self.selected_creature.num_features}F x {self.selected_creature.num_portfolio_positions}P, "
                f"Network: {net_info['nodes']} nodes ({net_info['inputs']}I, "
                f"{net_info['hidden']}H, {net_info['outputs']}O, {net_info['recurrent']}R), "
                f"{net_info['connections']} connections"
            )

            # Update global asset
            if selected_creature_return is not None:
                self.asset *= float(np.exp(selected_creature_return))
                self.asset_history.append(self.asset)
                self.output_records.append((current_date, selected_creature_return))
                self._append_output_record(current_date, selected_creature_return)
                self.write_log(f"  Global Asset (selected creature): {self.asset:.6f}")
            else:
                self.write_log(f"  WARNING: No portfolio return available for selected creature")

        # Elite-based reproduction - MUCH less frequent
        if self.current_step > 0 and self.current_step % 256 == 0:  # Every 256 steps instead of 128
            self.write_log(f"Starting elite-based reproduction at step {self.current_step}")
            self.elite_based_reproduction()

        # Log best creature portfolio periodically
        if self.current_step % 10 == 0 and self.population:
            best_creature = max(self.population, key=lambda x: x.energy)
            portfolio_str = ", ".join([f"{w:.3f}" for w in best_creature.portfolio])
            net_info = best_creature.get_network_size()
            species_id = getattr(best_creature, 'species_id', 'None')
            self.write_log(
                f"  Best Creature ID {best_creature.creature_id} (Species {species_id}) "
                f"Portfolio: [{portfolio_str}], "
                f"Network: {net_info['inputs']}I→{net_info['hidden']}H→{net_info['outputs']}O "
                f"({net_info['recurrent']} recurrent)"
            )

        self.current_step += 1
        return True

    def crossover(self, parent1, parent2):
        """NEAT crossover with innovation protection and dimension adaptation"""
        # Determine which parent is more fit
        if parent1.energy >= parent2.energy:
            fitter_parent, other_parent = parent1, parent2
        else:
            fitter_parent, other_parent = parent2, parent1

        # Create child with current dimensions
        child = NEATCreature(
            num_features=self.current_num_features,
            num_portfolio_positions=self.current_num_stocks + 1,
            long_only=self.long_only
        )

        # Clear default minimal network
        child.nodes = {}
        child.connections = {}
        child.next_node_id = 0

        # Inherit all nodes from both parents
        all_nodes = {}
        node_id_mapping = {}

        # Add nodes from both parents, preserving structural innovations
        for parent in [fitter_parent, other_parent]:
            for node_id, node in parent.nodes.items():
                # Create a unique key for the node based on its properties
                node_key = (node.node_type, node.input_index, node.output_index, node.activation_func)

                if node_key not in all_nodes:
                    new_node = NodeGene(
                        child.next_node_id,
                        node.node_type,
                        node.activation_func
                    )
                    new_node.input_index = node.input_index
                    new_node.output_index = node.output_index

                    # Ensure indices are within current bounds
                    if new_node.node_type == 'input' and new_node.input_index is not None:
                        if new_node.input_index >= self.current_num_features:
                            new_node.input_index = new_node.input_index % max(1, self.current_num_features)
                    if new_node.node_type == 'output' and new_node.output_index is not None:
                        if new_node.output_index >= self.current_num_stocks + 1:
                            new_node.output_index = new_node.output_index % max(1, self.current_num_stocks + 1)

                    all_nodes[node_key] = new_node
                    node_id_mapping[node_id] = child.next_node_id
                    child.nodes[child.next_node_id] = new_node
                    child.next_node_id += 1
                else:
                    # Map to existing node
                    existing_node = all_nodes[node_key]
                    for existing_id, existing_node_obj in child.nodes.items():
                        if existing_node_obj == existing_node:
                            node_id_mapping[node_id] = existing_id
                            break

        # NEAT crossover for connections
        all_innovations = set()
        all_innovations.update(fitter_parent.connections.keys())
        all_innovations.update(other_parent.connections.keys())

        for innovation in sorted(all_innovations):
            conn1 = fitter_parent.connections.get(innovation)
            conn2 = other_parent.connections.get(innovation)

            chosen_conn = None
            if conn1 and conn2:
                # Both parents have this innovation - choose randomly but bias towards fitter parent
                if random.random() < 0.6:
                    chosen_conn = conn1
                else:
                    chosen_conn = conn2
            elif conn1:
                # Only fitter parent has this innovation - always inherit
                chosen_conn = conn1
            elif conn2 and fitter_parent.energy - other_parent.energy < 0.1:
                # Only other parent has innovation, but parents are close in fitness
                if random.random() < 0.3:
                    chosen_conn = conn2

            if chosen_conn:
                # Map to new node IDs
                old_from = chosen_conn.from_node
                old_to = chosen_conn.to_node

                new_from = None
                new_to = None

                # Find the corresponding nodes in the child
                for parent in [fitter_parent, other_parent]:
                    if old_from in parent.nodes and old_from in node_id_mapping:
                        new_from = node_id_mapping[old_from]
                        break

                for parent in [fitter_parent, other_parent]:
                    if old_to in parent.nodes and old_to in node_id_mapping:
                        new_to = node_id_mapping[old_to]
                        break

                if new_from is not None and new_to is not None:
                    new_conn = ConnectionGene(
                        new_from,
                        new_to,
                        chosen_conn.weight,
                        chosen_conn.enabled,
                        innovation
                    )
                    child.connections[innovation] = new_conn

        # Ensure child has at least one input and one output
        self._ensure_minimal_connectivity(child)

        return child

    def _ensure_minimal_connectivity(self, creature):
        """Ensure creature has minimal connectivity"""
        has_input = any(node.node_type == 'input' for node in creature.nodes.values())
        has_output = any(node.node_type == 'output' for node in creature.nodes.values())

        if not has_input:
            # Add a random input
            input_node = NodeGene(creature.next_node_id, 'input', 'linear')
            input_node.input_index = random.randint(0, max(0, self.current_num_features - 1))
            creature.nodes[creature.next_node_id] = input_node
            creature.next_node_id += 1

        if not has_output:
            # Add a random output
            output_node = NodeGene(creature.next_node_id, 'output', 'sigmoid')
            output_node.output_index = random.randint(0, self.current_num_stocks)
            creature.nodes[creature.next_node_id] = output_node
            creature.next_node_id += 1

        # Ensure there's at least one connection
        if not creature.connections:
            input_nodes = [nid for nid, node in creature.nodes.items() if node.node_type == 'input']
            output_nodes = [nid for nid, node in creature.nodes.items() if node.node_type == 'output']

            if input_nodes and output_nodes:
                from_node = random.choice(input_nodes)
                to_node = random.choice(output_nodes)
                weight = random.uniform(-2.0, 2.0)
                connection = ConnectionGene(from_node, to_node, weight, True)
                creature.connections[connection.innovation] = connection

    def run_simulation(self, end_day=None):
        """Run the full NEAT simulation with multi-file support and elite reproduction"""
        if self.data_manager is None and self.stock_data is None:
            raise ValueError("Please load data first using load_data()")

        if end_day is None:
            if self.data_manager:
                end_day = self.data_manager.get_total_days() - 1
            else:
                end_day = len(self.stock_data) - 1

        self.asset_history = [self.asset]
        self.output_records = []
        self._init_output_csv()

        start_day = self.current_step
        mode_str = "Multi-file" if self.data_manager else "Single-file"
        self.write_log(f"\nStarting {mode_str} NEAT simulation with elite-based reproduction from day {start_day} to day {end_day}")
        self.write_log(f"Elite reproduction parameters: {self.elite_reproduction_rate*100:.1f}% elites, max {self.max_offspring_per_parent} offspring per parent")
        self.write_log(f"Population growth: Only adds offspring, removes creatures only when max capacity exceeded")
        self.write_log(f"Using persistent process pool with {self.max_workers} workers")
        
        if self.data_manager:
            self.write_log(f"Multi-file mode: {len(self.data_manager.file_configs)} input files")
            self.write_log(f"Adaptive dimensions: Starting with {self.current_num_features}F x {self.current_num_stocks}S")
            self.write_log(f"Population expansion: {self.expansion_increment} creatures per new file")
            
        self.write_log("=" * 80)

        for day in range(start_day, end_day + 1):
            if not self.step(day):
                break

            if len(self.population) == 0:
                print("\nPOPULATION EXTINCT!")
                break

            if (self.checkpoint_dir and self.checkpoint_interval and
                (self.current_step % self.checkpoint_interval) == 0 and self.current_step > 0):
                self.save_checkpoint()

        if self.checkpoint_dir:
            self.save_checkpoint()

        self._finalize_simulation()

    def _finalize_simulation(self):
        """Finalize simulation and save results"""
        mode_str = "Multi-file" if self.data_manager else "Single-file"
        self.write_log(f"\n=== {mode_str} NEAT SIMULATION WITH ELITE REPRODUCTION COMPLETE ===")
        self.write_log(f"Final population: {len(self.population)}")
        self.write_log(f"Final species count: {len(self.species)}")
        self.write_log(f"Total deaths: {self.dead_count}")
        self.write_log(f"Final dimensions: {self.current_num_features}F x {self.current_num_stocks}S")
        self.write_log(f"Final max population: {self.max_population} (started with {self.initial_max_population})")
        self.write_log(f"Elite reproduction: {self.elite_reproduction_rate*100:.1f}% of population eligible, {self.max_offspring_per_parent} max offspring per parent")

        if self.population:
            energies = [c.energy for c in self.population]
            ages = [self.current_step - c.birth_step for c in self.population]
            self.write_log(f"Final avg energy: {np.mean(energies):.2f}")
            self.write_log(f"Final max energy: {np.max(energies):.2f}")
            self.write_log(f"Final avg age: {np.mean(ages):.1f}")
            self.write_log(f"Final max age: {np.max(ages)}")

            # Calculate elite population stats
            elite_count = int(len(self.population) * self.elite_reproduction_rate)
            sorted_pop = sorted(self.population, key=lambda c: getattr(c, 'adjusted_fitness', c.energy), reverse=True)
            elite_energies = [c.energy for c in sorted_pop[:elite_count]]
            
            self.write_log(f"Elite population ({elite_count} creatures):")
            self.write_log(f"  Elite avg energy: {np.mean(elite_energies):.2f}")
            self.write_log(f"  Elite min energy: {np.min(elite_energies):.2f}")

            # Show final species summary
            self.write_log(f"\nFinal Species Summary:")
            for species_id, species in self.species.items():
                if species.members:
                    champion = species.get_champion()
                    net_info = champion.get_network_size() if champion else {}
                    elite_in_species = len([c for c in species.members if c in sorted_pop[:elite_count]])
                    self.write_log(
                        f"  Species {species_id}: {len(species.members)} members ({elite_in_species} elites), "
                        f"best fitness: {species.best_fitness:.3f}, "
                        f"age: {species.age}, "
                        f"max complexity: {species.max_nodes}N/{species.max_connections}C, "
                        f"recurrent: {species.has_recurrent}"
                    )

            # Show top performers with detailed network info
            self.write_log("\nFinal Top 5 NEAT creatures with full network details:")
            top_5 = sorted_pop[:5]
            for i, creature in enumerate(top_5):
                age = self.current_step - creature.birth_step
                net_info = creature.get_network_size()
                species_id = getattr(creature, 'species_id', 'None')
                elite_status = "ELITE" if i < elite_count else "NON-ELITE"
                self.write_log(
                    f"  {i+1}. ID: {creature.creature_id}, Energy: {creature.energy:.3f}, "
                    f"Age: {age}, Species: {species_id}, Status: {elite_status}, "
                    f"Dims: {creature.num_features}F x {creature.num_portfolio_positions}P, "
                    f"Network: {net_info['inputs']}I→{net_info['hidden']}H→{net_info['outputs']}O "
                    f"({net_info['recurrent']} recurrent connections)"
                )

                # Show detailed final network structure
                if hasattr(self, 'feature_names') and hasattr(self, 'position_names'):
                    detailed_structure = creature.get_network_structure_detailed(self.feature_names, self.position_names)
                    self.write_log("     Final Network Structure:")
                    for line in detailed_structure.split('\n'):
                        if line.strip():
                            self.write_log(f"       {line}")
                    self.write_log("")

            # Save selected creature portfolio
            if self.selected_creature:
                self._save_selected_creature_portfolio()

    def _save_selected_creature_portfolio(self):
        """Save the selected creature's portfolio to JSON"""
        selected_portfolio_str = ", ".join([f"{w:.3f}" for w in self.selected_creature.portfolio])
        species_id = getattr(self.selected_creature, 'species_id', 'None')
        net_info = self.selected_creature.get_network_size()

        self.write_log(f"Selected Creature ID {self.selected_creature.creature_id} "
                      f"(Species {species_id}) Portfolio: [{selected_portfolio_str}]")
        self.write_log(f"Selected Creature Network: {net_info}")

        # Create position names
        if hasattr(self, 'tickers') and self.tickers:
            position_names_with_cash = self.tickers.copy()
        else:
            position_names_with_cash = [f"Asset_{i}" for i in range(len(self.selected_creature.portfolio) - 1)]
        position_names_with_cash.append("__CASH__")

        # Ensure portfolio and names have same length
        if len(self.selected_creature.portfolio) != len(position_names_with_cash):
            self.write_log(f"WARNING: Portfolio length {len(self.selected_creature.portfolio)} != position names {len(position_names_with_cash)}")
            # Truncate or pad as needed
            min_len = min(len(self.selected_creature.portfolio), len(position_names_with_cash))
            portfolio_to_save = self.selected_creature.portfolio[:min_len]
            names_to_save = position_names_with_cash[:min_len]
        else:
            portfolio_to_save = self.selected_creature.portfolio
            names_to_save = position_names_with_cash

        data = [
            {"id": ticker, "weight": float(weight)}
            for ticker, weight in zip(names_to_save, portfolio_to_save)
        ]

        # Get current date string
        if self.data_manager:
            current_date = self.data_manager.get_date_by_index(self.current_step - 1)
        else:
            current_date = self.stock_data.index[self.current_step - 1]
            
        current_date_str = current_date.strftime("%Y%m%d")
        json_filename = f"ticket_to_buy_{current_date_str}.json"

        if self.output_json_dir:
            json_path = os.path.join(self.output_json_dir, json_filename)
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            self.write_log(f"Saved selected creature portfolio to JSON: {json_path}")

    def save_selected_log_returns(self, output_file: str):
        """Save cumulative log returns to CSV"""
        if not self.output_records:
            print("No selected creature log-return data to save.")
            return

        dates, total_log_returns = zip(*self.output_records)
        df_out = pd.DataFrame({
            "Date": dates,
            "TotalLogReturn": total_log_returns
        })
        df_out.to_csv(output_file, index=False)
        print(f"Saved selected creature log returns to: {output_file}")

    def _evaluate_creature(self, creature, returns_price_vec: np.ndarray, features_vec: np.ndarray):
        """Thread worker for creature evaluation"""
        try:
            # Forward pass
            new_portfolio = creature.forward(features_vec)

            # Calculate returns
            portfolio_log_return = self.calculate_portfolio_return(
                creature, returns_price_vec, creature.portfolio, new_portfolio
            )

            # Update energy
            new_energy = creature.energy * np.exp(portfolio_log_return) - DAILY_COST
            age = self.current_step - creature.birth_step
            alive = new_energy > 0.0

            return creature, new_portfolio, new_energy, portfolio_log_return, age, alive

        except Exception as e:
            # If evaluation fails, mark as dead
            self.write_log(f"Creature {creature.creature_id} evaluation failed: {e}")
            return creature, creature.portfolio, 0.0, -10.0, 0, False

    def _init_output_csv(self):
        """Initialize output CSV file"""
        if self.output_file is None:
            return

        self._output_file_initialized = False

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        with open(self.output_file, "w") as f:
            f.write("Date,LogReturn\n")

        self._output_file_initialized = True

    def _append_output_record(self, date, log_return):
        """Append record to output CSV"""
        if self.output_file is None:
            return

        if not self._output_file_initialized or not os.path.exists(self.output_file):
            self._init_output_csv()

        with open(self.output_file, "a") as f:
            f.write(f"{date.isoformat()},{log_return}\n")


def parse_args(argv=None):
    """Parse command line arguments with multi-file support and elite reproduction parameters"""
    if argv is None:
        argv = sys.argv[1:]

    # Defaults
    max_population = 2000
    log_file = "output.log"
    end_day = None
    input_file = "stock_data_vol.csv"  # Single file mode
    input_dir = None  # Multi-file mode
    output_file = None
    output_json_dir = None
    long_only = False
    checkpoint_dir = None
    checkpoint_interval = 4000
    reload = False
    complexity_panelty = 1.0
    expansion_increment = 2000
    elite_reproduction_rate = 0.15  # 15% of population can reproduce
    max_offspring_per_parent = 3    # Max 3 offspring per elite parent

    try:
        opts, args = getopt.getopt(
            argv,
            "m:l:i:d:e:o:j:c:k:p:x:E:O:Lrh",
            ["max-population=", "log-file=", "input-file=", "input-dir=", "end-day=", "output-file=",
             "json-output-dir=", "checkpoint-dir=", "checkpoint-interval=", "complexity_panelty=", 
             "expansion-increment=", "elite-rate=", "max-offspring=", "long-only", "reload", "help"]
        )
    except getopt.GetoptError as e:
        print(f"Error: {e}")
        print("Usage: script.py [-m N] [-l FILE] [-i FILE | -d DIR] [-e N] [-o FILE] [-j DIR] [-c DIR] [-k N] [-x N] [-E RATE] [-O N] [-L] [-r] [-h]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("NEAT Trading System with Elite-Based Reproduction")
            print("Usage: script.py [OPTIONS]")
            print("Options:")
            print("  -m, --max-population N    Maximum population size")
            print("  -l, --log-file FILE       Log file path")
            print("  -i, --input-file FILE     Input data file (single file mode)")
            print("  -d, --input-dir DIR       Input directory (multi-file mode)")
            print("  -e, --end-day N          End simulation at day N")
            print("  -o, --output-file FILE    Output file for results")
            print("  -j, --json-output-dir DIR Directory for JSON output")
            print("  -c, --checkpoint-dir DIR  Checkpoint directory")
            print("  -k, --checkpoint-interval N Checkpoint interval")
            print("  -p, --complexity-panelty 0-1 complexity panelty")
            print("  -x, --expansion-increment N Population increase per new file (default: 2000)")
            print("  -E, --elite-rate RATE    Elite reproduction rate (0.0-1.0, default: 0.15)")
            print("  -O, --max-offspring N    Max offspring per elite parent (default: 3)")
            print("  -L, --long-only          Enable long-only mode")
            print("  -r, --reload            Reload data from sources")
            print("  -h, --help              Show this help message")
            print()
            print("Elite Reproduction System - FIXED:")
            print("  Only the top --elite-rate% of creatures can reproduce")
            print("  Each elite can produce up to --max-offspring offspring")
            print("  NO creatures removed unless max population exceeded")
            print("  Population grows naturally until capacity limits")
            print("  Reproduction occurs every 256 steps")
            sys.exit(0)
        elif opt in ("-m", "--max-population"):
            max_population = int(arg)
        elif opt in ("-l", "--log-file"):
            log_file = arg
        elif opt in ("-i", "--input-file"):
            input_file = arg
        elif opt in ("-d", "--input-dir"):
            input_dir = arg
        elif opt in ("-e", "--end-day"):
            end_day = int(arg)
        elif opt in ("-o", "--output-file"):
            output_file = arg
        elif opt in ("-j", "--json-output-dir"):
            output_json_dir = arg
        elif opt in ("-c", "--checkpoint-dir"):
            checkpoint_dir = arg
        elif opt in ("-k", "--checkpoint-interval"):
            checkpoint_interval = int(arg)
        elif opt in ("-p", "--complexity-panelty"):
            complexity_panelty = float(arg)
        elif opt in ("-x", "--expansion-increment"):
            expansion_increment = int(arg)
        elif opt in ("-E", "--elite-rate"):
            elite_reproduction_rate = float(arg)
            if not 0.0 <= elite_reproduction_rate <= 1.0:
                print("Error: Elite reproduction rate must be between 0.0 and 1.0")
                sys.exit(2)
        elif opt in ("-O", "--max-offspring"):
            max_offspring_per_parent = int(arg)
            if max_offspring_per_parent < 1:
                print("Error: Max offspring per parent must be at least 1")
                sys.exit(2)
        elif opt in ("-L", "--long-only"):
            long_only = True
        elif opt in ("-r", "--reload"):
            reload = True

    return (max_population, log_file, input_file, input_dir, end_day, output_file,
            output_json_dir, checkpoint_dir, checkpoint_interval, complexity_panelty, 
            expansion_increment, elite_reproduction_rate, max_offspring_per_parent, long_only, reload, args)


# Main execution
if __name__ == "__main__":
    input_file_path = './input_files'
    output_file_path = './output_files'
    log_file_path = './log_files'

    # Create directories
    Path(output_file_path).mkdir(parents=True, exist_ok=True)
    Path(log_file_path).mkdir(parents=True, exist_ok=True)

    # Parse arguments
    (max_population, log_file, input_file, input_dir, end_day, output_file,
     output_json_dir, checkpoint_dir, checkpoint_interval, complexity_panelty, 
     expansion_increment, elite_reproduction_rate, max_offspring_per_parent, long_only, reload, _) = parse_args()

    if output_json_dir:
        Path(output_json_dir).mkdir(parents=True, exist_ok=True)

    # Determine mode and construct paths
    multi_file_mode = input_dir is not None
    
    if multi_file_mode:
        # Multi-file mode - input_dir should be a directory path
        actual_input_dir = input_dir
        if not os.path.isabs(actual_input_dir):
            actual_input_dir = os.path.join(input_file_path, actual_input_dir)
        print(f"Multi-file mode: Using input directory {actual_input_dir}")
        print(f"Population will expand by {expansion_increment} per new file")
    else:
        # Single file mode
        actual_input_dir = None
        input_file = f"{input_file_path}/{input_file}"
        print(f"Single-file mode: Using input file {input_file}")

    if output_file:
        output_file = f"{output_file_path}/{output_file}"
    log_file = f"{log_file_path}/{log_file}"

    mode_str = "Multi-file" if multi_file_mode else "Single-file"
    print(f"NEAT Trading System with FIXED Elite Reproduction ({mode_str} Mode)")
    print(f"Max Population: {max_population}")
    print(f"Elite Reproduction: {elite_reproduction_rate*100:.1f}% can reproduce, max {max_offspring_per_parent} offspring each")
    print(f"Population Management: NO removal unless max capacity exceeded")
    if multi_file_mode:
        print(f"Input directory: {actual_input_dir}")
    else:
        print(f"Input file: {input_file}")
    print(f"Log file: {log_file}")
    if output_file:
        print(f"Output file: {output_file}")

    # Use context manager to handle process pool lifecycle
    with NEATTradingSystem(
        commission_rate=0.001,
        max_population=max_population,
        log_file=log_file,
        output_json_dir=output_json_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        complexity_panelty=complexity_panelty,
        long_only=long_only,
        input_dir=actual_input_dir,
        reload=reload,
        expansion_increment=expansion_increment,
        elite_reproduction_rate=elite_reproduction_rate,
        max_offspring_per_parent=max_offspring_per_parent
    ) as system:

        system.output_file = output_file

        # Load data
        if multi_file_mode:
            system.load_data(None, reload=reload)  # input_path not used in multi-file mode
        else:
            system.load_data(input_file, reload=reload)

        # Handle checkpoints or initialize population
        if checkpoint_dir:
            system.load_checkpoint()
            if len(system.population) == 0:
                system.initialize_population(initial_size=1000)

                # Clean up existing files for fresh start
                for file_path in [output_file, log_file]:
                    if file_path and os.path.exists(file_path):
                        print(f"Removing existing file: {file_path}")
                        os.remove(file_path)
        else:
            system.initialize_population(initial_size=1000)

        # Run simulation
        system.run_simulation(end_day=end_day)

    print(f"NEAT simulation with FIXED elite reproduction ({mode_str} mode) completed!")