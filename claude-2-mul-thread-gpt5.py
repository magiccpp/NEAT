import getopt
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
import copy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: avoid oversubscription (important when you add threads)
# try:
#     torch.set_num_threads(1)
#     torch.set_num_interop_threads(1)
# except Exception:
#     pass

# Also consider environment vars before launching Python:
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

DAILY_COST = 0

# Set random seeds for reproducibility
# np.random.seed(42)
# torch.manual_seed(42)
# random.seed(42)

class LSTMCreature(nn.Module):
    """LSTM-based creature for portfolio prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, creature_id=None):
        super(LSTMCreature, self).__init__()
        self.creature_id = creature_id or random.randint(0, 100000)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = input_size + 1  # stocks + cash
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)
        
        # Initialize hidden state
        self.hidden = None
        
        # Creature properties
        self.energy = 1.0
        self.portfolio = np.zeros(self.output_size)
        self.portfolio[-1] = 1.0  # Start with 100% cash
        self.birth_step = 0
        self.fitness_history = []
        
    def forward(self, x):
        # x shape: (1, 1, input_size) for single day
        if self.hidden is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            self.hidden = (h_0, c_0)
        
        out, self.hidden = self.lstm(x, self.hidden)
        out = self.fc(out[:, -1, :])  # Take last output
        
        # Apply softmax to get portfolio weights (for absolute values)
        # Then apply signs separately
        signs = torch.tanh(out)  # Get values between -1 and 1
        abs_weights = F.softmax(torch.abs(out), dim=-1)
        portfolio = signs * abs_weights
        
        # Normalize to ensure sum of absolute values = 1
        portfolio = portfolio / torch.sum(torch.abs(portfolio), dim=-1, keepdim=True)
        
        return portfolio
    
    def reset_hidden(self):
        self.hidden = None
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.1 ):
        """Mutate the creature's weights"""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < mutation_rate:
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)

class NEATTradingSystem:
    """NEAT-based trading system with LSTM creatures"""
    
    def __init__(self, commission_rate=0.001, max_population=100000, log_file="output.log"):
        self.commission_rate = commission_rate
        self.max_population = max_population
        self.population = []
        self.dead_creatures = []
        self.current_step = 0
        self.stock_data = None
        self.num_stocks = None
        self.log_file = log_file

    def load_data(self, filepath='stock_data.csv'):
        """Load stock data from CSV"""
        self.stock_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.num_stocks = len(self.stock_data.columns)
        self.write_log(f"Loaded data with {self.num_stocks} stocks from {self.stock_data.index[0]} to {self.stock_data.index[-1]}")
        self.write_log(f"Total trading days: {len(self.stock_data)}")
        
    def initialize_population(self, initial_size=100):
        """Initialize the population with random creatures"""
        for i in range(initial_size):
            creature = LSTMCreature(self.num_stocks, hidden_size=64, num_layers=2)
            creature.birth_step = self.current_step
            self.population.append(creature)
        self.write_log(f"Initialized population with {initial_size} creatures")
        
    def calculate_portfolio_return(self, creature, returns_log, previous_portfolio, new_portfolio, eps=1e-12):
        """Calculate returns considering commission"""
        # Calculate portfolio change
        portfolio_change = np.abs(new_portfolio - previous_portfolio)
        turnover = np.sum(portfolio_change[:-1])  # No turnover on cash
        cost_frac = self.commission_rate * turnover                     # fraction of wealth
    
      # --- Portfolio return (exact with log returns) ---
        # Simple return before costs using start-of-day weights:
        # If you assume you trade at the END of day, use previous_portfolio here.
        gross_simple = np.dot(previous_portfolio[:-1], np.expm1(returns_log))  # scalar
        gross_log = np.log(max(1.0 + gross_simple, eps))     
        
        # Calculate returns (excluding cash position)
        net_log_return = gross_log + np.log(max(1.0 - cost_frac, eps))          # scalar
        return net_log_return

    def write_log(self, message):
        """Write a message to the log file"""
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def step(self, day_idx):
        """Execute one trading day with multi-threaded creature evaluation."""
        if day_idx >= len(self.stock_data):
            return False

        # Get today's returns
        returns = self.stock_data.iloc[day_idx].values
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # If nobody's alive, nothing to do
        if not self.population:
            self.write_log(f"Step {self.current_step}: Alive: 0, Died: 0, Total dead: {len(self.dead_creatures)}, "
                  f"Avg Energy: 0.00, Max Energy: 0.00, Best ID: N/A (age: 0), Max Age: 0, Avg Age: 0.0")
            self.current_step += 1
            return True

        # Decide an executor size; threads work well with PyTorch’s internal native ops
        max_workers = min(64, (os.cpu_count() or 4) * 2)

        # Run all creatures in parallel
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for creature in self.population:
                futures.append(executor.submit(self._evaluate_creature, creature, returns, returns_tensor))
            for fut in as_completed(futures):
                results.append(fut.result())

        # Apply updates on the main thread to avoid races on shared lists
        energies = []
        ages = []
        creatures_to_remove = []

        for creature, new_portfolio, new_energy, portfolio_log_return, age, alive in results:
            creature.fitness_history.append(portfolio_log_return)
            if alive:
                creature.energy = new_energy
                creature.portfolio = new_portfolio
                energies.append(creature.energy)
                ages.append(age)
            else:
                creatures_to_remove.append(creature)

        # Remove dead creatures (mutate population only here)
        for creature in creatures_to_remove:
            # Note: do not call creature.reset_hidden(); it’s dead anyway
            if creature in self.population:
                self.population.remove(creature)
            self.dead_creatures.append(creature)

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
        total_dead = len(self.dead_creatures)

        # write to the log file
        self.write_log(f"Step {self.current_step}: "
                       f"Alive: {len(self.population)}, "
                       f"Died: {dead_count}, "
                       f"Total dead: {total_dead}, "
                       f"Avg Energy: {avg_energy:.3f}, "
                       f"Max Energy: {max_energy:.3f}, "
                       f"Best ID: {best_id} (age: {best_age}), "
                       f"Max Age: {max_age}, "
                       f"Avg Age: {avg_age:.1f}")
        

        # Reproduction every 128 steps (skip step 0)
        if self.current_step > 0 and self.current_step % 128 == 0:
            self.reproduce(avg_energy)
        
        if self.current_step % 10 == 0:
            # print out current portforlio for the best creature in one line
            if self.population:
                best_creature = max(self.population, key=lambda x: x.energy)
                portfolio_str = ", ".join([f"{w:.3f}" for w in best_creature.portfolio])
                self.write_log(f"  Best Creature ID {best_creature.creature_id} Portfolio: [{portfolio_str}]")



        self.current_step += 1
        return True


    def reproduce(self, avg_energy):
        """Elite reproduction and crossover"""
        if len(self.population) < 2:
            self.write_log("Not enough creatures to reproduce")
            return
        
        self.write_log("\n=== REPRODUCTION EVENT ===")
        
        # Sort by energy (fitness)
        sorted_pop = sorted(self.population, key=lambda x: x.energy, reverse=True)
        
        # Select top 30% as elite
        elite_count = max(1, int(len(sorted_pop) * 0.3))
        elite = sorted_pop[:elite_count]
        
        # If population is at or near max, remove bottom 30% to make room
        if len(self.population) >= self.max_population * 0.7:  # If population is 70% or more of max
            bottom_count = int(len(sorted_pop) * 0.3)
            bottom_creatures = sorted_pop[-bottom_count:]
            self.write_log(f"Reproduction: Population at {len(self.population)}, removing {bottom_count} weakest creatures")
            
            # Remove bottom creatures
            for creature in bottom_creatures:
                self.population.remove(creature)
                self.dead_creatures.append(creature)
            
            self.write_log(f"Population after removal: {len(self.population)}")
            
        
        
        self.write_log(f"Elite creatures: {elite_count}")
        # Show elite creature details
        self.write_log("Top 5 Elite creatures:")
        for i, creature in enumerate(elite[:5]):
            age = self.current_step - creature.birth_step
            self.write_log(f"  {i+1}. ID: {creature.creature_id}, Energy: {creature.energy:.2f}, Age: {age}")
        
        # Create offspring
        offspring = []
        max_offspring = min(elite_count * 3, self.max_population - len(self.population))
        
        # Generate offspring from elite pairs
        offspring_created = 0
        for i in range(elite_count):
            if offspring_created >= max_offspring:
                break
                
            # Each elite can breed with 2-3 other elites
            num_mates = min(3, elite_count - 1)
            potential_mates = [e for e in elite if e != elite[i]]
            
            if potential_mates:
                mates = random.sample(potential_mates, min(num_mates, len(potential_mates)))
                
                for mate in mates:
                    if offspring_created >= max_offspring:
                        break
                    
                    # Create offspring through crossover
                    child = self.crossover(elite[i], mate)
                    child.mutate(mutation_rate=0.2, mutation_strength=0.1)
                    child.birth_step = self.current_step
                    child.energy = avg_energy
                    offspring.append(child)
                    offspring_created += 1
        
        self.write_log(f"Created {len(offspring)} offspring")
        
        # Add offspring to population
        self.population.extend(offspring)
        
        # Final check to ensure we don't exceed max population
        if len(self.population) > self.max_population:
            self.cull_population()
        
        self.write_log(f"Population after reproduction: {len(self.population)}")
        self.write_log("========================\n")
    
    def crossover(self, parent1, parent2):
        """Create offspring through crossover of two parents"""
        child = LSTMCreature(self.num_stocks, hidden_size=64, num_layers=2)
        
        # Crossover weights
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(child.parameters(), 
                                                       parent1.parameters(), 
                                                       parent2.parameters()):
                # Random crossover mask
                mask = torch.rand_like(child_param) > 0.5
                child_param.data = torch.where(mask, p1_param.data, p2_param.data)
        
        return child
    
    def cull_population(self):
        """Remove weakest creatures to maintain population limit"""
        self.write_log(f"Culling population from {len(self.population)} to {self.max_population}")
        
        # Sort by energy and keep only the strongest
        self.population = sorted(self.population, key=lambda x: x.energy, reverse=True)
        
        # Move culled creatures to dead list
        culled = self.population[self.max_population:]
        self.dead_creatures.extend(culled)
        
        self.population = self.population[:self.max_population]
    
    def run_simulation(self, start_day=0, end_day=None):
        """Run the full simulation"""
        if self.stock_data is None:
            raise ValueError("Please load data first using load_data()")
        
        if end_day is None:
            end_day = len(self.stock_data) - 0
        
        self.write_log(f"\nStarting simulation from day {start_day} to day {end_day}")
        self.write_log("=" * 80)
        
        for day in range(start_day, end_day + 1):
            if not self.step(day):
                break
            
            # Check if population is extinct
            if len(self.population) == 0:
                print("\nPOPULATION EXTINCT!")
                break
        
        self.write_log("\n=== SIMULATION COMPLETE ===")
        self.write_log(f"Final population: {len(self.population)}")
        self.write_log(f"Total deaths: {len(self.dead_creatures)}")
        
        if self.population:
            energies = [c.energy for c in self.population]
            ages = [self.current_step - c.birth_step for c in self.population]
            self.write_log(f"Final avg energy: {np.mean(energies):.2f}")
            self.write_log(f"Final max energy: {np.max(energies):.2f}")
            self.write_log(f"Final avg age: {np.mean(ages):.1f}")
            self.write_log(f"Final max age: {np.max(ages)}")
            
            # Show top performers
            self.write_log("\nTop 5 creatures:")
            top_5 = sorted(self.population, key=lambda x: x.energy, reverse=True)[:5]
            for i, creature in enumerate(top_5):
                age = self.current_step - creature.birth_step
                self.write_log(f"  {i+1}. ID: {creature.creature_id}, Energy: {creature.energy:.2f}, "
                      f"Age: {age} steps")

    def _evaluate_creature(self, creature, returns: np.ndarray, returns_tensor: torch.Tensor):
        """
        Thread worker: run forward pass, compute return & new energy, and
        return a tuple for main-thread application.
        """
        with torch.no_grad():
            # Forward pass for this creature (keeps its own hidden state)
            new_portfolio_t = creature(returns_tensor).squeeze()
            new_portfolio = new_portfolio_t.detach().cpu().numpy()

        # Compute portfolio return using current (previous) portfolio vs. new one
        portfolio_log_return = self.calculate_portfolio_return(
            creature, returns, creature.portfolio, new_portfolio
        )

        # Energy update: multiplicative gain/loss on current energy, minus daily cost
        # new_energy = creature.energy + portfolio_return * creature.energy - DAILY_COST
        new_energy = creature.energy * np.exp(portfolio_log_return) - DAILY_COST

        # Age for stats (do not mutate here; main thread will)
        age = self.current_step - creature.birth_step

        # Alive flag based on new energy
        alive = new_energy > 0.0

        return creature, new_portfolio, new_energy, portfolio_log_return, age, alive


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]  # <-- skip script name

    # Defaults
    max_population = 1000
    log_file = "output.log"

    try:
        # short options: -m, -l, -h
        # long options require '=' when they take a value
        opts, args = getopt.getopt(
            argv,
            "m:l:h",
            ["max-population=", "log-file=", "help"]
        )
    except getopt.GetoptError as e:
        print(f"Error: {e}")
        print("Usage: script.py [-m N | --max-population N] [-l FILE | --log-file FILE]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: script.py [-m N | --max-population N] [-l FILE | --log-file FILE]")
            sys.exit(0)
        elif opt in ("-m", "--max-population"):
            try:
                max_population = int(arg)
                if max_population <= 0:
                    raise ValueError("max-population must be positive")
            except ValueError as ve:
                print(f"Invalid --max-population: {ve}")
                sys.exit(2)
        elif opt in ("-l", "--log-file"):
            log_file = arg

    return max_population, log_file, args  # args = leftover positional args, if any


# Main execution
if __name__ == "__main__":
    # Create the trading system
    # get the input parameter for the max_population
    # use getopt to parse the parameters: --max-population N --log-file filename
    # Default values
    max_population, log_file, _ = parse_args()
    
    print(f"Max Population: {max_population}, log file: {log_file}")
    system = NEATTradingSystem(commission_rate=0.001, max_population=max_population, log_file=log_file)
    
    # Load data
    system.load_data('stock_data.csv')
    
    # Initialize population
    system.initialize_population(initial_size=1000)
    
    # Run simulation for first 1000 days (can be adjusted)
    system.run_simulation(start_day=0, end_day=10000)