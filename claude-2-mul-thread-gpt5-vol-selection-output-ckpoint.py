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
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import re

import yfinance as yfin
import os
import json
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

    def __init__(self, input_size, num_stocks, hidden_size=64, num_layers=2, creature_id=None, long_only=False):
        super(LSTMCreature, self).__init__()
        self.creature_id = creature_id or random.randint(0, 100000)
        self.input_size = input_size              # = 2 * num_stocks
        self.num_stocks = num_stocks
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = num_stocks + 1         # stocks + cash
        self.long_only = long_only
        # LSTM layers
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)

        # Initialize hidden state
        self.hidden = None

        # Creature properties
        self.energy = 1.0
        self.portfolio = np.zeros(self.output_size)
        self.portfolio[-1] = 1.0  # Start with 100% cash
        self.birth_step = 0
        self.fitness_history = []

    def forward(self, x: torch.Tensor):
        # x shape: (1, 1, input_size) for single day
        if self.hidden is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            self.hidden = (h_0, c_0)

        out, self.hidden = self.lstm(x, self.hidden)
        out = self.fc(out[:, -1, :])  # Take last output
        if self.long_only:
            # standard non-negative portfolio weights that sum to 1
            portfolio = F.softmax(out, dim=-1)
            return portfolio
        else:
            # original “signed‐softmax” trick to allow shorting
            signs = torch.tanh(out)                      # in (-1,1)
            abs_weights = F.softmax(torch.abs(out), dim=-1)
            portfolio = signs * abs_weights
            # normalize so sum of abs weights = 1
            portfolio = portfolio.div(portfolio.abs().sum(dim=-1, keepdim=True))
            return portfolio

    def reset_hidden(self):
        self.hidden = None

    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        """Mutate the creature's weights"""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < mutation_rate:
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)


class NEATTradingSystem:
    """NEAT-based trading system with LSTM creatures"""
    def __init__(self, commission_rate=0.001, max_population=100000, log_file="output.log", output_json_dir=None, long_only=False, checkpoint_dir=None, 
                 checkpoint_interval=None):
        self.commission_rate = commission_rate
        self.max_population = max_population
        self.population = []

        self.dead_count = 0
        self.dead_meta = []     # lightweight records only

        self.current_step = 0

        # Data holders
        self.stock_data = None        # original DataFrame
        self.features = None          # ndarray (days, 2*num_stocks) -> model input
        self.returns = None           # ndarray (days, num_stocks)    -> PnL calc
        self.num_stocks = None
        self.input_size = None

        self.log_file = log_file
        self.long_only = long_only
        self.output_json_dir = output_json_dir

        # NEW: stable selected creature (model choice)
        self.selected_creature = None
        self.selected_rank = None
        # NEW: real asset tracked for the chosen creature
        self.asset = 1.0
        self.asset_history = [self.asset]

        # NEW: records for CSV output: list of (date, cumulative_log_return)
        self.output_records = []
        # NEW: incremental CSV writing
        self.output_file = None
        self._output_file_initialized = False

        # NEW: checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval  = checkpoint_interval
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.output_json_dir:
            os.makedirs(self.output_json_dir, exist_ok=True)


    def save_checkpoint(self):
        """
        Save the entire population + selected_creature + selected_rank +
        current_step into a file named checkpoint_step_<current_step>.pt
        inside self.checkpoint_dir.
        """
        if not self.checkpoint_dir:
            return

        data = {
            "current_step":        self.current_step,
            "selected_rank":       self.selected_rank,
            # we save the ID, not the object
            "selected_creature_id": (
                self.selected_creature.creature_id
                if self.selected_creature is not None else None
            ),
            "global_asset":        self.asset,
            "creatures": []
        }

        for c in self.population:
            c_dict = {
                "creature_id":     c.creature_id,
                "state_dict":      c.state_dict(),
                "energy":          c.energy,
                "birth_step":      c.birth_step,
                "portfolio":       c.portfolio,
                "fitness_history": c.fitness_history,
            }
            data["creatures"].append(c_dict)

        filename = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_step_{self.current_step}.pt"
        )
        torch.save(data, filename)
        self.write_log(f"[Checkpoint saved at step {self.current_step} → {filename}]")


    def load_checkpoint(self):
        """
        Scan self.checkpoint_dir for files named checkpoint_step_<N>.pt,
        pick the one with largest N, and restore the state.
        """
        if not self.checkpoint_dir:
            return

        # collect all checkpoint files
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

        # load
        self.write_log(f"[Loading checkpoint from {best_file}]")
        data = torch.load(best_file, map_location="cpu", weights_only=False)

        # restore basic scalars
        self.current_step      = data.get("current_step", 0)
        self.selected_rank     = data.get("selected_rank", None)
        self.asset             = data.get("global_asset", 0)

        if self.asset == 0:
            # something is wrong, exit
            self.write_log(f"[Checkpoint load error: global_asset is zero]")
            return

        sel_id                = data.get("selected_creature_id", None)

        
        # rebuild population
        new_pop = []
        for cdata in data["creatures"]:
            c = LSTMCreature(
                input_size   = self.input_size,
                num_stocks   = self.num_stocks,
                hidden_size  = 64,
                num_layers   = 2,
                creature_id  = cdata["creature_id"],
                long_only    = self.long_only
            )
            c.load_state_dict(cdata["state_dict"])
            c.energy           = cdata["energy"]
            c.birth_step       = cdata["birth_step"]
            c.portfolio        = cdata["portfolio"]
            c.fitness_history  = cdata["fitness_history"]
            c.hidden           = None
            new_pop.append(c)

        self.population = new_pop

        # restore pointer to selected creature
        self.selected_creature = None
        if sel_id is not None:
            for c in self.population:
                if c.creature_id == sel_id:
                    self.selected_creature = c
                    break

        self.write_log(f"[Loaded checkpoint at step {self.current_step}, "
                       f"population size {len(self.population)}, "
                       f"selected_creature_id={sel_id}]")


    def update_selected_creature(self):
        """
        Maintain a stable 'chosen' creature:
        - Restrict to the top 10% by energy.
        - If the current selected creature is still in that top 10%, keep it.
        - Otherwise, choose the oldest creature within that top 10%.
        """
        if not self.population:
            self.selected_creature = None
            self.selected_rank = None
            return

        # Sort by energy (descending)
        sorted_pop = sorted(self.population, key=lambda c: c.energy, reverse=True)
        top_n = max(1, int(len(sorted_pop) * 0.01))
        top_group = sorted_pop[:top_n]

        # If the existing selected creature is still in the top group, keep it
        if self.selected_creature in top_group:
            self.selected_rank = sorted_pop.index(self.selected_creature)
            return

        # Otherwise pick the oldest creature from the top group
        oldest_in_top = max(
            top_group,
            key=lambda c: self.current_step - c.birth_step
        )

        self.selected_creature = oldest_in_top
        self.selected_rank = sorted_pop.index(oldest_in_top)

    def download_price_data(self, tickers, start_date, end_date, dest_dir):
        # create dest_dir if not exist
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

    def reload_data(self, filepath):
        data_dir = './data/prices'
        # Get all the columns from the CSV file
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        columns = [col.strip() for col in df.columns]  # Remove any leading/trailing whitespace from column names
        # remove 'Date' from the columns
        num_stocks = len(columns) // 2        
        columns = columns[:num_stocks]
        # remove the _price suffix from column name
        columns = [col.replace('_price', '') for col in columns]
        columns = [col.replace('_vol', '') for col in columns]
        tickers = [ticker.replace(' ', '-').strip() for ticker in columns]
        
        start = '1950-01-01'
        self.download_price_data(tickers, start, None, data_dir)
        self.write_log(f"Downloaded price data for {len(columns)} stocks")
        # create the new input data
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
        log_returns_df_vol = np.log(combined_df_vol / (combined_df_vol.shift(1) + 1e-9 ) + 1e-9)
        # drop the first row
        #log_returns_df_vol = log_returns_df_vol.iloc[1:]
        log_returns_df_vol = log_returns_df_vol.dropna()  # Drop the first row with

        price_columns = [col for col in log_returns_df_vol.columns if col.endswith('_price')]
        volume_columns = [col for col in log_returns_df_vol.columns if col.endswith('_vol')]
        log_returns_df_vol = log_returns_df_vol[price_columns + volume_columns]
        # find out the earliest date in the dataframe
        # back up filepath, then create it again
        backup_path = filepath + ".backup"
        os.rename(filepath, backup_path)
        log_returns_df_vol.to_csv(filepath, index_label='Date')
        self.write_log(f"Backed up original file to {backup_path} and wrote new data to {filepath}")

    def load_data(self, filepath, reload=False):
        """
        Load stock data from CSV.

        Assumption (default): columns are arranged as:
          [RET(stock_1), ..., RET(stock_N), VOL_LOGDIFF(stock_1), ..., VOL_LOGDIFF(stock_N)]
        with the same stock order in both halves. Total number of columns must be even.
        """
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
        # remove the _price suffix from column name
        columns = [col.replace('_price', '') for col in columns]
        columns = [col.replace('_vol', '') for col in columns]
        self.tickers = [ticker.replace(' ', '-').strip() for ticker in columns]
        # First half = price log-returns (used for PnL); second half = log diff of volume.
        ret_cols = df.columns[:num_stocks]
        vol_cols = df.columns[num_stocks:]

        returns = df[ret_cols].to_numpy(dtype=float)
        volumes = df[vol_cols].to_numpy(dtype=float)
        features = np.concatenate([returns, volumes], axis=1)  # order: [RET..., VOL...]

        self.stock_data = df
        self.num_stocks = num_stocks
        self.input_size = 2 * num_stocks
        self.returns = returns
        self.features = features

        self.write_log(
            f"Loaded data with {self.num_stocks} stocks (features per day: {self.input_size}) "
            f"from {df.index[0].date()} to {df.index[-1].date()}"
        )
        self.write_log(f"Total trading days: {len(df)}")

    def initialize_population(self, initial_size=100):
        """Initialize the population with random creatures"""
        for _ in range(initial_size):
            creature = LSTMCreature(
                input_size=self.input_size,
                num_stocks=self.num_stocks,
                hidden_size=64,
                num_layers=2,
                long_only=self.long_only
            )
            creature.birth_step = self.current_step
            self.population.append(creature)
        self.write_log(f"Initialized population with {initial_size} creatures")

    def calculate_portfolio_return(self, creature, returns_log, previous_portfolio, new_portfolio, eps=1e-12):
        """Calculate returns considering commission; returns_log is price log-returns (len=num_stocks)."""
        # Calculate portfolio change (excluding cash)
        portfolio_change = np.abs(new_portfolio - previous_portfolio)
        turnover = np.sum(portfolio_change[:-1])              # No turnover on cash
        cost_frac = self.commission_rate * turnover           # fraction of wealth

        # --- Portfolio return (exact with log returns) ---
        # simple return using start-of-day weights (trade at end-of-day assumption)
        gross_simple = np.dot(previous_portfolio[:-1], np.expm1(returns_log))  # scalar
        gross_log = np.log(max(1.0 + gross_simple, eps))

        net_log_return = gross_log + np.log(max(1.0 - cost_frac, eps))         # scalar
        return net_log_return

    def write_log(self, message):
        """Write a message to the log file"""
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def step(self, day_idx):
        """Execute one trading day with multi-threaded creature evaluation."""
        if day_idx >= len(self.stock_data):
            return False

        # NEW: get current date for output records
        current_date = self.stock_data.index[day_idx]

        # Full feature vector for the model (RET+VOL), and price returns for PnL
        x_features = self.features[day_idx]              # shape: (2*num_stocks,)
        r_price = self.returns[day_idx]                  # shape: (num_stocks,)

        features_tensor = torch.as_tensor(x_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # If nobody's alive, log and move on
        if not self.population:
            self.write_log(
                f"Step {self.current_step}: Alive: 0, Died: 0, Total dead: {self.dead_count}, "
                f"Avg Energy: 0.00, Max Energy: 0.00, Best ID: N/A (age: 0), Max Age: 0, Avg Age: 0.0"
            )
            # NEW: also clear selected creature when population is empty
            self.selected_creature = None
            self.selected_rank = None

            self.current_step += 1
            return True

        # Decide an executor size; threads work well with PyTorch’s internal native ops
        max_workers = min(64, (os.cpu_count() or 4) * 2)

        # Run all creatures in parallel
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for creature in self.population:
                futures.append(executor.submit(self._evaluate_creature, creature, r_price, features_tensor))
            for fut in as_completed(futures):
                results.append(fut.result())

        # Apply updates on the main thread to avoid races on shared lists
        energies = []
        ages = []
        creatures_to_remove = []

        # NEW: map creature -> today's portfolio_log_return
        portfolio_return_map = {}

        for creature, new_portfolio, new_energy, portfolio_log_return, age,  alive in results:
            portfolio_return_map[creature] = portfolio_log_return
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
            # record lightweight death metadata
            self.dead_count += 1
            self.dead_meta.append({
                "id": creature.creature_id,
                "birth_step": creature.birth_step,
                "death_step": self.current_step,
                "age": self.current_step - creature.birth_step,
                "energy_at_death": float(getattr(creature, "energy", 0.0)),
            })

            # actively drop large refs (helps GC)
            creature.hidden = None
            creature.portfolio = None
            creature.fitness_history = None
            del creature

        if self.current_step % 200 == 0:
            gc.collect()

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

        # NEW: update the stable selected creature based on top 10%
        self.update_selected_creature()

        # write to the log file (main step summary)
        self.write_log(
            f"Step {self.current_step}: "
            f"Date: {current_date}, "
            f"Alive: {len(self.population)}, "
            f"Died: {dead_count}, "
            f"Total dead: {total_dead}, "
            f"Avg Energy: {avg_energy:.3f}, "
            f"Max Energy: {max_energy:.3f}, "
            f"Best ID: {best_id} (age: {best_age}), "
            f"Max Age: {max_age}, "
            f"Avg Age: {avg_age:.1f}"
        )

        # NEW: log info about current selected creature and record cumulative log-return
        if self.selected_creature is not None:
            sel_age = self.current_step - self.selected_creature.birth_step
            rank_str = (
                f"{self.selected_rank + 1}/{len(self.population)}"
                if self.selected_rank is not None else f"?/{len(self.population)}"
            )
            self.write_log(
                f"  Selected Creature: ID {self.selected_creature.creature_id}, "
                f"Energy: {self.selected_creature.energy:.3f}, "
                f"Age: {sel_age}, Rank: {rank_str}"
            )

            # NEW: grow asset with the selected creature's portfolio return
            sel_ret = portfolio_return_map.get(self.selected_creature)
            if sel_ret is not None:
                self.asset *= float(np.exp(sel_ret))   # no DAILY_COST here
                self.asset_history.append(self.asset)
                cum_log = float(np.log(self.asset))
                # record (Date, TotalLogReturn)
                self.output_records.append((current_date, sel_ret))
                self._append_output_record(current_date, sel_ret)
                self.write_log(f"  Global Asset (selected creature): {self.asset:.6f}")

        # Reproduction every 128 steps (skip step 0)
        if self.current_step > 0 and self.current_step % 128 == 0:
            self.reproduce(avg_energy)

        if self.current_step % 10 == 0:
            # print out current portfolio for the best creature in one line
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

            for creature in bottom_creatures:
                self.population.remove(creature)
                self.dead_count += 1

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
        child = LSTMCreature(
            input_size=self.input_size,
            num_stocks=self.num_stocks,
            hidden_size=64,
            num_layers=2,
            long_only=self.long_only
        )

        # Crossover weights
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(child.parameters(),
                                                       parent1.parameters(),
                                                       parent2.parameters()):
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
        for creature in culled:
            self.dead_count += 1
            self.dead_meta.append({
                "id": creature.creature_id,
                "birth_step": creature.birth_step,
                "death_step": self.current_step,
                "age": self.current_step - creature.birth_step,
                "energy_at_death": float(getattr(creature, "energy", 0.0)),
                "reason": "cull_to_max_population",
            })
            creature.hidden = None
            creature.portfolio = None
            creature.fitness_history = None
            del creature

        self.population = self.population[:self.max_population]

        if self.current_step % 200 == 0:
            gc.collect()

        self.population = self.population[:self.max_population]

    def run_simulation(self, end_day=None):
        """Run the full simulation"""
        if self.stock_data is None:
            raise ValueError("Please load data first using load_data()")

        if end_day is None:
            end_day = len(self.stock_data) - 1  # inclusive

        # NEW: reset asset and records for this run
        self.asset_history = [self.asset]
        self.output_records = []

        # NEW: prepare output CSV for this run
        self._init_output_csv()
        start_day = self.current_step
        self.write_log(f"\nStarting simulation from day {start_day} to day {end_day}")
        self.write_log("=" * 80)
        
        for day in range(start_day, end_day + 1):
            if not self.step(day):
                break

            # Check if population is extinct
            if len(self.population) == 0:
                print("\nPOPULATION EXTINCT!")
                break

            if self.checkpoint_dir and (self.current_step % self.checkpoint_interval) == 0 and self.current_step > 0:
               self.save_checkpoint()
        
        if self.checkpoint_dir:
            self.save_checkpoint()

        self.write_log("\n=== SIMULATION COMPLETE ===")
        self.write_log(f"Final population: {len(self.population)}")
        self.write_log(f"Total deaths: {self.dead_count}")

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

            selected_portfolio_str = ", ".join([f"{w:.3f}" for w in self.selected_creature.portfolio])
            self.write_log(f"Selected Creature ID {self.selected_creature.creature_id} Portfolio: [{selected_portfolio_str}]")
            print(f"number of tickers: {len(self.tickers)}, number in portfolio: {len(self.selected_creature.portfolio)}")
            self.tickers.append("__CASH__")
            assert(len(self.selected_creature.portfolio) == len(self.tickers))
            
            data = [
                {"id": ticker, "weight": float(weight)}
                for ticker, weight in zip(self.tickers, self.selected_creature.portfolio)
            ]
            # get current date in YYYYMMDD format
            current_date_str = self.stock_data.index[self.current_step - 1].strftime("%Y%m%d")
            json_filename = f"ticket_to_buy_{current_date_str}.json"
            if self.output_json_dir:
                json_path = os.path.join(self.output_json_dir, json_filename)
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
                self.write_log(f"Saved selected creature portfolio to JSON: {json_path}")


    # NEW: save (Date, TotalLogReturn) to CSV
    def save_selected_log_returns(self, output_file: str):
        """
        Save the cumulative log return of the selected creature to CSV.

        Columns:
          - Date
          - TotalLogReturn  (cumulative log-return up to that date)
        """
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


    def _evaluate_creature(self, creature, returns_price_vec: np.ndarray, features_tensor: torch.Tensor):
        """
        Thread worker: run forward pass (2N features), compute return using price returns (N),
        and return a tuple for main-thread application.
        """
        with torch.no_grad():
            # Forward pass for this creature (keeps its own hidden state)
            new_portfolio_t = creature(features_tensor).squeeze()
            new_portfolio = new_portfolio_t.detach().cpu().numpy()

        # Compute portfolio return using current (previous) portfolio vs. new one
        portfolio_log_return = self.calculate_portfolio_return(
            creature, returns_price_vec, creature.portfolio, new_portfolio
        )

        # Energy update: multiplicative gain/loss on current energy, minus daily cost
        new_energy = creature.energy * np.exp(portfolio_log_return) - DAILY_COST

        # Age for stats (do not mutate here; main thread will)
        age = self.current_step - creature.birth_step

        # Alive flag based on new energy
        alive = new_energy > 0.0

        return creature, new_portfolio, new_energy, portfolio_log_return, age, alive

    # NEW: initialize CSV file (called at start of run)
    def _init_output_csv(self):
        """
        Prepare the CSV file for incremental logging.
        If file exists, remove it and write header again.
        """
        if self.output_file is None:
            return

        # Reset state
        self._output_file_initialized = False

        # Remove old file if exists
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        # Write header
        with open(self.output_file, "w") as f:
            f.write("Date,LogReturn\n")

        self._output_file_initialized = True

    # NEW: append a single (date, log_return) row
    def _append_output_record(self, date, log_return):
        """
        Append one row to the CSV for the selected creature.
        """
        if self.output_file is None:
            return

        # Ensure header exists if something went wrong
        if not self._output_file_initialized or not os.path.exists(self.output_file):
            self._init_output_csv()

        # date is a pandas Timestamp; use ISO format
        with open(self.output_file, "a") as f:
            f.write(f"{date.isoformat()},{log_return}\n")

def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]  # <-- skip script name

    # Defaults
    max_population = 2000
    log_file = "output.log"
    end_day = None
    input_file = "stock_data_vol.csv"
    output_file = None  # NEW: default, derived from input_file if not provided
    long_only = False
    checkpoint_dir       = None
    checkpoint_interval  = 4000    # ← default interval
    reload = False
    try:
        # short options: -m, -l, -i, -e, -o, -h
        # long options require '=' when they take a value
        opts, args = getopt.getopt(
            argv,
            "m:l:i:e:o:j:c:k:Lrh",
            ["max-population=", "log-file=", "input-file=", "end-day=", "output-file=", "json-output-dir=", "checkpoint-dir=", "checkpoint-interval=", "long-only", "reload", "help"]
        )
    except getopt.GetoptError as e:
        print(f"Error: {e}")
        print("Usage: script.py [-m N | --max-population N] "
              "[-l FILE | --log-file FILE] "
              "[-i FILE | --input-file FILE] "
              "[-e N | --end-day N] "
              "[-o FILE | --output-file FILE]"
              "[-j JSON_OUTPUT_DIR | --json-output-dir JSON_OUTPUT_DIR]"
              "[-c CHECKPOINT_DIR | --checkpoint-dir CHECKPOINT_DIR]"
              "[-k N|--checkpoint-interval N] "
              "[-r Reload]"
              "[-L long-only]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: script.py [-m N | --max-population N] "
                  "[-l FILE | --log-file FILE] "
                  "[-i FILE | --input-file FILE] "
                  "[-e N | --end-day N] "
                  "[-o FILE | --output-file FILE]"
                  "[-j JSON_OUTPUT_DIR | --json-output-dir JSON_OUTPUT_DIR]"
                  "-c CHECKPOINT_DIR | --checkpoint-dir CHECKPOINT_DIR"
                  "[-k N|--checkpoint-interval N] "
                  "[-r Reload]"
                  "[-L long-only]")
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
        elif opt in ("-i", "--input-file"):
            input_file = arg
        elif opt in ("-e", "--end-day"):
            # convert string to int
            end_day = int(arg)
        elif opt in ("-o", "--output-file"):
            output_file = arg            
        elif opt in ("-j", "--json-output-dir"):
            output_json_dir = arg
        elif opt in ("-c","--checkpoint-dir"):
            checkpoint_dir = arg
        elif opt in ("-k","--checkpoint-interval"):
            checkpoint_interval = int(arg)
        elif opt in ("-L", "--long-only"):
            long_only = True
        elif opt in ("-r", "--reload"):
            reload = True

    return max_population, log_file, input_file, end_day, output_file, output_json_dir, checkpoint_dir, checkpoint_interval, long_only, reload, args  # args = leftover positional args, if any


# Main execution
if __name__ == "__main__":
    input_file_path = './input_files'
    output_file_path = './output_files'
    log_file_path = './log_files'

    # create output_file_path in the case it does not exist
    Path(output_file_path).mkdir(parents=True, exist_ok=True)
    Path(log_file_path).mkdir(parents=True, exist_ok=True)


    max_population, log_file, input_file, end_day, output_file, output_json_dir, checkpoint_dir, checkpoint_interval, long_only, reload, _ = parse_args()
    Path(output_json_dir).mkdir(parents=True, exist_ok=True)

    # Derive default output file name if not given
    input_file = f"{input_file_path}/{input_file}"
    output_file = f"{output_file_path}/{output_file}"
    log_file = f"{log_file_path}/{log_file}"

    print(f"Max Population: {max_population}, log file: {log_file}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    system = NEATTradingSystem(commission_rate=0.001, max_population=max_population, log_file=log_file, output_json_dir=output_json_dir, checkpoint_dir=checkpoint_dir, 
                               checkpoint_interval=checkpoint_interval, long_only=long_only)
    
    system.output_file = output_file
    # Load data (expects returns then volume log-diffs)
    system.load_data(input_file, reload=reload)
    
    if checkpoint_dir:
        system.load_checkpoint()
        if len(system.population) == 0:
            system.initialize_population(initial_size=1000)
            # if output file exists, remove it
            if os.path.exists(output_file):
                print("Removing existing output file:", output_file)
                os.remove(output_file)

            if os.path.exists(log_file):
                print("Removing existing log file:", log_file)
                os.remove(log_file)

    # Run simulation (adjust end_day to your dataset length)
    system.run_simulation(end_day=end_day)
