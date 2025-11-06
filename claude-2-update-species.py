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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Global / defaults
# =========================
DAILY_COST = 0
DEVICE = torch.device("cpu")  # keep CPU for thread-safety; you can route to CUDA if you disable ThreadPoolExecutor

# For reproducibility if you want:
# np.random.seed(42); torch.manual_seed(42); random.seed(42)

# =========================
# Helper: StackedLSTM with variable hidden sizes per layer
# =========================
class StackedLSTM(nn.Module):
    """
    A stack of independent nn.LSTM layers that allows different hidden sizes per layer.
    Input is (batch=1, seq=1, features=input_size).
    """
    def __init__(self, input_size: int, hidden_sizes: list[int]):
        super().__init__()
        assert len(hidden_sizes) >= 1
        self.hidden_sizes = list(hidden_sizes)
        self.layers = nn.ModuleList()
        in_size = input_size
        for h in self.hidden_sizes:
            self.layers.append(nn.LSTM(in_size, h, num_layers=1, batch_first=True))
            in_size = h

    def forward(self, x, hidden):
        """
        x: (1, 1, in_features). hidden: list of (h, c) pairs per layer or None.
        Returns: last_layer_output (1, 1, last_hidden), new_hidden_list
        """
        new_hidden = []
        out = x
        for i, lstm in enumerate(self.layers):
            h_i, c_i = (None, None)
            if hidden is not None and hidden[i] is not None:
                h_i, c_i = hidden[i]
            out, (h_o, c_o) = lstm(out, None if h_i is None else (h_i, c_i))
            new_hidden.append((h_o, c_o))
        return out, new_hidden

    def init_hidden(self, batch_size: int = 1):
        hiddens = []
        for h in self.hidden_sizes:
            hiddens.append((
                torch.zeros(1, batch_size, h, device=DEVICE),
                torch.zeros(1, batch_size, h, device=DEVICE),
            ))
        return hiddens

# =========================
# Genome encodes architecture (+ hyperparams) for NEAT-style evolution
# =========================
class Genome:
    """
    Encodes the architecture of a variable-depth LSTM:
      - hidden_sizes: list[int] for each stacked LSTM layer.
    """
    def __init__(self, input_size: int, output_size: int, hidden_sizes: Optional[list[int]] = None):
        self.input_size = input_size
        self.output_size = output_size
        # If not provided, start with a simple 1-2 layer architecture
        if hidden_sizes is None:
            # small initial sizes so population can grow
            hidden_sizes = [64] if random.random() < 0.5 else [64, 64]
        self.hidden_sizes = list(hidden_sizes)

    def clone(self) -> "Genome":
        g = Genome(self.input_size, self.output_size, self.hidden_sizes.copy())
        return g

    # Compatibility distance for speciation (simple, architecture-only)
    def distance(self, other: "Genome",
                 c_layers: float = 1.0,
                 c_size: float = 0.5) -> float:
        # layer count difference
        d_layers = abs(len(self.hidden_sizes) - len(other.hidden_sizes))
        # hidden size differences for aligned layers
        L = min(len(self.hidden_sizes), len(other.hidden_sizes))
        if L > 0:
            d_sizes = sum(abs(self.hidden_sizes[i] - other.hidden_sizes[i]) for i in range(L)) / L
        else:
            d_sizes = 0.0
        return c_layers * d_layers + c_size * d_sizes

    # Structural mutations
    def mutate_structure(self,
                         p_add_layer=0.10,
                         p_remove_layer=0.07,
                         p_resize_layer=0.20,
                         min_layers=1,
                         max_layers=5,
                         min_hidden=16,
                         max_hidden=256,
                         resize_step=(8, 32)):
        # Add a layer
        if random.random() < p_add_layer and len(self.hidden_sizes) < max_layers:
            # Insert at random position with a sampled size
            pos = random.randint(0, len(self.hidden_sizes))
            new_h = random.randint(min_hidden, max_hidden)
            self.hidden_sizes.insert(pos, new_h)

        # Remove a layer
        if random.random() < p_remove_layer and len(self.hidden_sizes) > min_layers:
            pos = random.randrange(len(self.hidden_sizes))
            del self.hidden_sizes[pos]

        # Resize a random layer
        if random.random() < p_resize_layer and len(self.hidden_sizes) > 0:
            pos = random.randrange(len(self.hidden_sizes))
            step = random.randint(resize_step[0], resize_step[1])
            if random.random() < 0.5:
                self.hidden_sizes[pos] = max(min_hidden, self.hidden_sizes[pos] - step)
            else:
                self.hidden_sizes[pos] = min(max_hidden, self.hidden_sizes[pos] + step)

# =========================
# Creature = (Genome + Model + Params)
# =========================
class LSTMCreature(nn.Module):
    """NEAT-style LSTM creature with variable architecture from Genome."""
    def __init__(self, input_size: int, output_size: int, genome: Optional[Genome] = None, creature_id=None):
        super().__init__()
        self.creature_id = creature_id or random.randint(0, 100000)
        self.input_size = input_size
        self.output_size = output_size

        # Genome
        self.genome = genome or Genome(input_size, output_size)

        # Build model from genome
        self._build_from_genome()

        # State
        self.hidden = None
        self.energy = 1.0
        self.portfolio = np.zeros(self.output_size)
        self.portfolio[-1] = 1.0  # start 100% cash
        self.birth_step = 0
        self.fitness_history = []

    def _build_from_genome(self):
        # Variable LSTM stack + linear head
        self.lstm_stack = StackedLSTM(self.input_size, self.genome.hidden_sizes).to(DEVICE)
        last_hidden = self.genome.hidden_sizes[-1]
        self.fc = nn.Linear(last_hidden, self.output_size).to(DEVICE)

    def forward(self, x: torch.Tensor):
        # x: (1,1,input_size)
        if self.hidden is None or len(self.hidden) != len(self.genome.hidden_sizes):
            self.hidden = self.lstm_stack.init_hidden(batch_size=x.size(0))
        out, self.hidden = self.lstm_stack(x, self.hidden)
        out = self.fc(out[:, -1, :])  # (1, output_size)

        # Signed-softmax trick to allow shorts but constrain L1 to 1
        signs = torch.tanh(out)                          # [-1, 1]
        abs_w = F.softmax(torch.abs(out), dim=-1)        # sum(abs_w)=1
        w = signs * abs_w
        w = w / torch.sum(torch.abs(w), dim=-1, keepdim=True).clamp_min(1e-12)
        return w

    def reset_hidden(self):
        self.hidden = None

    def mutate_weights(self, mutation_rate=0.1, mutation_strength=0.1):
        with torch.no_grad():
            for p in self.parameters():
                if p.ndim == 0:  # scalar guard
                    continue
                mask = (torch.rand_like(p) < mutation_rate).float()
                noise = torch.randn_like(p) * mutation_strength
                p.add_(mask * noise)

    def mutate_structure(self):
        # Save old params to partially transfer where possible
        old_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        old_genome = self.genome.clone()

        # Mutate genome
        self.genome.mutate_structure()

        # Rebuild network per new genome
        self._build_from_genome()

        # Try partial weight inheritance (layer-aligned where shapes match)
        with torch.no_grad():
            new_state = self.state_dict()
            for k in new_state.keys():
                if k in old_state and old_state[k].shape == new_state[k].shape:
                    new_state[k].copy_(old_state[k])
            self.load_state_dict(new_state)

        # Reset hidden since layer layout changed
        self.reset_hidden()

    def clone_like(self):
        # Deep copy genome and new network with copied weights
        child = LSTMCreature(self.input_size, self.output_size, genome=self.genome.clone())
        child.load_state_dict(copy.deepcopy(self.state_dict()))
        child.reset_hidden()
        return child

# =========================
# Species container
# =========================
class Species:
    def __init__(self, representative: LSTMCreature, species_id: int):
        self.species_id = species_id
        self.representative = representative  # creature
        self.members: list[LSTMCreature] = [representative]
        self.age = 0

    def add(self, c: LSTMCreature):
        self.members.append(c)

    def update_representative(self):
        # pick new rep (e.g., median or best energy); here best energy
        self.representative = max(self.members, key=lambda x: x.energy)

# =========================
# Trading System (NEAT + speciation)
# =========================
class NEATTradingSystem:
    """NEAT-based trading system with LSTM creatures and speciation."""
    def __init__(self, commission_rate=0.001, max_population=100000, log_file="output.log",
                 speciation_threshold=25.0,  # compatibility threshold; tune as needed
                 speciation_c_layers=2.0,
                 speciation_c_size=0.2):
        self.commission_rate = commission_rate
        self.max_population = max_population
        self.population: list[LSTMCreature] = []
        self.dead_creatures: list[LSTMCreature] = []
        self.current_step = 0
        self.stock_data = None
        self.num_stocks = None
        self.log_file = log_file

        # Speciation parameters
        self.species: list[Species] = []
        self.next_species_id = 1
        self.speciation_threshold = speciation_threshold
        self.speciation_c_layers = speciation_c_layers
        self.speciation_c_size = speciation_c_size

    # ---------- Data ----------
    def load_data(self, filepath='stock_data.csv'):
        self.stock_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.num_stocks = len(self.stock_data.columns)
        self.write_log(f"Loaded data with {self.num_stocks} stocks from {self.stock_data.index[0]} to {self.stock_data.index[-1]}")
        self.write_log(f"Total trading days: {len(self.stock_data)}")

    # ---------- Init ----------
    def initialize_population(self, initial_size=100):
        self.population.clear()
        output_size = self.num_stocks + 1
        for _ in range(initial_size):
            creature = LSTMCreature(self.num_stocks, output_size)
            creature.birth_step = self.current_step
            self.population.append(creature)
        self.write_log(f"Initialized population with {initial_size} creatures")
        self._speciate_population()

    # ---------- Returns ----------
    def calculate_portfolio_return(self, creature, returns_log, previous_portfolio, new_portfolio, eps=1e-12):
        # turnover on non-cash
        portfolio_change = np.abs(new_portfolio - previous_portfolio)
        turnover = np.sum(portfolio_change[:-1])  # ignore cash bucket
        cost_frac = self.commission_rate * turnover

        # Use start-of-day weights for simple gross return
        gross_simple = np.dot(previous_portfolio[:-1], np.expm1(returns_log))
        gross_log = np.log(max(1.0 + gross_simple, eps))
        net_log_return = gross_log + np.log(max(1.0 - cost_frac, eps))
        return net_log_return

    # ---------- Logging ----------
    def write_log(self, message):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    # ---------- Step ----------
    def step(self, day_idx):
        if day_idx >= len(self.stock_data):
            return False

        returns = self.stock_data.iloc[day_idx].values
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)

        if not self.population:
            self.write_log(f"Step {self.current_step}: Alive: 0, Died: 0, Total dead: {len(self.dead_creatures)}, "
                           f"Avg Energy: 0.00, Max Energy: 0.00, Best ID: N/A (age: 0), Max Age: 0, Avg Age: 0.0")
            self.current_step += 1
            return True

        max_workers = min(64, (os.cpu_count() or 4) * 2)

        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for creature in self.population:
                futures.append(executor.submit(self._evaluate_creature, creature, returns, returns_tensor))
            for fut in as_completed(futures):
                results.append(fut.result())

        energies, ages = [], []
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

        for creature in creatures_to_remove:
            if creature in self.population:
                self.population.remove(creature)
            self.dead_creatures.append(creature)

        if energies:
            avg_energy = float(np.mean(energies))
            max_energy = float(np.max(energies))
            avg_age = float(np.mean(ages))
            max_age = int(np.max(ages))
            best_creature = max(self.population, key=lambda x: x.energy)
            best_id = best_creature.creature_id
            best_age = self.current_step - best_creature.birth_step
        else:
            avg_energy = max_energy = avg_age = 0.0
            max_age = 0
            best_id = "N/A"
            best_age = 0

        dead_count = len(creatures_to_remove)
        total_dead = len(self.dead_creatures)

        self.write_log(f"Step {self.current_step}: "
                       f"Alive: {len(self.population)}, "
                       f"Died: {dead_count}, "
                       f"Total dead: {total_dead}, "
                       f"Avg Energy: {avg_energy:.3f}, "
                       f"Max Energy: {max_energy:.3f}, "
                       f"Best ID: {best_id} (age: {best_age}), "
                       f"Max Age: {max_age}, "
                       f"Avg Age: {avg_age:.1f}")

        if self.current_step % 10 == 0 and self.population:
            best_creature = max(self.population, key=lambda x: x.energy)
            portfolio_str = ", ".join([f"{w:.3f}" for w in best_creature.portfolio])
            arch = "-".join(map(str, best_creature.genome.hidden_sizes))
            self.write_log(f"  Best ID {best_creature.creature_id} Arch[{arch}] Portfolio: [{portfolio_str}]")

        # Reproduction every 128 steps (skip 0)
        if self.current_step > 0 and self.current_step % 128 == 0:
            self.reproduce(avg_energy)

        self.current_step += 1
        return True

    # ---------- Evaluation worker ----------
    def _evaluate_creature(self, creature, returns: np.ndarray, returns_tensor: torch.Tensor):
        with torch.no_grad():
            new_portfolio_t = creature(returns_tensor).squeeze()
            new_portfolio = new_portfolio_t.detach().cpu().numpy()

        portfolio_log_return = self.calculate_portfolio_return(
            creature, returns, creature.portfolio, new_portfolio
        )
        new_energy = creature.energy * np.exp(portfolio_log_return) - DAILY_COST
        age = self.current_step - creature.birth_step
        alive = new_energy > 0.0
        return creature, new_portfolio, new_energy, portfolio_log_return, age, alive

    # ---------- Speciation ----------
    def _speciate_population(self):
        # reset species
        self.species = []
        for c in self.population:
            placed = False
            for sp in self.species:
                d = c.genome.distance(sp.representative.genome,
                                      c_layers=self.speciation_c_layers,
                                      c_size=self.speciation_c_size)
                if d < self.speciation_threshold:
                    sp.add(c)
                    placed = True
                    break
            if not placed:
                sp = Species(representative=c, species_id=self.next_species_id)
                self.next_species_id += 1
                self.species.append(sp)
        # optional: update reps
        for sp in self.species:
            sp.update_representative()

    # ---------- Reproduction with species protection ----------
    def reproduce(self, avg_energy):
        if len(self.population) < 2:
            self.write_log("Not enough creatures to reproduce")
            return

        self.write_log("\n=== REPRODUCTION EVENT ===")
        # (Re)assign species before mating
        self._speciate_population()
        self.write_log(f"Species count: {len(self.species)}")

        # Optional cull if near max: remove weakest across whole pop
        if len(self.population) >= int(self.max_population * 0.9):
            keep_n = int(self.max_population * 0.7)
            self.write_log(f"Culling: {len(self.population)} -> {keep_n} before reproduction")
            self.population.sort(key=lambda x: x.energy, reverse=True)
            culled = self.population[keep_n:]
            self.dead_creatures.extend(culled)
            self.population = self.population[:keep_n]
            # Re-speciate after culling
            self._speciate_population()

        total_offspring = 0
        new_children: list[LSTMCreature] = []

        # Per-species reproduction
        for sp in self.species:
            sp.members.sort(key=lambda x: x.energy, reverse=True)
            n = len(sp.members)
            if n == 0:
                continue

            # Elitism per species (top 30%)
            elite_count = max(1, int(n * 0.30))
            elites = sp.members[:elite_count]

            # Allocate offspring proportional to species size (simple)
            # You could weight by average adjusted fitness. Here proportional by size:
            species_quota = max(1, int((len(self.population) * 0.6) * (n / max(1, len(self.population)))))

            self.write_log(f"Species {sp.species_id}: members={n}, elites={elite_count}, quota={species_quota}")

            # Breed within-species
            for _ in range(species_quota):
                p1 = random.choice(elites)
                p2 = random.choice(sp.members)
                if p1 is p2 and n > 1:
                    # avoid exact same object if possible
                    p2 = random.choice(sp.members[1:] if len(sp.members) > 1 else sp.members)

                child = self._crossover(p1, p2)

                # Structural mutation chance
                if random.random() < 0.20:
                    child.mutate_structure()

                # Always allow some weight mutation
                child.mutate_weights(mutation_rate=0.10, mutation_strength=0.10)

                child.birth_step = self.current_step
                child.energy = max(0.5 * avg_energy, 1.0)  # seed with avg (or 1.0 min)
                new_children.append(child)
                total_offspring += 1

        # Add children and clamp to max
        self.population.extend(new_children)
        if len(self.population) > self.max_population:
            self.cull_population()

        # Log elites
        all_sorted = sorted(self.population, key=lambda x: x.energy, reverse=True)
        self.write_log(f"Created {total_offspring} offspring; population now {len(self.population)}")
        self.write_log("Top 5 after reproduction:")
        for i, c in enumerate(all_sorted[:5]):
            age = self.current_step - c.birth_step
            arch = "-".join(map(str, c.genome.hidden_sizes))
            self.write_log(f"  {i+1}. ID: {c.creature_id}, E={c.energy:.2f}, Age={age}, Arch[{arch}]")
        self.write_log("========================\n")

    def _crossover(self, p1: LSTMCreature, p2: LSTMCreature) -> LSTMCreature:
        """
        Layer-index alignment crossover for architecture and parameters.
        """
        # --- Architecture crossover ---
        h1, h2 = p1.genome.hidden_sizes, p2.genome.hidden_sizes
        L = min(len(h1), len(h2))
        child_hidden = []
        for i in range(L):
            child_hidden.append(h1[i] if random.random() < 0.5 else h2[i])
        # Inherit an extra tail with small probability
        tails = []
        if len(h1) > L:
            tails.extend(h1[L:])
        if len(h2) > L:
            tails.extend(h2[L:])
        if tails and random.random() < 0.30:
            child_hidden.append(random.choice(tails))

        # Ensure at least one layer
        if len(child_hidden) == 0:
            child_hidden = [random.choice(h1 if h1 else [64])]

        child_genome = Genome(p1.input_size, p1.output_size, hidden_sizes=child_hidden)
        child = LSTMCreature(p1.input_size, p1.output_size, genome=child_genome)

        # --- Parameter crossover (where shapes match) ---
        with torch.no_grad():
            c_state = child.state_dict()
            p1_state = p1.state_dict()
            p2_state = p2.state_dict()
            for k in c_state.keys():
                s1 = p1_state.get(k, None)
                s2 = p2_state.get(k, None)
                if s1 is not None and s1.shape == c_state[k].shape and s2 is not None and s2.shape == c_state[k].shape:
                    mask = torch.rand_like(c_state[k]) < 0.5
                    c_state[k].copy_(torch.where(mask, s1, s2))
                elif s1 is not None and s1.shape == c_state[k].shape:
                    c_state[k].copy_(s1)
                elif s2 is not None and s2.shape == c_state[k].shape:
                    c_state[k].copy_(s2)
            child.load_state_dict(c_state)
        child.reset_hidden()
        return child

    def cull_population(self):
        self.write_log(f"Culling population from {len(self.population)} to {self.max_population}")
        self.population.sort(key=lambda x: x.energy, reverse=True)
        culled = self.population[self.max_population:]
        self.dead_creatures.extend(culled)
        self.population = self.population[:self.max_population]

    # ---------- Simulation ----------
    def run_simulation(self, start_day=0, end_day=None):
        if self.stock_data is None:
            raise ValueError("Please load data first using load_data()")
        if end_day is None:
            end_day = len(self.stock_data) - 1

        self.write_log(f"\nStarting simulation from day {start_day} to day {end_day}")
        self.write_log("=" * 80)

        for day in range(start_day, end_day + 1):
            if not self.step(day):
                break
            if len(self.population) == 0:
                self.write_log("\nPOPULATION EXTINCT!")
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

            self.write_log("\nTop 5 creatures:")
            top_5 = sorted(self.population, key=lambda x: x.energy, reverse=True)[:5]
            for i, creature in enumerate(top_5):
                age = self.current_step - creature.birth_step
                arch = "-".join(map(str, creature.genome.hidden_sizes))
                self.write_log(f"  {i+1}. ID: {creature.creature_id}, Energy: {creature.energy:.2f}, Age: {age}, Arch[{arch}]")

# =========================
# CLI
# =========================
def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]  # <-- skip script name

    max_population = 1000
    log_file = "output.log"

    try:
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

    return max_population, log_file, args

# =========================
# Main
# =========================
if __name__ == "__main__":
    max_population, log_file, _ = parse_args()
    print(f"Max Population: {max_population}, log file: {log_file}")

    system = NEATTradingSystem(
        commission_rate=0.001,
        max_population=max_population,
        log_file=log_file,
        # You can tune these three for how clustered species are:
        speciation_threshold=25.0,
        speciation_c_layers=2.0,
        speciation_c_size=0.2
    )

    # Load data
    system.load_data('stock_data.csv')

    # Initialize population (you can start smaller and let it grow)
    system.initialize_population(initial_size=1000)

    # Run simulation
    system.run_simulation(start_day=0, end_day=10000)
