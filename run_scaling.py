"""
Scaling experiment: how does each strategy's fairness and starvation rate
change as the number of competing agents increases?

Agent counts: 2, 4, 6, 8, 12, 16, 20, 30
Each count averaged over 10 seeds.

Usage:
    python run_scaling.py
    python run_scaling.py --ticks 200 --seeds 10
"""

import argparse
import math
import os
import random

import matplotlib.pyplot as plt

from agents import Agent
from baselines import (ExponentialBackoffAgent, PriorityAgingAgent,
                       AIMDAgent, GreedyAgent, RandomAgent,
                       UCB1Agent, WinRateAdaptiveAgent)
from simulation import Simulation
from metrics import compute_all


AGENT_COUNTS = [2, 4, 6, 8, 12, 16, 20, 30]

STRATEGIES = [
    ('Emotion (blind)',     'emotion'),
    ('Priority Aging',     'aging'),
    ('Exp Backoff',        'backoff'),
    ('AIMD',               'aimd'),
    ('Random',             'random'),
    ('UCB1',               'ucb1'),
    ('Win-Rate Adaptive',  'winrate'),
    ('Greedy',             'greedy'),
]


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def make_agents(kind, n, seed):
    rng = random.Random(seed)

    def attrs():
        return dict(
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
        )

    agents = []
    for i in range(n):
        a = attrs()
        if kind == 'emotion':
            agents.append(Agent(
                agent_id=i,
                persistence=round(rng.uniform(0.2, 0.8), 2),
                risk_aversion=round(rng.uniform(0.1, 0.5), 2),
                **a,
            ))
        elif kind == 'aging':
            agents.append(PriorityAgingAgent(agent_id=i, **a))
        elif kind == 'backoff':
            agents.append(ExponentialBackoffAgent(agent_id=i, **a))
        elif kind == 'aimd':
            agents.append(AIMDAgent(agent_id=i, **a))
        elif kind == 'random':
            agents.append(RandomAgent(agent_id=i, **a))
        elif kind == 'ucb1':
            agents.append(UCB1Agent(agent_id=i, **a))
        elif kind == 'winrate':
            agents.append(WinRateAdaptiveAgent(agent_id=i, **a))
        elif kind == 'greedy':
            agents.append(GreedyAgent(agent_id=i, **a))
    return agents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def mean(xs): return sum(xs) / len(xs)
def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticks', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    # results[strategy_name][n] = {'fairness': [...], 'starvation': [...], 'deviation': [...]}
    results = {
        name: {n: {'fairness': [], 'starvation': [], 'deviation': []} for n in AGENT_COUNTS}
        for name, _ in STRATEGIES
    }

    total = len(STRATEGIES) * len(AGENT_COUNTS) * len(seeds)
    done = 0

    for name, kind in STRATEGIES:
        for n in AGENT_COUNTS:
            for seed in seeds:
                agents = make_agents(kind, n, seed)
                sim = Simulation(agents=agents, num_ticks=args.ticks,
                                 social_awareness=False, visibility='blind',
                                 seed=seed)
                history = sim.run()
                r = compute_all(agents, history)
                results[name][n]['fairness'].append(r['jains_fairness'])
                results[name][n]['starvation'].append(r['starvation_rate'])
                results[name][n]['deviation'].append(r['ideal_share_deviation'])
                done += 1
            print(f"  {name:22s}  n={n:2d}  "
                  f"fairness={mean(results[name][n]['fairness']):.4f}  "
                  f"starvation={mean(results[name][n]['starvation']):.4f}")

    print(f"\nDone. Plotting...")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------

    colors = [
        '#4a90d9',  # Emotion — blue
        '#e67e22',  # Priority Aging — orange
        '#2ecc71',  # Exp Backoff — green
        '#e74c3c',  # AIMD — red
        '#9b59b6',  # Random — purple
        '#1abc9c',  # UCB1 — teal
        '#f39c12',  # Win-Rate Adaptive — yellow
        '#e91e8c',  # Greedy — pink
    ]

    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("Scaling: Strategy Performance vs Number of Agents", fontsize=13)

    for (name, _), color in zip(STRATEGIES, colors):
        xs = AGENT_COUNTS
        fair_means  = [mean(results[name][n]['fairness'])   for n in xs]
        fair_stds   = [std(results[name][n]['fairness'])    for n in xs]
        starv_means = [mean(results[name][n]['starvation']) for n in xs]
        dev_means   = [mean(results[name][n]['deviation'])  for n in xs]

        axes[0].plot(xs, fair_means, marker='o', label=name, color=color)
        axes[0].fill_between(xs,
                             [f - s for f, s in zip(fair_means, fair_stds)],
                             [f + s for f, s in zip(fair_means, fair_stds)],
                             alpha=0.12, color=color)

        axes[1].plot(xs, starv_means, marker='o', label=name, color=color)
        axes[2].plot(xs, dev_means,   marker='o', label=name, color=color)

    axes[0].set_title("Jain's Fairness Index (higher = fairer)")
    axes[0].set_xlabel("Number of Agents")
    axes[0].set_ylabel("Fairness")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(AGENT_COUNTS)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Starvation Rate (lower = better)")
    axes[1].set_xlabel("Number of Agents")
    axes[1].set_ylabel("Fraction of Agents Starved")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(AGENT_COUNTS)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Ideal Share Deviation (lower = more efficient)")
    axes[2].set_xlabel("Number of Agents")
    axes[2].set_ylabel("Mean Abs Deviation from Ideal (fraction)")
    axes[2].set_xticks(AGENT_COUNTS)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'results/scaling.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
