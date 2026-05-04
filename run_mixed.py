"""
Mixed population experiment: agents using different strategies compete simultaneously
for the same tier slots.

Instead of homogeneous populations (all emotion, all backoff, etc.), this runs a single
simulation where emotion agents, priority aging agents, and backoff agents all compete
against each other directly.

Usage:
    python run_mixed.py                        # defaults: 2 of each strategy, 500 ticks, 10 seeds
    python run_mixed.py --per-strategy 3       # 3 of each = 9 total agents
    python run_mixed.py --ticks 1000 --seeds 20
    python run_mixed.py --seed 42 --explain    # single seed with explanations
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
from metrics import jains_fairness_index, starvation_rate


# ---------------------------------------------------------------------------
# Mixed population factory
# ---------------------------------------------------------------------------

STRATEGY_LABELS = [
    'Emotion', 'Priority Aging', 'Exp Backoff',
    'AIMD', 'Greedy', 'Random', 'UCB1', 'Win-Rate Adaptive',
]


def make_mixed_population(per_strategy, seed=None):
    """
    Create a mixed population with `per_strategy` agents of each type.
    All agents share the same ID space and compete in the same simulation.
    Returns (agents, strategy_map) where strategy_map: agent_id -> strategy label.
    """
    rng = random.Random(seed)
    agents = []
    strategy_map = {}
    agent_id = 0

    def attrs():
        return dict(
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
        )

    specs = [
        ('Emotion',           lambda aid: Agent(agent_id=aid, persistence=round(rng.uniform(0.2, 0.8), 2), risk_aversion=round(rng.uniform(0.1, 0.5), 2), **attrs())),
        ('Priority Aging',    lambda aid: PriorityAgingAgent(agent_id=aid, **attrs())),
        ('Exp Backoff',       lambda aid: ExponentialBackoffAgent(agent_id=aid, **attrs())),
        ('AIMD',              lambda aid: AIMDAgent(agent_id=aid, **attrs())),
        ('Greedy',            lambda aid: GreedyAgent(agent_id=aid, **attrs())),
        ('Random',            lambda aid: RandomAgent(agent_id=aid, **attrs())),
        ('UCB1',              lambda aid: UCB1Agent(agent_id=aid, **attrs())),
        ('Win-Rate Adaptive', lambda aid: WinRateAdaptiveAgent(agent_id=aid, **attrs())),
    ]

    for label, factory in specs:
        for _ in range(per_strategy):
            a = factory(agent_id)
            strategy_map[agent_id] = label
            agents.append(a)
            agent_id += 1

    return agents, strategy_map


# ---------------------------------------------------------------------------
# Per-strategy metrics
# ---------------------------------------------------------------------------

def strategy_metrics(agents, strategy_map):
    """Aggregate metrics broken down by strategy."""
    groups = {label: [] for label in STRATEGY_LABELS}
    for agent in agents:
        label = strategy_map[agent.agent_id]
        groups[label].append(agent)

    results = {}
    for label, group in groups.items():
        if not group:
            continue
        rewards = [a.total_reward for a in group]
        total = sum(a.wins + a.losses for a in group)
        wins  = sum(a.wins for a in group)

        # Starvation: fraction of agents in group that went 20+ ticks without reward
        starved = sum(
            1 for a in group
            if len(a.state_log) >= 20 and not any(e['won'] for e in a.state_log[-20:])
        )

        results[label] = {
            'mean_reward':    sum(rewards) / len(rewards),
            'total_reward':   sum(rewards),
            'win_rate':       wins / total if total > 0 else 0.0,
            'starvation_rate': starved / len(group),
            'fairness':       jains_fairness_index(rewards),
            'agent_rewards':  rewards,
        }
    return results


def overall_fairness(agents):
    """Jain's fairness index across all agents regardless of strategy."""
    return jains_fairness_index([a.total_reward for a in agents])


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_mixed(per_strategy=2, num_ticks=500, seed=42,
              social_awareness=True, visibility='transparent'):
    agents, strategy_map = make_mixed_population(per_strategy, seed=seed)
    sim = Simulation(
        agents=agents,
        num_ticks=num_ticks,
        social_awareness=social_awareness,
        visibility=visibility,
        seed=seed,
    )
    sim.run()
    metrics = strategy_metrics(agents, strategy_map)
    fairness = overall_fairness(agents)
    return agents, strategy_map, metrics, fairness


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_mixed_results(metrics, fairness, seed=None, num_ticks=None):
    sep = '=' * 55
    label = f"  seed={seed}" if seed is not None else ""
    print(f"\n{sep}")
    print(f"  Mixed Population Results{label}")
    if num_ticks:
        print(f"  Ticks: {num_ticks}   Overall Fairness: {fairness:.4f}")
    print(sep)
    for strategy in STRATEGY_LABELS:
        if strategy not in metrics:
            continue
        m = metrics[strategy]
        print(f"  {strategy}")
        print(f"    Mean Reward  : {m['mean_reward']:.1f}")
        print(f"    Win Rate     : {m['win_rate']:.3f}")
        print(f"    Starvation   : {m['starvation_rate']:.3f}")
        print(f"    Intra-Fairness: {m['fairness']:.4f}")
        print()


def print_aggregate_mixed(all_metrics, all_fairness, seeds):
    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    sep = '=' * 55
    print(f"\n{sep}")
    print(f"  Mixed Population — Aggregate ({len(seeds)} seeds)")
    print(sep)
    print(f"  Overall Fairness: {mean(all_fairness):.4f} ± {std(all_fairness):.4f}\n")

    for strategy in STRATEGY_LABELS:
        rewards = [m[strategy]['mean_reward'] for m in all_metrics if strategy in m]
        winrates = [m[strategy]['win_rate'] for m in all_metrics if strategy in m]
        starvation = [m[strategy]['starvation_rate'] for m in all_metrics if strategy in m]
        if not rewards:
            continue
        print(f"  {strategy}")
        print(f"    Mean Reward  : {mean(rewards):.1f} ± {std(rewards):.1f}")
        print(f"    Win Rate     : {mean(winrates):.3f} ± {std(winrates):.3f}")
        print(f"    Starvation   : {mean(starvation):.3f} ± {std(starvation):.3f}")
        print()


def plot_mixed(all_metrics, all_fairness, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    strategies = STRATEGY_LABELS
    colors = ['#4a90d9', '#e67e22', '#2ecc71', '#e74c3c',
              '#e91e8c', '#9b59b6', '#1abc9c', '#f39c12']

    mean_rewards  = [mean([m[s]['mean_reward']  for m in all_metrics]) for s in strategies]
    std_rewards   = [std( [m[s]['mean_reward']  for m in all_metrics]) for s in strategies]
    mean_winrates = [mean([m[s]['win_rate']      for m in all_metrics]) for s in strategies]
    std_winrates  = [std( [m[s]['win_rate']      for m in all_metrics]) for s in strategies]
    mean_starv    = [mean([m[s]['starvation_rate'] for m in all_metrics]) for s in strategies]
    std_starv     = [std( [m[s]['starvation_rate'] for m in all_metrics]) for s in strategies]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Mixed Population: Strategies Competing Directly (mean ± std)', fontsize=13)

    axes[0].bar(strategies, mean_rewards, yerr=std_rewards, color=colors,
                capsize=5, error_kw={'linewidth': 1.5})
    axes[0].set_title('Mean Reward per Agent')
    axes[0].set_ylabel('Reward')
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(strategies, mean_winrates, yerr=std_winrates, color=colors,
                capsize=5, error_kw={'linewidth': 1.5})
    axes[1].set_title('Win Rate')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Fraction of Ticks Won')
    axes[1].tick_params(axis='x', rotation=15)

    axes[2].bar(strategies, mean_starv, yerr=std_starv, color=colors,
                capsize=5, error_kw={'linewidth': 1.5})
    axes[2].set_title('Starvation Rate (lower = better)')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('Fraction of Agents Starved')
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = os.path.join(save_dir, 'mixed_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- Reward distribution violin-style scatter per strategy ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Mixed Population: Individual Agent Rewards per Strategy', fontsize=13)

    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        all_agent_rewards = []
        for m in all_metrics:
            all_agent_rewards.extend(m[strategy]['agent_rewards'])
        jitter = [i + random.uniform(-0.15, 0.15) for _ in all_agent_rewards]
        ax.scatter(jitter, all_agent_rewards, alpha=0.4, color=color, s=20, label=strategy)
        ax.plot([i - 0.3, i + 0.3], [mean(all_agent_rewards)] * 2,
                color=color, linewidth=2.5)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies)
    ax.set_ylabel('Total Reward')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'mixed_reward_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Mixed population: emotion, priority aging, and backoff agents compete directly'
    )
    parser.add_argument('--per-strategy', type=int, default=2,
                        help='Agents per strategy (default: 2, total=6)')
    parser.add_argument('--ticks', type=int, default=500,
                        help='Simulation ticks (default: 500)')
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of seeds to average over (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run a single seed with full output')
    parser.add_argument('--explain', action='store_true',
                        help='Print per-agent explanations (single seed mode only)')
    args = parser.parse_args()

    per_strategy = args.per_strategy
    ticks = args.ticks

    # --- Single seed mode ---
    if args.seed is not None:
        agents, strategy_map, metrics, fairness = run_mixed(
            per_strategy=per_strategy,
            num_ticks=ticks,
            seed=args.seed,
        )
        print_mixed_results(metrics, fairness, seed=args.seed, num_ticks=ticks)

        if args.explain:
            print("--- Per-Agent Explanations (last tick) ---")
            for agent in agents:
                print(f"\n[{strategy_map[agent.agent_id]}]")
                print(agent.explain_last_decision())
        return

    # --- Multi-seed mode ---
    seeds = list(range(args.seeds))
    n_total = per_strategy * len(STRATEGY_LABELS)
    print(f"Mixed population: {per_strategy} × {len(STRATEGY_LABELS)} strategies "
          f"= {n_total} agents, {ticks} ticks, {len(seeds)} seeds\n")

    all_metrics = []
    all_fairness = []

    for seed in seeds:
        _, _, metrics, fairness = run_mixed(
            per_strategy=per_strategy,
            num_ticks=ticks,
            seed=seed,
        )
        all_metrics.append(metrics)
        all_fairness.append(fairness)
        print(f"  seed {seed} done  (fairness={fairness:.4f})")

    print_aggregate_mixed(all_metrics, all_fairness, seeds)
    plot_mixed(all_metrics, all_fairness)


if __name__ == '__main__':
    main()
