"""
Entry point for the emotion-based multi-agent resource competition simulation.

Usage:
    python run.py                          # defaults: 6 agents, 200 ticks, 10 seeds
    python run.py --agents 10 --ticks 500
    python run.py --seeds 20               # more seeds for tighter confidence
    python run.py --explain                # print per-agent decision explanations
    python run.py --seed 7                 # single seed mode (skips multi-seed)

Runs all 6 experimental conditions across multiple seeds and saves plots to results/.
"""

import argparse
import math
import os
import random

import matplotlib.pyplot as plt

from agents import Agent
from baselines import (ExponentialBackoffAgent, RoundRobinAgent, PriorityAgingAgent,
                       AIMDAgent, GreedyAgent, RandomAgent, UCB1Agent, WinRateAdaptiveAgent)
from simulation import Simulation
from metrics import compute_all, frustration_over_time


# ---------------------------------------------------------------------------
# Agent factories — same seed produces identical attribute distributions
# ---------------------------------------------------------------------------

def make_emotion_agents(n, seed=None):
    rng = random.Random(seed)
    agents = []
    for i in range(n):
        agents.append(Agent(
            agent_id=i,
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
            persistence=round(rng.uniform(0.2, 0.8), 2),
            risk_aversion=round(rng.uniform(0.1, 0.5), 2),
        ))
    return agents


def make_backoff_agents(n, seed=None):
    rng = random.Random(seed)
    agents = []
    for i in range(n):
        agents.append(ExponentialBackoffAgent(
            agent_id=i,
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
        ))
    return agents


def make_priority_aging_agents(n, seed=None):
    rng = random.Random(seed)
    agents = []
    for i in range(n):
        agents.append(PriorityAgingAgent(
            agent_id=i,
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
        ))
    return agents


def make_greedy_agents(n, seed=None):
    rng = random.Random(seed)
    return [GreedyAgent(agent_id=i,
                        skill_level=round(rng.uniform(0.4, 1.0), 2),
                        regen_rate=round(rng.uniform(5.0, 20.0), 1),
                        max_energy=100.0) for i in range(n)]


def make_random_agents(n, seed=None):
    rng = random.Random(seed)
    return [RandomAgent(agent_id=i,
                        skill_level=round(rng.uniform(0.4, 1.0), 2),
                        regen_rate=round(rng.uniform(5.0, 20.0), 1),
                        max_energy=100.0) for i in range(n)]


def make_ucb1_agents(n, seed=None):
    rng = random.Random(seed)
    return [UCB1Agent(agent_id=i,
                      skill_level=round(rng.uniform(0.4, 1.0), 2),
                      regen_rate=round(rng.uniform(5.0, 20.0), 1),
                      max_energy=100.0) for i in range(n)]


def make_winrate_agents(n, seed=None):
    rng = random.Random(seed)
    return [WinRateAdaptiveAgent(agent_id=i,
                                 skill_level=round(rng.uniform(0.4, 1.0), 2),
                                 regen_rate=round(rng.uniform(5.0, 20.0), 1),
                                 max_energy=100.0) for i in range(n)]


def make_aimd_agents(n, seed=None):
    rng = random.Random(seed)
    agents = []
    for i in range(n):
        agents.append(AIMDAgent(
            agent_id=i,
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
        ))
    return agents


def make_rr_agents(n, seed=None):
    rng = random.Random(seed)
    agents = []
    for i in range(n):
        agents.append(RoundRobinAgent(
            agent_id=i,
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
            agent_index=i,
            total_agents=n,
        ))
    return agents


# ---------------------------------------------------------------------------
# Run a single experiment
# ---------------------------------------------------------------------------

def run_experiment(name, agents, num_ticks=200, social_awareness=False,
                   visibility='blind', seed=42):
    sim = Simulation(
        agents=agents,
        num_ticks=num_ticks,
        social_awareness=social_awareness,
        visibility=visibility,
        seed=seed,
    )
    history = sim.run()
    results = compute_all(agents, history)
    results['name'] = name
    results['num_agents'] = len(agents)
    results['num_ticks'] = num_ticks
    return results, history, agents


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_results(results):
    sep = '=' * 55
    print(f"\n{sep}")
    print(f"  {results['name']}")
    print(sep)
    print(f"  Agents : {results['num_agents']}   Ticks : {results['num_ticks']}")
    print(f"  Jain's Fairness Index : {results['jains_fairness']}")
    print(f"  Starvation Rate       : {results['starvation_rate']}")
    tier = results['tier_distribution']
    print(f"  Tier Distribution     : T1={tier[1]:.2f}  T2={tier[2]:.2f}  T3={tier[3]:.2f}")
    print(f"  Win Rates             : {results['win_rates']}")
    print(f"  Total Rewards         : {results['total_rewards']}")


def print_aggregate(name, fairness_list, starvation_list):
    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    sep = '=' * 55
    print(f"\n{sep}")
    print(f"  {name}  [{len(fairness_list)} seeds]")
    print(sep)
    print(f"  Jain's Fairness : {mean(fairness_list):.4f} ± {std(fairness_list):.4f}")
    print(f"  Starvation Rate : {mean(starvation_list):.4f} ± {std(starvation_list):.4f}")


def plot_results_aggregate(condition_names, fairness_means, fairness_stds,
                           starvation_means, starvation_stds,
                           deviation_means=None, deviation_stds=None,
                           save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    short_names = [n.replace('Emotion | ', '').replace('Baseline | ', '')
                   for n in condition_names]
    x = range(len(condition_names))

    colors = ['steelblue' if n.startswith('Emotion') else 'tomato'
              for n in condition_names]

    ncols = 3 if deviation_means is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    fig.suptitle('Emotion-Based vs Baseline Resource Competition (mean ± std)', fontsize=13)

    axes[0].bar(x, fairness_means, yerr=fairness_stds, color=colors,
                capsize=5, error_kw={'linewidth': 1.5})
    axes[0].set_title("Jain's Fairness Index (higher = fairer)")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fairness")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(short_names, rotation=20, ha='right')

    axes[1].bar(x, starvation_means, yerr=starvation_stds, color=colors,
                capsize=5, error_kw={'linewidth': 1.5})
    axes[1].set_title("Starvation Rate (lower = better)")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of Agents Starved")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(short_names, rotation=20, ha='right')

    if deviation_means is not None:
        dev_stds = deviation_stds if deviation_stds is not None else [0] * len(deviation_means)
        axes[2].bar(x, deviation_means, yerr=dev_stds, color=colors,
                    capsize=5, error_kw={'linewidth': 1.5})
        axes[2].set_title("Ideal Share Deviation (lower = more efficient)")
        axes[2].set_ylabel("Mean Abs Deviation from Ideal (fraction)")
        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels(short_names, rotation=20, ha='right')

    from matplotlib.patches import Patch
    legend = [Patch(color='steelblue', label='Emotion'), Patch(color='tomato', label='Baseline')]
    fig.legend(handles=legend, loc='upper right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


def plot_results(all_results, history_map, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    names = [r['name'] for r in all_results]
    fairness = [r['jains_fairness'] for r in all_results]
    starvation = [r['starvation_rate'] for r in all_results]
    deviation = [r['ideal_share_deviation'] for r in all_results]

    # --- Bar chart: fairness, starvation, ideal share deviation ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle('Emotion-Based vs Baseline Resource Competition', fontsize=13)

    short_names = [n.replace('Emotion | ', '').replace('Baseline | ', '') for n in names]

    axes[0].bar(short_names, fairness, color='steelblue')
    axes[0].set_title("Jain's Fairness Index (higher = fairer)")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fairness")
    axes[0].tick_params(axis='x', rotation=20)

    axes[1].bar(short_names, starvation, color='tomato')
    axes[1].set_title("Starvation Rate (lower = better)")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of Agents Starved")
    axes[1].tick_params(axis='x', rotation=20)

    axes[2].bar(short_names, deviation, color='mediumpurple')
    axes[2].set_title("Ideal Share Deviation (lower = more efficient)")
    axes[2].set_ylabel("Mean Abs Deviation from Ideal (fraction)")
    axes[2].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    path = os.path.join(save_dir, 'single_seed_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    # --- Frustration over time for emotion conditions ---
    emotion_conditions = [
        r for r in all_results if r['name'].startswith('Emotion')
    ]
    if emotion_conditions:
        fig, ax = plt.subplots(figsize=(12, 5))
        for result in emotion_conditions:
            name = result['name']
            history = history_map[name]
            # average frustration across all agents per tick
            ticks = range(len(history))
            avg_frustration = [
                sum(
                    tick['agent_states'][aid]['frustration']
                    for aid in tick['agent_states']
                ) / len(tick['agent_states'])
                for tick in history
            ]
            ax.plot(ticks, avg_frustration,
                    label=name.replace('Emotion | ', ''))
        ax.set_title('Average Agent Frustration Over Time')
        ax.set_xlabel('Tick')
        ax.set_ylabel('Mean Frustration')
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(save_dir, 'frustration_over_time.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    # --- Tier distribution stacked bar ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(all_results))
    t1 = [r['tier_distribution'][1] for r in all_results]
    t2 = [r['tier_distribution'][2] for r in all_results]
    t3 = [r['tier_distribution'][3] for r in all_results]
    ax.bar(x, t1, label='Tier 1 (high)', color='gold')
    ax.bar(x, t2, bottom=t1, label='Tier 2 (mid)', color='steelblue')
    ax.bar(x, [t1[i]+t2[i] for i in x], bottom=[0]*len(x),
           label='_nolegend_', color='steelblue', alpha=0)
    ax.bar(x, t3, bottom=[t1[i]+t2[i] for i in x], label='Tier 3 (low)', color='gray')
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_names, rotation=20)
    ax.set_title('Average Tier Distribution per Condition')
    ax.set_ylabel('Fraction of Agent-Ticks')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'tier_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONDITION_SPECS = [
    ('Emotion | No Social | Blind',       'emotion',  False, 'blind'),
    ('Emotion | No Social | Transparent', 'emotion',  False, 'transparent'),
    ('Emotion | Social | Blind',          'emotion',  True,  'blind'),
    ('Emotion | Social | Transparent',    'emotion',  True,  'transparent'),
    ('Baseline | Priority Aging',         'aging',    False, 'blind'),
    ('Baseline | Exp Backoff',            'backoff',  False, 'blind'),
    ('Baseline | AIMD',                   'aimd',     False, 'blind'),
    ('Baseline | Greedy',                 'greedy',   False, 'blind'),
    ('Baseline | Random',                 'random',   False, 'blind'),
    ('Baseline | UCB1',                   'ucb1',     False, 'blind'),
    ('Baseline | Win-Rate Adaptive',      'winrate',  False, 'blind'),
]


def make_agents(kind, n, seed):
    if kind == 'emotion':
        return make_emotion_agents(n, seed)
    elif kind == 'backoff':
        return make_backoff_agents(n, seed)
    elif kind == 'aging':
        return make_priority_aging_agents(n, seed)
    elif kind == 'aimd':
        return make_aimd_agents(n, seed)
    elif kind == 'greedy':
        return make_greedy_agents(n, seed)
    elif kind == 'random':
        return make_random_agents(n, seed)
    elif kind == 'ucb1':
        return make_ucb1_agents(n, seed)
    elif kind == 'winrate':
        return make_winrate_agents(n, seed)
    else:
        return make_rr_agents(n, seed)


def main():
    parser = argparse.ArgumentParser(
        description='Emotion-based multi-agent resource competition'
    )
    parser.add_argument('--agents', type=int, default=6,
                        help='Number of agents (default: 6)')
    parser.add_argument('--ticks', type=int, default=200,
                        help='Number of simulation ticks (default: 200)')
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of random seeds to average over (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run a single seed and print full per-tick output')
    parser.add_argument('--explain', action='store_true',
                        help='Print per-agent decision explanations (single seed mode only)')
    args = parser.parse_args()

    n = args.agents
    ticks = args.ticks

    # --- Single seed mode ---
    if args.seed is not None:
        seed = args.seed
        all_results = []
        history_map = {}
        agent_map = {}

        for name, kind, social_awareness, visibility in CONDITION_SPECS:
            agents = make_agents(kind, n, seed)
            results, history, agents_out = run_experiment(
                name=name, agents=agents, num_ticks=ticks,
                social_awareness=social_awareness, visibility=visibility, seed=seed,
            )
            print_results(results)
            all_results.append(results)
            history_map[name] = history
            agent_map[name] = agents_out

        plot_results(all_results, history_map)

        if args.explain:
            print("\n--- Per-Agent Decision Explanations (last tick) ---")
            for name, agents in agent_map.items():
                print(f"\n{'─'*50}\n  {name}\n{'─'*50}")
                for agent in agents:
                    print(agent.explain_last_decision())
                    print()
        return

    # --- Multi-seed mode ---
    seeds = list(range(args.seeds))
    print(f"Running {len(CONDITION_SPECS)} conditions × {len(seeds)} seeds "
          f"({n} agents, {ticks} ticks each)...\n")

    # Accumulate per-condition results across seeds
    condition_fairness   = {name: [] for name, *_ in CONDITION_SPECS}
    condition_starvation = {name: [] for name, *_ in CONDITION_SPECS}
    condition_deviation  = {name: [] for name, *_ in CONDITION_SPECS}
    # Keep last seed's history for plots
    last_history_map = {}
    last_results_list = []

    for seed in seeds:
        seed_results = []
        for name, kind, social_awareness, visibility in CONDITION_SPECS:
            agents = make_agents(kind, n, seed)
            results, history, _ = run_experiment(
                name=name, agents=agents, num_ticks=ticks,
                social_awareness=social_awareness, visibility=visibility, seed=seed,
            )
            condition_fairness[name].append(results['jains_fairness'])
            condition_starvation[name].append(results['starvation_rate'])
            condition_deviation[name].append(results['ideal_share_deviation'])
            seed_results.append(results)
            if seed == seeds[-1]:
                last_history_map[name] = history
                last_results_list.append(results)

        print(f"  seed {seed} done")

    # Print aggregate results
    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    print()
    condition_names = [name for name, *_ in CONDITION_SPECS]
    fairness_means  = [mean(condition_fairness[n])   for n in condition_names]
    fairness_stds   = [std(condition_fairness[n])    for n in condition_names]
    starv_means     = [mean(condition_starvation[n]) for n in condition_names]
    starv_stds      = [std(condition_starvation[n])  for n in condition_names]
    dev_means       = [mean(condition_deviation[n])  for n in condition_names]
    dev_stds        = [std(condition_deviation[n])   for n in condition_names]

    for name in condition_names:
        print_aggregate(name, condition_fairness[name], condition_starvation[name])

    # Plots: error bars from multi-seed + frustration/tier from last seed
    plot_results_aggregate(condition_names, fairness_means, fairness_stds,
                           starv_means, starv_stds, dev_means, dev_stds)
    plot_results(last_results_list, last_history_map)


if __name__ == '__main__':
    main()
