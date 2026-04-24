"""
Runs a mixed-population simulation and exports full tick-by-tick data to JSON
for the interactive website visualization.

Usage:
    python export_json.py                  # 2 per strategy, 200 ticks, seed 42
    python export_json.py --ticks 300 --seed 7
    python export_json.py --per-strategy 3
"""

import argparse
import json
import os
import random

from agents import Agent
from baselines import ExponentialBackoffAgent, PriorityAgingAgent
from simulation import Simulation


STRATEGY_LABELS = ['Emotion', 'Priority Aging', 'Exp Backoff']
STRATEGY_COLORS = {
    'Emotion':        '#4a90d9',
    'Priority Aging': '#e8a838',
    'Exp Backoff':    '#e05c5c',
}


def make_mixed_population(per_strategy, seed=None):
    rng = random.Random(seed)
    agents = []
    strategy_map = {}
    agent_id = 0

    for _ in range(per_strategy):
        a = Agent(
            agent_id=agent_id,
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
            persistence=round(rng.uniform(0.2, 0.8), 2),
            risk_aversion=round(rng.uniform(0.1, 0.5), 2),
        )
        strategy_map[agent_id] = 'Emotion'
        agents.append(a)
        agent_id += 1

    for _ in range(per_strategy):
        a = PriorityAgingAgent(
            agent_id=agent_id,
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
        )
        strategy_map[agent_id] = 'Priority Aging'
        agents.append(a)
        agent_id += 1

    for _ in range(per_strategy):
        a = ExponentialBackoffAgent(
            agent_id=agent_id,
            skill_level=round(rng.uniform(0.4, 1.0), 2),
            regen_rate=round(rng.uniform(5.0, 20.0), 1),
            max_energy=100.0,
        )
        strategy_map[agent_id] = 'Exp Backoff'
        agents.append(a)
        agent_id += 1

    return agents, strategy_map


def agent_metadata(agent, strategy):
    meta = {
        'id': agent.agent_id,
        'strategy': strategy,
        'color': STRATEGY_COLORS[strategy],
        'skill_level': agent.skill_level,
        'regen_rate': agent.regen_rate,
        'max_energy': agent.max_energy,
    }
    if strategy == 'Emotion':
        meta['persistence'] = agent.persistence
        meta['risk_aversion'] = agent.risk_aversion
    return meta


def build_export(agents, strategy_map, history):
    """Build the full JSON structure from simulation results."""

    # Agent metadata
    agents_meta = [agent_metadata(a, strategy_map[a.agent_id]) for a in agents]

    # Tick-by-tick data
    ticks = []
    for tick_data in history:
        t = tick_data['tick']
        winners = {str(tier): aid for tier, aid in tick_data['winners'].items()}

        agent_states = {}
        for agent in agents:
            if t >= len(agent.state_log):
                continue
            log = agent.state_log[t]
            strategy = strategy_map[agent.agent_id]

            state = {
                'tier': log['tier'],
                'speed': log.get('speed', agent._speed if hasattr(agent, '_speed') else 0),
                'bid': log.get('bid', 0),
                'won': log['won'],
                'reward': log['reward'],
                'energy': log['energy'],
                'total_reward': sum(e['reward'] for e in agent.state_log[:t+1]),
            }

            # Emotion-specific fields
            if strategy == 'Emotion':
                state.update({
                    'action': log.get('action', 'normal'),
                    'frustration': log['frustration'],
                    'frustration_after': log.get('frustration_after', log['frustration']),
                    'reward_reinforcement': log['reward_reinforcement'],
                    'social_cost': log['social_cost'],
                    'p_try_harder': log['p_try_harder'],
                    'p_give_up': log['p_give_up'],
                    'p_stay': log['p_stay'],
                })
            elif strategy == 'Priority Aging':
                state.update({
                    'accumulated_priority': log.get('accumulated_priority', agent.skill_level),
                    'ticks_since_win': log.get('ticks_since_win', 0),
                })
            elif strategy == 'Exp Backoff':
                state.update({
                    'consecutive_losses': log.get('consecutive_losses', 0),
                    'wait_ticks_remaining': log.get('wait_ticks_remaining', 0),
                })

            agent_states[str(agent.agent_id)] = state

        ticks.append({
            'tick': t,
            'winners': winners,
            'tier_counts': tick_data['tier_counts'],
            'agents': agent_states,
        })

    return {
        'metadata': {
            'num_agents': len(agents),
            'num_ticks': len(history),
            'strategies': STRATEGY_LABELS,
            'strategy_colors': STRATEGY_COLORS,
            'tier_rewards': {1: 10, 2: 5, 3: 2},
        },
        'agents': agents_meta,
        'ticks': ticks,
    }


def main():
    parser = argparse.ArgumentParser(description='Export simulation data to JSON for website')
    parser.add_argument('--per-strategy', type=int, default=2)
    parser.add_argument('--ticks', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/simulation_data.json')
    args = parser.parse_args()

    print(f"Running simulation: {args.per_strategy * 3} agents, "
          f"{args.ticks} ticks, seed={args.seed}")

    agents, strategy_map = make_mixed_population(args.per_strategy, seed=args.seed)
    sim = Simulation(
        agents=agents,
        num_ticks=args.ticks,
        social_awareness=True,
        visibility='transparent',
        seed=args.seed,
    )
    history = sim.run()

    data = build_export(agents, strategy_map, history)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(data, f)

    print(f"Exported {len(history)} ticks → {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
