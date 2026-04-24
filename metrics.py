def jains_fairness_index(rewards):
    """
    Jain's Fairness Index over a list of total rewards.
    Returns 1.0 for perfect equality, 1/n for maximum inequality.
    """
    n = len(rewards)
    if n == 0 or sum(rewards) == 0:
        return 0.0
    numerator = sum(rewards) ** 2
    denominator = n * sum(r ** 2 for r in rewards)
    return numerator / denominator


def starvation_rate(agents, window=20):
    """
    Fraction of agents that went `window` or more consecutive ticks
    without winning any reward (evaluated over the final `window` ticks).
    """
    if not agents:
        return 0.0
    starved = 0
    for agent in agents:
        log = agent.state_log
        if len(log) < window:
            continue
        recent = log[-window:]
        if not any(entry['won'] for entry in recent):
            starved += 1
    return starved / len(agents)


def tier_distribution(history, num_agents):
    """
    Average fraction of agents targeting each tier per tick.
    Returns dict {1: float, 2: float, 3: float}.
    """
    if not history or num_agents == 0:
        return {1: 0.0, 2: 0.0, 3: 0.0}
    sums = {1: 0, 2: 0, 3: 0}
    for tick in history:
        for tier in [1, 2, 3]:
            sums[tier] += tick['tier_counts'].get(tier, 0)
    n = len(history) * num_agents
    return {tier: sums[tier] / n for tier in [1, 2, 3]}


def win_rates(agents):
    """Win rate per agent over the full simulation."""
    result = {}
    for agent in agents:
        total = agent.wins + agent.losses
        result[agent.agent_id] = round(agent.wins / total, 4) if total > 0 else 0.0
    return result


def frustration_over_time(history, agent_id):
    """Frustration trajectory for a single agent across all ticks."""
    return [
        tick['agent_states'][agent_id]['frustration']
        for tick in history
        if agent_id in tick['agent_states']
    ]


def compute_all(agents, history):
    """Compute and return all metrics as a dict."""
    rewards = [a.total_reward for a in agents]
    return {
        'jains_fairness': round(jains_fairness_index(rewards), 4),
        'starvation_rate': round(starvation_rate(agents), 4),
        'tier_distribution': tier_distribution(history, len(agents)),
        'win_rates': win_rates(agents),
        'total_rewards': {a.agent_id: a.total_reward for a in agents},
    }
