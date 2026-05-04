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


def tier_access_equity(agents, tier=1):
    """
    Jain's fairness index applied to tier-1 (or any tier) wins across agents.
    Measures whether all agents get equal access to a specific tier, not just
    equal total rewards. An agent that always wins tier 3 and never tier 1
    scores well on Jain's overall but poorly here.

    Returns value in [1/n, 1.0] — 1.0 means perfectly equal tier access.
    """
    tier_wins = []
    for agent in agents:
        wins_at_tier = sum(
            1 for entry in agent.state_log
            if entry['tier'] == tier and entry['won']
        )
        tier_wins.append(wins_at_tier)
    return jains_fairness_index(tier_wins)


def per_agent_tier_win_share(agents):
    """
    For each agent, the fraction of their wins that came from each tier.
    Returns dict: agent_id -> {1: float, 2: float, 3: float}
    """
    result = {}
    for agent in agents:
        wins_by_tier = {1: 0, 2: 0, 3: 0}
        for entry in agent.state_log:
            if entry['won']:
                wins_by_tier[entry['tier']] += 1
        total_wins = sum(wins_by_tier.values())
        if total_wins == 0:
            result[agent.agent_id] = {1: 0.0, 2: 0.0, 3: 0.0}
        else:
            result[agent.agent_id] = {t: round(wins_by_tier[t] / total_wins, 3) for t in [1, 2, 3]}
    return result


def ideal_share_deviation(rewards, num_ticks, tier_rewards=None):
    """
    Mean absolute deviation from ideal share as a fraction of ideal.

    Ideal share = sum of top-n tier rewards per tick × ticks / n agents,
    where n = number of agents. This caps the claimable reward at what
    the population could actually win (e.g. 2 agents can only fill 2 tiers).

    Returns a deviation score: 0.0 = every agent got exactly their ideal share,
    higher = greater divergence from ideal. Lower is better.
    """
    if tier_rewards is None:
        tier_rewards = {1: 10, 2: 5, 3: 2}

    n = len(rewards)
    if n == 0 or num_ticks == 0:
        return 0.0

    sorted_rewards = sorted(tier_rewards.values(), reverse=True)
    n_claimable = min(n, len(sorted_rewards))
    max_per_tick = sum(sorted_rewards[:n_claimable])
    ideal = (max_per_tick * num_ticks) / n

    if ideal == 0:
        return 0.0

    deviations = [abs(r - ideal) for r in rewards]
    return round(sum(deviations) / len(deviations) / ideal, 4)


def compute_all(agents, history):
    """Compute and return all metrics as a dict."""
    rewards = [a.total_reward for a in agents]
    return {
        'jains_fairness': round(jains_fairness_index(rewards), 4),
        'starvation_rate': round(starvation_rate(agents), 4),
        'ideal_share_deviation': ideal_share_deviation(rewards, len(history)),
        'tier_distribution': tier_distribution(history, len(agents)),
        'win_rates': win_rates(agents),
        'total_rewards': {a.agent_id: a.total_reward for a in agents},
    }
