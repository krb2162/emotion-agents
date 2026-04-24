import random

TIER_REWARDS = {1: 10, 2: 5, 3: 2}


class Simulation:
    """
    Runs the multi-agent resource competition.

    Each tick:
      1. All agents decide their target tier and speed simultaneously
         (imperfect information — no agent knows others' choices)
      2. Bids are computed and highest bid wins each tier (one winner per tier)
      3. Winners receive reward; losers receive nothing
      4. All agents pay their energy cost regardless of outcome
      5. Emotional state and learned weights are updated
      6. Tick is logged to history

    Parameters
    ----------
    agents          : list of Agent (or baseline) objects
    num_ticks       : simulation length (default 200)
    social_awareness: whether social_cost mechanics are active
    visibility      : 'blind' — agents know only their own state
                      'transparent' — agents see all others' emotional state
    seed            : random seed for reproducibility
    """

    def __init__(self, agents, num_ticks=200, social_awareness=False,
                 visibility='blind', seed=None):
        self.agents = agents
        self.num_ticks = num_ticks
        self.social_awareness = social_awareness
        self.visibility = visibility
        self.history = []

        if seed is not None:
            random.seed(seed)

    def run(self):
        for tick in range(self.num_ticks):
            self._step(tick)
        return self.history

    def _step(self, tick):
        # --- Phase 1: decisions (simultaneous, imperfect information) ---
        decisions = {}
        for agent in self.agents:
            other_agents = self.agents if self.visibility == 'transparent' else None
            tier, speed = agent.decide(
                tick=tick,
                other_agents=other_agents,
                social_awareness=self.social_awareness,
                visibility=self.visibility,
            )
            decisions[agent.agent_id] = (tier, speed)

        # --- Phase 2: collect bids per tier ---
        tier_bidders = {1: [], 2: [], 3: []}
        for agent in self.agents:
            tier, _ = decisions[agent.agent_id]
            bid = agent.compute_bid()
            tier_bidders[tier].append((bid, agent))

        # --- Phase 3: resolve winners (highest bid, random tiebreak) ---
        winners = {}   # tier -> winning agent_id
        for tier, bidders in tier_bidders.items():
            if not bidders:
                continue
            random.shuffle(bidders)                          # shuffle first for tiebreak
            bidders.sort(key=lambda x: x[0], reverse=True)  # stable: ties already randomised
            winners[tier] = bidders[0][1].agent_id

        # --- Phase 4 & 5: apply outcomes and log ---
        tier_counts = {1: 0, 2: 0, 3: 0}
        agent_states = {}

        for agent in self.agents:
            tier, _ = decisions[agent.agent_id]
            won = winners.get(tier) == agent.agent_id
            reward = TIER_REWARDS[tier] if won else 0
            tier_counts[tier] += 1

            agent.apply_outcome(
                won=won,
                reward=reward,
                social_awareness=self.social_awareness,
            )

            agent_states[agent.agent_id] = {
                'tier': tier,
                'won': won,
                'reward': reward,
                'frustration': round(agent.frustration, 3),
                'social_cost': round(agent.social_cost, 3),
                'energy': round(agent.energy, 3),
                'total_reward': agent.total_reward,
            }

        self.history.append({
            'tick': tick,
            'winners': dict(winners),
            'tier_counts': tier_counts,
            'agent_states': agent_states,
        })
