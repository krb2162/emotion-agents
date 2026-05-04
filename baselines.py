import random

ENERGY_COST_RATE = 0.15


class AIMDAgent:
    """
    Baseline: Additive Increase, Multiplicative Decrease (AIMD).

    The classic distributed fairness mechanism from TCP congestion control
    (Jacobson 1988). On a win, aggressiveness increases by a fixed additive
    step. On a loss, it is cut by a multiplicative factor. Tier is derived
    from current speed: fast → tier 1, mid → tier 2, slow → tier 3.

    This is the structural inverse of the emotion agent: retreat is triggered
    by loss (not by winning), and the agent never voluntarily yields.

    Explainability: current speed, last event (win/loss), tier derived from speed.
    """

    ALPHA = 0.1   # additive increase per win
    BETA  = 0.5   # multiplicative decrease factor per loss

    # Speed thresholds for tier selection
    T1_SPEED = 0.55
    T2_SPEED = 0.30

    def __init__(self, agent_id, skill_level, regen_rate, max_energy):
        self.agent_id    = agent_id
        self.skill_level = skill_level
        self.regen_rate  = regen_rate
        self.max_energy  = max_energy

        self.energy       = max_energy
        self.total_reward = 0.0
        self.wins         = 0
        self.losses       = 0
        self.win_history  = []
        self.state_log    = []

        self._speed = 0.5   # starts mid-range
        self._tier  = 2

        # stub emotional state (interface compatibility)
        self.frustration = 0.0
        self.social_cost = 0.0

    def _speed_to_tier(self, speed):
        if speed >= self.T1_SPEED:
            return 1
        elif speed >= self.T2_SPEED:
            return 2
        else:
            return 3

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        self._tier = self._speed_to_tier(self._speed)
        return self._tier, self._speed

    def compute_bid(self):
        return self.skill_level * self._speed

    def apply_outcome(self, won, reward, social_awareness=False):
        energy_cost = self._speed * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        if won:
            self.wins += 1
            self.total_reward += reward
            self.win_history.append(True)
            self._speed = min(1.0, self._speed + self.ALPHA)
        else:
            self.losses += 1
            self.win_history.append(False)
            self._speed = max(0.1, self._speed * self.BETA)

        self._tier = self._speed_to_tier(self._speed)

        self.state_log.append({
            'tier':        self._tier,
            'won':         won,
            'reward':      reward if won else 0,
            'speed':       round(self._speed, 3),
            'energy':      round(self.energy, 3),
            'total_reward': self.total_reward,
        })

    def explain_last_decision(self):
        if not self.state_log:
            return f"AIMDAgent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        lines = [
            f"AIMDAgent {self.agent_id} | tick decision:",
            f"  target tier : {s['tier']}",
            f"  speed       : {s['speed']}  (ALPHA={self.ALPHA}, BETA={self.BETA})",
            f"  energy      : {s['energy']} / {self.max_energy}",
            f"  outcome     : {'WON' if s['won'] else 'LOST'}  reward={s['reward']}",
        ]
        return "\n".join(lines)


class GreedyAgent:
    """
    Baseline: always target tier 1 at maximum speed.

    No state, no adaptation. Pure defection — maximizes personal reward
    at the cost of fairness. Establishes the selfish upper bound on reward
    and lower bound on fairness.
    """

    def __init__(self, agent_id, skill_level, regen_rate, max_energy):
        self.agent_id    = agent_id
        self.skill_level = skill_level
        self.regen_rate  = regen_rate
        self.max_energy  = max_energy

        self.energy       = max_energy
        self.total_reward = 0.0
        self.wins         = 0
        self.losses       = 0
        self.win_history  = []
        self.state_log    = []

        self._tier  = 1
        self._speed = 1.0
        self.frustration = 0.0
        self.social_cost = 0.0

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        return self._tier, self._speed

    def compute_bid(self):
        return self.skill_level * self._speed

    def apply_outcome(self, won, reward, social_awareness=False):
        energy_cost = self._speed * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        if won:
            self.wins += 1
            self.total_reward += reward
            self.win_history.append(True)
        else:
            self.losses += 1
            self.win_history.append(False)

        self.state_log.append({
            'tier': self._tier, 'won': won,
            'reward': reward if won else 0,
            'energy': round(self.energy, 3),
            'total_reward': self.total_reward,
        })

    def explain_last_decision(self):
        if not self.state_log:
            return f"GreedyAgent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        return (f"GreedyAgent {self.agent_id} | always tier 1, speed 1.0 | "
                f"{'WON' if s['won'] else 'LOST'} reward={s['reward']}")


class RandomAgent:
    """
    Baseline: pick a tier uniformly at random each tick, mid speed.

    Zero info, zero strategy. Establishes a floor — any adaptive strategy
    should beat this on fairness and starvation.
    """

    def __init__(self, agent_id, skill_level, regen_rate, max_energy):
        self.agent_id    = agent_id
        self.skill_level = skill_level
        self.regen_rate  = regen_rate
        self.max_energy  = max_energy

        self.energy       = max_energy
        self.total_reward = 0.0
        self.wins         = 0
        self.losses       = 0
        self.win_history  = []
        self.state_log    = []

        self._tier  = 1
        self._speed = 0.5
        self.frustration = 0.0
        self.social_cost = 0.0

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        self._tier = random.randint(1, 3)
        return self._tier, self._speed

    def compute_bid(self):
        return self.skill_level * self._speed

    def apply_outcome(self, won, reward, social_awareness=False):
        energy_cost = self._speed * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        if won:
            self.wins += 1
            self.total_reward += reward
            self.win_history.append(True)
        else:
            self.losses += 1
            self.win_history.append(False)

        self.state_log.append({
            'tier': self._tier, 'won': won,
            'reward': reward if won else 0,
            'energy': round(self.energy, 3),
            'total_reward': self.total_reward,
        })

    def explain_last_decision(self):
        if not self.state_log:
            return f"RandomAgent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        return (f"RandomAgent {self.agent_id} | random tier={s['tier']} | "
                f"{'WON' if s['won'] else 'LOST'} reward={s['reward']}")


class UCB1Agent:
    """
    Baseline: Upper Confidence Bound (UCB1) multi-armed bandit on tiers.

    Treats each tier as an arm. Picks the tier with the highest
    UCB1 score: mean_reward + sqrt(2 * ln(total_pulls) / pulls_at_tier).
    The exploration bonus favors under-tried tiers early, then exploits
    the historically best tier.

    Optimizes personal reward, not fairness. Expected to cluster at tier 1
    once it learns tier rewards, producing poor Jain's index.
    """

    import math as _math

    def __init__(self, agent_id, skill_level, regen_rate, max_energy):
        self.agent_id    = agent_id
        self.skill_level = skill_level
        self.regen_rate  = regen_rate
        self.max_energy  = max_energy

        self.energy       = max_energy
        self.total_reward = 0.0
        self.wins         = 0
        self.losses       = 0
        self.win_history  = []
        self.state_log    = []

        # Bandit state: pulls and reward sums per tier
        self._pulls  = {1: 0, 2: 0, 3: 0}
        self._totals = {1: 0.0, 2: 0.0, 3: 0.0}
        self._total_pulls = 0
        self._tier  = 1
        self._speed = 0.6

        self.frustration = 0.0
        self.social_cost = 0.0

    def _ucb1_score(self, tier):
        import math
        if self._pulls[tier] == 0:
            return float('inf')   # force exploration of untried tiers first
        mean = self._totals[tier] / self._pulls[tier]
        bonus = math.sqrt(2 * math.log(self._total_pulls) / self._pulls[tier])
        return mean + bonus

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        self._tier = max([1, 2, 3], key=self._ucb1_score)
        return self._tier, self._speed

    def compute_bid(self):
        return self.skill_level * self._speed

    def apply_outcome(self, won, reward, social_awareness=False):
        energy_cost = self._speed * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        self._pulls[self._tier] += 1
        self._total_pulls += 1
        self._totals[self._tier] += reward   # 0 if lost, tier reward if won

        if won:
            self.wins += 1
            self.total_reward += reward
            self.win_history.append(True)
        else:
            self.losses += 1
            self.win_history.append(False)

        self.state_log.append({
            'tier': self._tier, 'won': won,
            'reward': reward if won else 0,
            'pulls': dict(self._pulls),
            'energy': round(self.energy, 3),
            'total_reward': self.total_reward,
        })

    def explain_last_decision(self):
        if not self.state_log:
            return f"UCB1Agent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        scores = {t: round(self._ucb1_score(t), 3) for t in [1, 2, 3]}
        return (f"UCB1Agent {self.agent_id} | tier={s['tier']} UCB scores={scores} | "
                f"{'WON' if s['won'] else 'LOST'} reward={s['reward']}")


class WinRateAdaptiveAgent:
    """
    Baseline: adapt tier and speed based on own rolling win rate.

    Tracks win rate over a recent window. High win rate → compete harder
    (tier 1, higher speed). Low win rate → back off (tier 2 or 3, lower speed).
    Same local information as the emotion agent but without any emotional
    state — tests whether frustration/reinforcement complexity adds value.

    Thresholds:
        win_rate > 0.5  → tier 1, speed 0.8  (doing well, push harder)
        win_rate > 0.2  → tier 2, speed 0.5  (neutral, hold position)
        win_rate ≤ 0.2  → tier 3, speed 0.3  (struggling, conserve)
    """

    WINDOW = 20

    def __init__(self, agent_id, skill_level, regen_rate, max_energy):
        self.agent_id    = agent_id
        self.skill_level = skill_level
        self.regen_rate  = regen_rate
        self.max_energy  = max_energy

        self.energy       = max_energy
        self.total_reward = 0.0
        self.wins         = 0
        self.losses       = 0
        self.win_history  = []
        self.state_log    = []

        self._tier  = 1
        self._speed = 0.6
        self.frustration = 0.0
        self.social_cost = 0.0

    def _win_rate(self):
        if not self.win_history:
            return 0.5   # optimistic start
        recent = self.win_history[-self.WINDOW:]
        return sum(recent) / len(recent)

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        wr = self._win_rate()
        if wr > 0.5:
            self._tier, self._speed = 1, 0.8
        elif wr > 0.2:
            self._tier, self._speed = 2, 0.5
        else:
            self._tier, self._speed = 3, 0.3
        return self._tier, self._speed

    def compute_bid(self):
        return self.skill_level * self._speed

    def apply_outcome(self, won, reward, social_awareness=False):
        energy_cost = self._speed * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        if won:
            self.wins += 1
            self.total_reward += reward
            self.win_history.append(True)
        else:
            self.losses += 1
            self.win_history.append(False)

        self.state_log.append({
            'tier': self._tier, 'won': won,
            'reward': reward if won else 0,
            'win_rate': round(self._win_rate(), 3),
            'energy': round(self.energy, 3),
            'total_reward': self.total_reward,
        })

    def explain_last_decision(self):
        if not self.state_log:
            return f"WinRateAdaptiveAgent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        return (f"WinRateAdaptiveAgent {self.agent_id} | tier={s['tier']} "
                f"win_rate={s['win_rate']} | "
                f"{'WON' if s['won'] else 'LOST'} reward={s['reward']}")


class ExponentialBackoffAgent:
    """
    Baseline: tier-aware exponential backoff.

    Agents step down tiers progressively on consecutive losses rather than
    jumping straight to tier 3:
      - 0 consecutive losses  → tier 1 (full effort)
      - 1 consecutive loss    → tier 2 (back off one level, still competing)
      - 2+ consecutive losses → tier 3 with exponential wait (2^n ticks, cap 16)

    Any win resets to tier 1 with fresh consecutive_losses = 0.

    Decision is purely mechanical — no emotional state.
    Explainability: consecutive loss count, current tier, wait ticks remaining.
    """

    def __init__(self, agent_id, skill_level, regen_rate, max_energy):
        self.agent_id = agent_id
        self.skill_level = skill_level
        self.regen_rate = regen_rate
        self.max_energy = max_energy

        self.energy = max_energy
        self.total_reward = 0.0
        self.wins = 0
        self.losses = 0
        self.win_history = []
        self.state_log = []

        self.consecutive_losses = 0
        self.wait_ticks_remaining = 0
        self._speed = 0.7
        self._tier = 1

        self.frustration = 0.0
        self.social_cost = 0.0

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        if self.wait_ticks_remaining > 0:
            # Waiting out backoff at tier 3
            self.wait_ticks_remaining -= 1
            self._tier = 3
            self._speed = 0.2
        elif self.consecutive_losses == 0:
            self._tier = 1
            self._speed = 0.7
        elif self.consecutive_losses == 1:
            # One loss — drop to tier 2, still competing
            self._tier = 2
            self._speed = 0.5
        else:
            # Two+ losses — tier 3 with exponential wait
            self._tier = 3
            self._speed = 0.3
        return self._tier, self._speed

    def compute_bid(self):
        return self.skill_level * self._speed

    def apply_outcome(self, won, reward, social_awareness=False):
        energy_cost = self._speed * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        if won:
            self.wins += 1
            self.total_reward += reward
            self.consecutive_losses = 0
            self.wait_ticks_remaining = 0
            self.win_history.append(True)
        else:
            self.losses += 1
            self.win_history.append(False)
            # Escalate only when not already waiting
            if self.wait_ticks_remaining == 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= 2:
                    self.wait_ticks_remaining = min(2 ** (self.consecutive_losses - 1), 16)

        self.state_log.append({
            'tier': self._tier,
            'won': won,
            'reward': reward if won else 0,
            'consecutive_losses': self.consecutive_losses,
            'wait_ticks_remaining': self.wait_ticks_remaining,
            'energy': round(self.energy, 3),
            'total_reward': self.total_reward,
        })

    def explain_last_decision(self):
        if not self.state_log:
            return f"BackoffAgent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        lines = [
            f"BackoffAgent {self.agent_id} | tick decision:",
            f"  target tier        : {s['tier']}",
            f"  consecutive losses : {s['consecutive_losses']}",
            f"  wait ticks left    : {s['wait_ticks_remaining']}",
            f"  energy             : {s['energy']} / {self.max_energy}",
            f"  outcome            : {'WON' if s['won'] else 'LOST'}  reward={s['reward']}",
        ]
        return "\n".join(lines)


class PriorityAgingAgent:
    """
    OS-style priority aging baseline.

    Each agent has a base priority (skill_level). Every tick they lose,
    their accumulated priority grows by aging_rate. On a win it resets
    to base. Highest accumulated priority wins the tier.

    This is how real OS schedulers prevent starvation — weak processes
    gradually accumulate enough priority to eventually win.

    Tier targeting: always competes at tier 1. If accumulated priority
    is very low (energy depleted), falls back to tier 3.

    Explainability: base priority, accumulated priority, ticks since last win.
    """

    AGING_RATE = 0.08   # priority gained per lost tick
    # MLFQ demotion thresholds: ticks without a win before dropping tier
    T1_LIMIT = 8    # after 8 ticks without win → drop to tier 2
    T2_LIMIT = 20   # after 20 ticks without win → drop to tier 3

    def __init__(self, agent_id, skill_level, regen_rate, max_energy):
        self.agent_id = agent_id
        self.skill_level = skill_level
        self.regen_rate = regen_rate
        self.max_energy = max_energy

        self.energy = max_energy
        self.total_reward = 0.0
        self.wins = 0
        self.losses = 0
        self.win_history = []
        self.state_log = []

        self.accumulated_priority = skill_level
        self.ticks_since_win = 0
        self._tier = 1
        self._speed = 0.7

        self.frustration = 0.0
        self.social_cost = 0.0

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        # MLFQ-style: demote tier based on time since last win.
        # Win at any tier promotes back to tier 1.
        if self.ticks_since_win >= self.T2_LIMIT:
            self._tier = 3
            self._speed = 0.3
        elif self.ticks_since_win >= self.T1_LIMIT:
            self._tier = 2
            self._speed = 0.5
        else:
            self._tier = 1
            self._speed = 0.7
        return self._tier, self._speed

    def compute_bid(self):
        return self.accumulated_priority * self._speed

    def apply_outcome(self, won, reward, social_awareness=False):
        energy_cost = self._speed * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        if won:
            self.wins += 1
            self.total_reward += reward
            self.win_history.append(True)
            self.accumulated_priority = self.skill_level   # reset to base
            self.ticks_since_win = 0
        else:
            self.losses += 1
            self.win_history.append(False)
            self.accumulated_priority += self.AGING_RATE   # age up
            self.ticks_since_win += 1

        self.state_log.append({
            'tier': self._tier,
            'won': won,
            'reward': reward if won else 0,
            'base_priority': round(self.skill_level, 3),
            'accumulated_priority': round(self.accumulated_priority, 3),
            'ticks_since_win': self.ticks_since_win,
            'energy': round(self.energy, 3),
            'total_reward': self.total_reward,
        })

    def explain_last_decision(self):
        if not self.state_log:
            return f"PriorityAgingAgent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        lines = [
            f"PriorityAgingAgent {self.agent_id} | tick decision:",
            f"  target tier         : {s['tier']}",
            f"  base priority       : {s['base_priority']}  (skill level)",
            f"  accumulated priority: {s['accumulated_priority']}  (+{self.AGING_RATE}/lost tick)",
            f"  ticks since win     : {s['ticks_since_win']}",
            f"  energy              : {s['energy']} / {self.max_energy}",
            f"  outcome             : {'WON' if s['won'] else 'LOST'}  reward={s['reward']}",
        ]
        return "\n".join(lines)


class RoundRobinAgent:
    """
    Baseline: agents take turns at tier 1 in strict fixed rotation
    determined by their index. On other ticks they target tier 3 lightly.

    Decision is purely mechanical — no emotional state.
    Explainability: whose turn it is.
    """

    def __init__(self, agent_id, skill_level, regen_rate, max_energy,
                 agent_index, total_agents):
        self.agent_id = agent_id
        self.skill_level = skill_level
        self.regen_rate = regen_rate
        self.max_energy = max_energy
        self.agent_index = agent_index
        self.total_agents = total_agents

        self.energy = max_energy
        self.total_reward = 0.0
        self.wins = 0
        self.losses = 0
        self.win_history = []
        self.state_log = []

        self._tier = 1
        self._speed = 0.7

        # stub emotional state
        self.frustration = 0.0
        self.social_cost = 0.0

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        position = (tick - self.agent_index) % self.total_agents
        if position == 0:
            self._tier, self._speed = 1, 0.8
        elif position == 1:
            self._tier, self._speed = 2, 0.5
        elif position == 2:
            self._tier, self._speed = 3, 0.3
        else:
            self._tier, self._speed = 3, 0.1
        return self._tier, self._speed

    def compute_bid(self):
        return self.skill_level * self._speed

    def apply_outcome(self, won, reward, social_awareness=False):
        energy_cost = self._speed * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        if won:
            self.wins += 1
            self.total_reward += reward
            self.win_history.append(True)
        else:
            self.losses += 1
            self.win_history.append(False)

        self.state_log.append({
            'tier': self._tier,
            'won': won,
            'reward': reward if won else 0,
            'my_turn': self._tier == 1,
            'energy': round(self.energy, 3),
            'total_reward': self.total_reward,
        })

    def explain_last_decision(self):
        if not self.state_log:
            return f"RoundRobinAgent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        lines = [
            f"RoundRobinAgent {self.agent_id} | tick decision:",
            f"  target tier : {s['tier']}",
            f"  my turn     : {s['my_turn']}  (index {self.agent_index} of {self.total_agents})",
            f"  energy      : {s['energy']} / {self.max_energy}",
            f"  outcome     : {'WON' if s['won'] else 'LOST'}  reward={s['reward']}",
        ]
        return "\n".join(lines)
