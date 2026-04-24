import random

ENERGY_COST_RATE = 0.15


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
        if tick % self.total_agents == self.agent_index:
            self._tier = 1
            self._speed = 0.8
        else:
            self._tier = 3
            self._speed = 0.2
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
