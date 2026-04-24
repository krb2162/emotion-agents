import random


TIER_REWARDS = {1: 10, 2: 5, 3: 2}
ENERGY_COST_RATE = 0.15   # fraction of max_energy spent per tick at full speed (1.0)
FRUSTRATION_THRESHOLD = 0.3


class Agent:
    """
    Emotion-based agent that competes for tiered resources.

    Fixed attributes (heterogeneous, set at creation):
        skill_level     - base bid strength
        regen_rate      - energy recovered per tick
        max_energy      - energy cap
        persistence     - base tendency to try harder / stay at same tier
        risk_aversion   - base tendency to drop to a lower tier

    Dynamic emotional state:
        frustration          - builds on loss, decays on win
        reward_reinforcement - builds on win, signals recent success
        social_cost          - builds on consecutive wins (used in social conditions)

    Learned behavioral weights (shift gradually via learning_rate):
        p_try_harder  - probability of increasing speed when frustrated
        p_give_up     - probability of dropping to a lower tier when frustrated
        p_stay        - probability of staying same tier/speed when frustrated
    """

    def __init__(self, agent_id, skill_level, regen_rate, max_energy,
                 persistence, risk_aversion, learning_rate=0.05):
        # Fixed
        self.agent_id = agent_id
        self.skill_level = skill_level
        self.regen_rate = regen_rate
        self.max_energy = max_energy
        self.persistence = persistence
        self.risk_aversion = risk_aversion
        self.learning_rate = learning_rate

        # Dynamic
        self.energy = max_energy
        self.current_tier = 1
        self.speed_used = 0.5

        # Emotional state
        self.frustration = 0.0
        self.reward_reinforcement = 0.0
        self.social_cost = 0.0

        # Learned weights — initialized from personality
        # persistence splits between try_harder and stay
        self.p_try_harder = persistence * 0.6
        self.p_stay = persistence * 0.4
        self.p_give_up = risk_aversion
        self._normalize_weights()

        # Tracking
        self.wins = 0
        self.losses = 0
        self.total_reward = 0.0
        self.win_history = []          # bool per tick
        self.last_frustrated_action = None
        self.state_log = []            # interpretability record

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def decide(self, tick, other_agents=None, social_awareness=False,
               visibility='blind'):
        """
        Choose target tier and speed for this tick.
        Sets self.current_tier and self.speed_used.
        Returns (tier, speed).
        """
        if social_awareness:
            self._update_social_cost(other_agents, visibility)

        # Transparent mode: use others' frustration to gauge competition even
        # without full social awareness — high avg frustration means heavy
        # contention, so conserve energy by reducing speed slightly.
        if not social_awareness and visibility == 'transparent' and other_agents:
            others_frust = [
                a.frustration for a in other_agents
                if a.agent_id != self.agent_id
            ]
            if others_frust:
                avg_frust = sum(others_frust) / len(others_frust)
                competition_factor = 1.0 - 0.25 * avg_frust
                self.speed_used = max(0.1, min(1.0, self.speed_used * competition_factor))

        if self.frustration > FRUSTRATION_THRESHOLD:
            action = self._frustrated_decision()
        else:
            action = 'normal'
            self.last_frustrated_action = None

        if action == 'try_harder':
            self.speed_used = min(1.0, self.speed_used + 0.2)
            self.last_frustrated_action = 'try_harder'
        elif action == 'give_up':
            self.current_tier = min(3, self.current_tier + 1)
            self.speed_used = max(0.1, self.speed_used - 0.15)
            self.last_frustrated_action = 'give_up'
        elif action == 'stay':
            self.last_frustrated_action = 'stay'
        # 'normal' — no change

        # Cap speed by available energy
        max_affordable = self.energy / (self.max_energy * ENERGY_COST_RATE)
        self.speed_used = max(0.1, min(self.speed_used, max_affordable, 1.0))

        return self.current_tier, self.speed_used

    def compute_bid(self):
        """Bid strength — called after decide() each tick."""
        return (self.skill_level
                * self.speed_used
                * (1.0 + self.frustration)
                * (1.0 - self.social_cost))

    def apply_outcome(self, won, reward, social_awareness=False):
        """Update emotional state, energy, and learned weights after resolution."""
        # Capture pre-outcome state for accurate logging
        frustration_before = self.frustration
        social_cost_before = self.social_cost

        # Deduct energy
        energy_cost = self.speed_used * ENERGY_COST_RATE * self.max_energy
        self.energy = max(0.0, self.energy - energy_cost)

        if won:
            self.wins += 1
            self.total_reward += reward
            self.win_history.append(True)

            self.frustration = max(0.0, self.frustration - 0.15)
            self.reward_reinforcement = min(1.0, self.reward_reinforcement + 0.1)
            if social_awareness:
                self.social_cost = min(1.0, self.social_cost + 0.05)

            # Restore ambition — drift tier back up and ease speed
            self.current_tier = max(1, self.current_tier - 1)
            self.speed_used = max(0.3, self.speed_used - 0.1)

            # Learning: reward try_harder if that was the frustrated action
            if self.last_frustrated_action == 'try_harder':
                lr = self.learning_rate
                self.p_try_harder += lr * (1.0 - self.p_try_harder)
                self._normalize_weights()

        else:
            self.losses += 1
            self.win_history.append(False)

            self.frustration = min(1.0, self.frustration + 0.1)
            self.reward_reinforcement = max(0.0, self.reward_reinforcement * 0.95)

            # Learning: penalize try_harder if it failed, nudge toward give_up
            if self.last_frustrated_action == 'try_harder':
                lr = self.learning_rate
                self.p_try_harder = max(0.05, self.p_try_harder - lr * self.p_try_harder)
                self.p_give_up = min(0.85, self.p_give_up + lr * (1.0 - self.p_give_up))
                self._normalize_weights()

        # Passive energy recovery each tick
        self.energy = min(self.max_energy, self.energy + self.regen_rate)

        # Slow decay of reward reinforcement
        self.reward_reinforcement *= 0.99

        self._log_state(won, reward if won else 0, frustration_before, social_cost_before,
                        self.last_frustrated_action or 'normal')

    def explain_last_decision(self):
        """Human-readable explanation of the most recent decision."""
        if not self.state_log:
            return f"Agent {self.agent_id}: no decisions yet."
        s = self.state_log[-1]
        action = self.last_frustrated_action or 'normal'
        lines = [
            f"Agent {self.agent_id} | tick decision:",
            f"  target tier : {s['tier']}",
            f"  speed used  : {s['speed']}  (bid={s['bid']})",
            f"  frustration : {s['frustration']} (before) → {s['frustration_after']} (after)  |  action: {action}",
            f"  reinforcement: {s['reward_reinforcement']}  social_cost: {s['social_cost']}",
            f"  weights     : try_harder={s['p_try_harder']}  give_up={s['p_give_up']}  stay={s['p_stay']}",
            f"  energy      : {s['energy']} / {self.max_energy}",
            f"  outcome     : {'WON' if s['won'] else 'LOST'}  reward={s['reward']}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frustrated_decision(self):
        r = random.random()
        if r < self.p_try_harder:
            return 'try_harder'
        elif r < self.p_try_harder + self.p_give_up:
            return 'give_up'
        else:
            return 'stay'

    def _update_social_cost(self, other_agents, visibility):
        dominance_threshold = 0.6
        empathy_threshold = 0.5
        yield_threshold = 0.7

        own_win_rate = self._recent_win_rate(window=20)
        if own_win_rate > dominance_threshold:
            self.social_cost = min(1.0, self.social_cost + 0.05)

        if visibility == 'transparent' and other_agents:
            others_frustration = [
                a.frustration for a in other_agents
                if a.agent_id != self.agent_id
            ]
            if others_frustration:
                avg = sum(others_frustration) / len(others_frustration)
                if avg > empathy_threshold:
                    self.social_cost = min(1.0, self.social_cost + 0.05)

        # Hard threshold: forced yield to a lower tier, then reset
        if self.social_cost > yield_threshold:
            self.current_tier = min(3, self.current_tier + 1)
            self.social_cost = 0.0

    def _recent_win_rate(self, window=20):
        if not self.win_history:
            return 0.0
        recent = self.win_history[-window:]
        return sum(recent) / len(recent)

    def _normalize_weights(self):
        total = self.p_try_harder + self.p_stay + self.p_give_up
        if total == 0:
            self.p_try_harder = self.p_stay = self.p_give_up = 1 / 3
        else:
            self.p_try_harder /= total
            self.p_stay /= total
            self.p_give_up /= total

    def _log_state(self, won, reward, frustration_before, social_cost_before, action):
        self.state_log.append({
            'tier': self.current_tier,
            'speed': round(self.speed_used, 3),
            'bid': round(self.compute_bid(), 3),
            'won': won,
            'reward': reward,
            'action': action,
            'frustration': round(frustration_before, 3),      # pre-outcome (drove the decision)
            'frustration_after': round(self.frustration, 3),  # post-outcome
            'reward_reinforcement': round(self.reward_reinforcement, 3),
            'social_cost': round(social_cost_before, 3),      # pre-outcome
            'energy': round(self.energy, 3),
            'p_try_harder': round(self.p_try_harder, 3),
            'p_give_up': round(self.p_give_up, 3),
            'p_stay': round(self.p_stay, 3),
        })
