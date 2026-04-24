# Emotion-Based Multi-Agent Resource Competition — Design Document

## Overview

A multi-agent simulation where agents compete for tiered resources. Agents maintain
interpretable emotional state variables that drive decision-making. The system measures
starvation and fairness outcomes and compares against classical baselines.

---

## Project Structure

```
project/
  agents.py        # Agent class with emotional state and learning
  simulation.py    # Competition loop, tier resolution
  baselines.py     # Exponential backoff, round robin
  metrics.py       # Fairness, starvation tracking
  run.py           # Entry point, experiment configs
  results/         # Output plots, logs
```

---

## Resource Tier Structure

- 3 tiers, **one winner per tier per tick**
- Resource resets each tick — no depletion across rounds
- Agents commit to one tier per tick (locked in), can switch next tick
- Agents have **imperfect information** — they do not know how many others are bidding on the same tier

| Tier | Reward | Contention |
|------|--------|------------|
| 1    | 10 units | High     |
| 2    | 5 units  | Medium   |
| 3    | 2 units  | Low      |

---

## Agent Attributes

### Fixed at Creation (Heterogeneous)
```
skill_level       # base bid strength, varies per agent
regen_rate        # energy recovered per tick
max_energy        # energy cap
personality {
    persistence   # base tendency to try harder / stay
    risk_aversion # base tendency to drop tiers
}
```

### Dynamic State (Changes Each Tick)
```
energy                  # current energy pool
current_tier            # tier agent is targeting this tick
speed_used              # energy committed this tick

# Emotional state
frustration             # builds on loss, decays on win
reward_reinforcement    # builds on win, signals recent success
social_cost             # builds on consecutive wins (Condition B)

# Learned behavioral weights (shift gradually over time)
P(try_harder)           # probability of increasing speed when frustrated
P(give_up)              # probability of dropping to lower tier when frustrated
P(stay)                 # probability of staying same tier/speed when frustrated
```

---

## Bid Formula

```
bid = skill_level * speed_used * (1 + frustration) * (1 - social_cost)
energy -= speed_used * energy_cost_rate
```

- Higher speed = stronger bid, more energy spent
- Frustration increases bid aggressiveness
- Social cost handicaps dominant agents (Condition B)
- Energy is deducted regardless of win or loss
- If energy is low, agent cannot commit high speed

---

## Tick Flow

```
1. Each agent independently decides: target tier + speed (energy commitment)
2. All decisions locked simultaneously — no information about opponents' choices
3. Competition resolved: highest bid wins each tier
4. Winner receives tier reward; losers receive nothing
5. Energy deducted from all agents who bid, win or lose
6. Emotional state updated based on outcome
7. Learned weights updated based on outcome
8. Agents reassess for next tick
```

---

## Emotional State Update Rules

### On Loss
```
frustration += frustration_rate
reward_reinforcement *= decay_rate
```

### On Win
```
frustration = max(0, frustration - frustration_decay)
reward_reinforcement += reinforcement_rate
social_cost += social_cost_rate       # only in Condition B
```

### Energy Recovery (Each Tick)
```
energy = min(max_energy, energy + regen_rate)
```

---

## Frustration Decision Branch

When an agent is frustrated (frustration > threshold), it chooses one of three responses:

```
Choice 1 — Stay:        same tier, same speed. Wait for an opening.
Choice 2 — Try Harder:  increase speed, spend more energy, stronger bid.
Choice 3 — Give Up:     drop to lower tier, conserve energy.
```

Choice is sampled from learned probability weights:
```
P(try_harder) + P(give_up) + P(stay) = 1.0
```

---

## Learning — Gradual Weight Shifting

Personality sets initial weights. Experience nudges them each tick using a learning rate.

```
# If tried harder and won:
P(try_harder) += learning_rate * (1 - P(try_harder))

# If tried harder and lost:
P(try_harder) -= learning_rate * P(try_harder)
P(give_up)    += learning_rate * (1 - P(give_up))

# Normalize after each update so weights sum to 1
```

- `learning_rate` is small — personality dominates early, experience accumulates over time
- Agents with identical starting personalities can diverge significantly based on history
- Early luck matters — a strong agent that loses early may learn to give up

---

## Social Awareness (Condition B Only)

Agents self-limit when they are dominating. Two signals drive this:

### Self-Monitoring
```
own_win_rate = wins_last_N_ticks / N
if own_win_rate > dominance_threshold:
    social_cost increases faster
```

### Other-Awareness
```
# Agent observes other agents' frustration levels
avg_others_frustration = mean(frustration of all other agents)
if avg_others_frustration > empathy_threshold:
    social_cost increases faster
```

### Social Cost Effect
```
# Continuous: reduces bid strength naturally (already in bid formula)
bid = skill_level * speed_used * (1 + frustration) * (1 - social_cost)

# Hard threshold: if social_cost > yield_threshold, agent forced to lower tier this tick
if social_cost > yield_threshold:
    current_tier = max(1, current_tier - 1)
    social_cost = 0   # reset after yielding
```

---

## Experimental Conditions

Number of agents and simulation length are fully configurable. Default simulation length: **200 ticks**.

### Two Axes: Social Awareness × Visibility

| Condition | Social Awareness | Visibility | Description |
|-----------|-----------------|------------|-------------|
| 1 | Off | Blind | Purely self-interested, no knowledge of others |
| 2 | Off | Transparent | No social awareness but can see others' emotional state |
| 3 | On | Blind | Self-monitors own win rate only, unaware of others' state |
| 4 | On | Transparent | Full social awareness — self-monitoring + others' frustration |

### Visibility Conditions Defined

**Blind:**
- Agent knows only its own emotional state
- Does not know how many other agents exist
- Social awareness (if on) uses self-monitoring only — own win rate

**Transparent:**
- Agent sees all other agents' emotional states in full
- Knows the full population size
- Social awareness (if on) uses self-monitoring + others' frustration levels

### Additional Axes to Vary
- `learning_rate` — fast vs slow adaptation
- personality distributions — persistent vs risk-averse populations
- number of agents vs tier slots — controls contention level

---

## Baselines

| Baseline | Description |
|----------|-------------|
| Exponential backoff | After each loss, agent waits exponentially longer before retrying |
| Round robin | Agents take turns in fixed rotation regardless of state |

---

## Metrics

| Metric | Description |
|--------|-------------|
| Jain's Fairness Index | How evenly rewards are distributed across agents |
| Starvation rate | % of agents that go N+ ticks without any reward |
| Tier distribution | How agents spread across tiers over time |
| Win rate per agent | Individual dominance tracking |
| Emotional state over time | Frustration, reinforcement, social cost trajectories |
| Death rate (optional) | If survival mechanics added later |

---

## Interpretability

Every agent decision is explainable by its internal state at that timestep:

```
Tick 42 — Agent 3:
  Targeted tier 1, speed 0.8
  Reason: frustration=0.71 triggered try_harder (P=0.65), reward_reinforcement=0.2
  Outcome: lost (outbid by Agent 1)
  Update: frustration -> 0.78, P(try_harder) slightly reduced
```

This satisfies the course explainability requirement — no black box, full audit trail.

---

## Open Questions (To Settle Before Coding)

- [ ] Exact threshold values for frustration triggers (tune experimentally)
- [ ] Value of N for win rate window in social awareness (self-monitoring)
