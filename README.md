# Measuring Starvation and Fairness in Resource Competition Between Agents with "Emotions"

CS 4580/5580 Final Project

---

## Overview

This project builds a multi-agent resource competition simulation in which agents compete
for limited resources organized into reward tiers. Each agent maintains interpretable
internal state variables inspired by emotion — frustration, reward reinforcement, and
social cost — which accumulate over time and drive decision-making. The system measures
starvation and fairness outcomes and compares this emotive approach against classical
resource competition strategies.

The core question: **do emotion-inspired behavioral regulators produce fairer, more
starvation-resistant resource allocation than classical scheduling approaches in
heterogeneous agent populations?**

---

## How to Run

**Homogeneous experiment** (all agents use the same strategy):
```bash
# Default: 6 agents, 200 ticks, averaged over 10 seeds
python run.py

# More ticks, more seeds
python run.py --ticks 500 --seeds 20

# Single seed with full output and per-agent explanations
python run.py --seed 42 --explain

# Specify number of agents
python run.py --agents 10 --ticks 500
```

**Mixed population experiment** (all strategies compete simultaneously):
```bash
# Default: 2 of each strategy = 6 agents, 500 ticks, 10 seeds
python run_mixed.py

# More agents per strategy
python run_mixed.py --per-strategy 3 --ticks 1000

# Single seed with per-agent explanations
python run_mixed.py --seed 42 --explain
```

Output plots are saved to `results/`:
- `comparison.png` — fairness and starvation bar chart with error bars (homogeneous)
- `frustration_over_time.png` — mean agent frustration trajectory per emotion condition
- `tier_distribution.png` — where agents spend their time across tiers
- `mixed_comparison.png` — reward, win rate, starvation per strategy (mixed population)
- `mixed_reward_distribution.png` — individual agent reward scatter per strategy

---

## Project Structure

```
agents.py       — Emotion-based Agent class
simulation.py   — Tick loop and competition resolution
baselines.py    — ExponentialBackoffAgent, PriorityAgingAgent, RoundRobinAgent
metrics.py      — Jain's fairness index, starvation rate, tier distribution, win rates
run.py          — Homogeneous experiment: one strategy per run, all conditions compared
run_mixed.py    — Mixed population experiment: all strategies compete simultaneously
DESIGN.md       — Full design document with all decisions and rationale
results/        — Output plots
```

---

## System Design

### Resource Tiers

Three tiers reset every tick. One winner per tier per tick. Agents commit to a tier
simultaneously with no knowledge of others' choices (imperfect information).

| Tier | Reward | Contention |
|------|--------|------------|
| 1    | 10 units | High     |
| 2    | 5 units  | Medium   |
| 3    | 2 units  | Low      |

### Agent Attributes (Heterogeneous)

Agents differ in fixed attributes set at creation:

```
skill_level    — base bid strength
regen_rate     — energy recovered per tick
max_energy     — energy cap
persistence    — base tendency to try harder or stay when frustrated
risk_aversion  — base tendency to drop to a lower tier when frustrated
```

### Emotional State

Each agent carries three emotional variables that evolve each tick:

```
frustration          — builds on loss, decays on win
reward_reinforcement — builds on win, signals recent success
social_cost          — builds on consecutive wins, handicaps dominant agents
```

### Decision Making

Each tick an agent chooses a target tier and how much speed (energy) to commit.
Speed increases bid strength but costs energy regardless of outcome.

**Bid formula:**
```
bid = skill_level × speed_used × (1 + frustration) × (1 - social_cost)
```

When frustration exceeds a threshold, the agent faces a three-way decision:

| Choice | Action | Logic |
|--------|--------|-------|
| Stay | Same tier, same speed | Wait for an opening |
| Try Harder | Increase speed, spend more energy | Fight for the resource |
| Give Up | Drop to a lower tier | Accept a smaller reward |

### Learning

Agents start with behavioral weights initialized from personality (persistence,
risk_aversion). Weights shift gradually over time based on outcomes:

```
# Try harder succeeded → reinforce that behavior
p_try_harder += learning_rate × (1 - p_try_harder)

# Try harder failed → penalize and nudge toward give_up
p_try_harder -= learning_rate × p_try_harder
p_give_up    += learning_rate × (1 - p_give_up)
```

This means agents with identical starting personalities diverge based on experience.
Early luck matters — a skilled agent that loses early may learn to give up.

### Social Awareness (Condition B)

When social awareness is enabled, agents self-limit when dominating:

- **Self-monitoring:** tracks own recent win rate; increases social cost when above threshold
- **Other-awareness (transparent mode):** observes others' frustration; increases social cost when others are suffering
- **Hard yield:** if social cost crosses a threshold, agent is forced to a lower tier and social cost resets

This produces voluntary yielding — dominant agents back off not because they are forced to,
but because their accumulated social cost handicaps their bids.

---

## Experimental Conditions

Four emotion conditions × three baselines, run across multiple random seeds:

| Condition | Social Awareness | Visibility |
|-----------|-----------------|------------|
| Emotion \| No Social \| Blind | Off | Own state only |
| Emotion \| No Social \| Transparent | Off | Sees all agents' emotional state |
| Emotion \| Social \| Blind | On | Own state only |
| Emotion \| Social \| Transparent | On | Sees all agents' emotional state |
| Baseline \| Priority Aging | — | OS-style: priority grows each lost tick, resets on win |
| Baseline \| Exp Backoff | — | After N losses: wait 2^N ticks before retrying |
| Baseline \| Round Robin | — | Fixed rotation: each agent gets tier 1 in turn |

---

## Results (6 agents, 500 ticks, 10 seeds)

Both baselines were updated to use all three tiers before these results were collected.
Exponential backoff now steps down progressively (tier 1 → tier 2 → tier 3 wait) and
priority aging uses MLFQ-style demotion (drop a tier every 8/20 ticks without a win,
promote back to tier 1 on any win). This makes the comparison fairer.

| Condition | Jain's Fairness | Starvation Rate |
|-----------|----------------|-----------------|
| Emotion \| Social \| Blind | **0.9416 ± 0.022** | **0.000** |
| Emotion \| Social \| Transparent | 0.9400 ± 0.023 | 0.000 |
| Emotion \| No Social \| Transparent | 0.9395 ± 0.015 | 0.000 |
| Emotion \| No Social \| Blind | 0.9270 ± 0.017 | 0.000 |
| Baseline \| Round Robin | 0.9153 ± 0.001 | 0.000 |
| Baseline \| Priority Aging | 0.7991 ± 0.114 | 0.000 |
| Baseline \| Exp Backoff | 0.4221 ± 0.000 | 0.000 |

Jain's Fairness Index ranges from 1/n (maximally unfair) to 1.0 (perfectly equal).
Starvation rate is the fraction of agents going 20+ consecutive ticks without any reward.

### Key Findings

**All emotion conditions outperform priority aging and exponential backoff on fairness.**
The emotion system achieves 0.93–0.94 fairness with zero starvation. Priority aging
reaches 0.80 with high variance. Exponential backoff improves significantly over the
original implementation (0.42 vs 0.24) once tier 2 is included, but still trails
emotion by a wide margin.

**Tier-aware baselines are a fairer comparison.** Adding tier 2 to both baselines
meaningfully improved their performance — backoff no longer death-spirals and priority
aging distributes more evenly. This validates the limitation we identified: the original
results overstated the emotion system's advantage.

**Priority aging solves starvation but not fairness.** The winner resets to base priority
while others remain accumulated high, creating instability and reward inequality. The
high variance (±0.11) reflects sensitivity to the specific skill distribution each seed.

**Social awareness improves fairness.** The inclusion mechanic (social cost) produces
measurably fairer outcomes than purely self-interested emotion agents, confirming that
voluntary yielding by dominant agents contributes to system-level fairness.

**Counterintuitive: Social Blind (0.9416) slightly edges Social Transparent (0.9400).**
Seeing others' frustration causes agents to over-adjust their speed downward, reducing
their effectiveness without proportionally improving fairness. More information is not
always better.

### Interpretability

Every agent decision is fully explainable by its internal state. Example output:

```
Agent 3 | tick decision:
  target tier : 1
  speed used  : 0.7  (bid=0.367)
  frustration : 0.25 (before) → 0.10 (after)  |  action: try_harder
  reinforcement: 0.99  social_cost: 0.0
  weights     : try_harder=0.353  give_up=0.561  stay=0.087
  energy      : 92.5 / 100.0
  outcome     : WON  reward=10
```

The agent targeted tier 1, committed high speed because frustration triggered try_harder
(sampled from learned weights), won, and frustration decayed as a result. No black box.

---

## Mixed Population Results (2 per strategy = 6 agents, 500 ticks, 10 seeds)

To address the homogeneous population limitation, a second experiment ran all three
strategies competing simultaneously in the same simulation.

| Strategy | Mean Reward | Win Rate | Starvation |
|----------|-------------|----------|------------|
| Priority Aging | **2028.8 ± 563** | 0.406 ± 0.113 | 0.000 |
| Exp Backoff | 1221.3 ± 450 | **0.474 ± 0.065** | 0.000 |
| Emotion | 694.0 ± 106 | 0.397 ± 0.001 | **0.000** |

Overall system fairness collapsed to **0.6739 ± 0.097** — far below the 0.93 seen in
homogeneous emotion populations. With tier-aware baselines, backoff becomes significantly
more competitive (mean reward 1221 vs 730 previously), making the mixed population
results more meaningful.

### What This Reveals

**Priority aging exploits emotional cooperation.** When emotion agents get frustrated
and migrate to lower tiers, priority aging agents accumulate priority and hammer tier 1
unopposed. The social cost mechanic — designed to make dominant agents yield — causes
emotion agents to yield ground that priority aging agents immediately occupy.

This is an instance of the **cooperator-defector problem** studied in evolutionary game
theory. Emotional regulation is a cooperative strategy: it works optimally when all
participants play by the same norms. When a purely self-interested strategy (priority
aging never yields) enters the mix, it exploits the cooperators.

**Emotion agents remain the most consistent.** Despite lower mean rewards, emotion agents
have the tightest variance (±105 vs ±633 for priority aging). Emotion agents never
starve and maintain stable behavior across all seeds. Priority aging wins more on average
but is highly dependent on the specific skill distribution of that run.

**The key insight:** emotion-based regulation is a *social contract*. Its effectiveness
depends on adoption. In a uniform population it produces near-optimal fairness. In a
mixed population it is exploitable — which raises the question of what enforcement or
incentive mechanisms would make cooperation stable.

---

## Limitations and Concerns

These results are directionally correct but should be interpreted with the following
caveats:

### 1. Baselines are now tier-aware (addressed)

Both baselines were updated to use all three tiers. Exponential backoff steps down
progressively on consecutive losses (tier 1 → tier 2 → tier 3 wait). Priority aging
uses MLFQ-style demotion based on time since last win. This meaningfully improved both
baselines and produces a fairer comparison. Emotion agents' advantage is real but smaller
than the original results suggested.

### 2. Priority aging is untuned

The aging rate (0.08 per lost tick) was chosen without parameter search. A well-tuned
priority aging system would likely perform significantly better. The results show what
priority aging does at one parameter setting, not its best-case performance.

### 3. Homogeneous populations (partially addressed)

The homogeneous experiment tests each strategy against itself. The mixed population
experiment addresses this directly — see Mixed Population Results above. The mixed
results reveal that emotion agents are exploitable by selfish strategies, which is an
important qualification of the homogeneous findings.

### 4. Exponential backoff is a weak comparison

Exponential backoff was not designed for heterogeneous multi-tier competition. It fails
so completely (0.235 fairness, 50% starvation) that it mostly serves as a lower bound
rather than a meaningful baseline. Its inclusion is useful to demonstrate the starvation
problem, but it should not be the primary comparison for the emotion system.

### 5. Fairness metric rewards participation

Jain's index is computed over total rewards. Because emotion agents win across all three
tiers while baselines concentrate on tier 1, emotion agents accumulate more total rewards
and with more even distribution. This inflates the fairness metric relative to a metric
that only measures tier 1 access equity.

### 6. Seeds are not truly independent experiments

Seeds 0–9 vary agent attribute distributions (skill, regen, personality) but not the
fundamental system dynamics. Variance across seeds reflects sensitivity to agent
attributes, not to different environmental conditions.

---

## What This System Is Not

- **Not a neural network.** All decisions are driven by explicit, inspectable state variables
  and deterministic update rules. Every decision has a traceable explanation.
- **Not a classical scheduler.** Unlike OS schedulers, agents are self-interested and
  adaptive. Fairness is emergent, not enforced.
- **Not a complete solution.** This is a simulation with known limitations. The results
  suggest emotion-based regulation is promising but require stronger baselines and mixed
  population testing to fully validate.

---

## Future Work

- Tier-aware priority aging baseline to isolate emotional contribution
- Stability mechanisms for mixed populations: what incentives or enforcement make
  emotional cooperation stable against selfish strategies (analogous to mechanism
  design in game theory)
- Parameter sensitivity analysis for aging rate, frustration threshold, learning rate
- LLM-hybrid extension: use a language model to set initial emotional state based on
  contextual situation description, with deterministic rules handling evolution
- Survival mechanics: agents with finite lifespan must accumulate minimum resources to
  survive, adding existential stakes to the emotional dynamics
- Formal analysis of convergence and fairness bounds
