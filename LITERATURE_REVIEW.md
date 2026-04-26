# Literature Review: Emotion-Based Multi-Agent Resource Competition

CS 5580 Final Project — Annotated Bibliography

This review covers four areas relevant to the project: (1) emotion and affect in AI agents,
(2) fairness and starvation in resource allocation, (3) classical scheduling strategies used
as baselines, and (4) cooperation and defection in multi-agent systems.

---

## 1. Emotion and Affect in Artificial Agents

**Picard, R. W. (1997). *Affective Computing*. MIT Press.**

Picard's foundational text argues that emotion is not peripheral to intelligence but central
to it — that systems capable of recognizing, expressing, and having functionally emotional
states will be more adaptive and effective. The book distinguishes between emotion as a
communication signal and emotion as an internal regulatory mechanism. This project draws
directly on the regulatory view: frustration, reward reinforcement, and social cost are not
outputs displayed to others but internal variables that modulate bidding behavior and tier
selection. Picard's framing justifies treating emotion as a control variable rather than a
label.

---

**Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of Emotions*.
Cambridge University Press.**

The OCC model provides a systematic account of how emotions arise from appraisals of events
relative to goals, standards, and attitudes. Under OCC, frustration maps to disappointment
or distress arising from undesirable outcomes relative to expectations — exactly the
accumulation pattern used in this project (frustration builds on loss, decays on win). The
model also distinguishes emotions by their action tendencies: distress motivates either
persistence or withdrawal, corresponding directly to the try_harder / give_up decision
implemented here. OCC gives the emotional state variables a grounded theoretical basis rather
than treating them as arbitrary knobs.

---

**Velásquez, J. D. (1997). Modeling emotions and other motivations in synthetic agents.
*Proceedings of the Fourteenth National Conference on Artificial Intelligence (AAAI-97)*.**

Velásquez proposes a computational model in which multiple emotion processes run in parallel,
each monitoring the environment for conditions that trigger them, and compete to influence
behavior based on their intensity. This is structurally similar to the three-way frustrated
decision (try_harder, give_up, stay) implemented here: multiple possible actions are
weighted by learned probabilities derived from emotional state, and one is sampled. The key
shared insight is that behavior under emotional arousal is probabilistic rather than
deterministic, and the probabilities shift with experience — which this project implements
via the learning rate update to p_try_harder and p_give_up after each outcome.

---

**Gratch, J., & Marsella, S. (2004). A domain-independent framework for modeling emotion.
*Cognitive Systems Research*, 5(4), 269–306.**

Gratch and Marsella's EMA (Emotion and Adaptation) framework models emotion as arising from
the continuous appraisal of events against goals, with emotions feeding back into replanning
and coping behavior. Their notion of coping — problem-focused (change the situation) vs.
emotion-focused (change the interpretation) — maps onto the try_harder vs. give_up split in
this project. An agent that increases speed when frustrated is engaging in problem-focused
coping; one that drops to a lower tier is engaging in emotion-focused coping (accepting a
reduced goal). EMA also emphasizes that emotional state should be fully auditable at each
step, which aligns with the interpretability requirement of this project.

---

**Bates, J. (1994). The role of emotion in believable agents. *Communications of the ACM*,
37(7), 122–125.**

Bates focuses on emotion as the mechanism that makes synthetic agents appear believable and
purposeful rather than mechanical. He argues that even simple emotional dynamics — agents
that respond to success and failure in ways that mirror frustration and satisfaction —
produce qualitatively richer behavior than purely reactive systems. This project is not
primarily about believability, but the same principle applies to interpretability: agents
whose behavior changes in traceable response to accumulated emotional state are more
understandable to an observer than agents following fixed rules, because their state history
explains their current choices.

---

**Liu, H., Dai, Y., Tan, H., Lei, Y., Zhou, Y., & Wu, Z. (2025). Outraged AI: Large language
models prioritise emotion over cost in fairness enforcement. *arXiv:2510.17880 [cs.CL]*.
https://doi.org/10.48550/arXiv.2510.17880**

This paper provides the first causal evidence that emotion guides moral decision-making in
LLM agents, tested through altruistic third-party punishment — where an agent incurs
personal cost to enforce fairness norms. Across 796,100 decisions, LLM agents used negative
emotion (elicited by unfairness) to drive punishment behavior, often more strongly than
human participants. This directly supports the theoretical motivation of the present project:
emotion functions as a causal regulator of agent behavior, not merely a descriptive label.
Critically, their finding that LLMs prioritize emotion over cost — enforcing norms in an
almost all-or-none manner — contrasts with the behavior observed here, where emotion agents
are arguably too cost-sensitive, yielding readily when dominating. The divergence suggests
that rule-based emotional variables (this project) and LLM-emergent emotion differ in their
cost calibration. Their proposed future direction — integrating emotion with context-sensitive
reasoning — aligns with the LLM-hybrid extension identified in this project's future work.

---

## 2. Fairness and Starvation in Resource Allocation

**Jain, R., Chiu, D., & Hawe, W. (1984). A quantitative measure of fairness and
discrimination for resource allocation in shared computer systems. *DEC Technical Report
TR-301*.**

This paper introduces Jain's Fairness Index, the primary metric used to evaluate outcomes
in this project. The index J = (Σxᵢ)² / (n · Σxᵢ²) ranges from 1/n (maximally unfair,
one agent receives everything) to 1.0 (perfectly equal distribution). Jain et al. motivate
the metric by showing it satisfies four desirable properties: population size independence,
scale independence, boundedness, and continuity. Using this metric rather than simple
variance or Gini coefficient allows fair comparison across conditions with different numbers
of agents and total reward levels.

---

**Corbató, F. J., Merwin-Daggett, M., & Daley, R. C. (1962). An experimental time-sharing
system. *Proceedings of the AFIPS Spring Joint Computer Conference*, 21, 335–344.**

This paper introduces the Compatible Time-Sharing System (CTSS), one of the earliest
multi-level time-sharing systems, and is the conceptual ancestor of the priority aging
baseline used in this project. The key insight is that processes should be demoted to lower
priority queues the more CPU time they consume, preventing any single process from
monopolizing the system. The priority aging baseline implements this in reverse — priority
accumulates when an agent is not winning (analogous to a process waiting), and resets on a
win (analogous to a quantum expiry). This mirrors how modern OS schedulers handle starvation
prevention.

---

**Metcalfe, R. M., & Boggs, D. R. (1976). Ethernet: Distributed packet switching for local
computer networks. *Communications of the ACM*, 19(7), 395–404.**

Metcalfe and Boggs introduce the CSMA/CD protocol with binary exponential backoff for
collision resolution — the direct inspiration for the exponential backoff baseline. Under
CSMA/CD, a node that detects a collision waits a random interval drawn from an exponentially
growing window before retransmitting, reducing the probability of repeated collisions. The
baseline adapts this to multi-tier resource competition: consecutive losses trigger
progressively longer waits at lower tiers, with the wait window capped at 16 ticks. The
original protocol was designed for network throughput, not fairness, which explains why
the backoff baseline underperforms on Jain's index — it was never optimized for equitable
distribution.

---

## 3. Multi-Agent Systems

**Shoham, Y., & Leyton-Brown, K. (2009). *Multiagent Systems: Algorithmic, Game-Theoretic,
and Logical Foundations*. Cambridge University Press.**

This textbook provides the theoretical scaffolding for reasoning about agent behavior in
shared environments. Particularly relevant is the treatment of mechanism design — the
question of how to structure rules and incentives so that self-interested agents produce
socially desirable outcomes as a byproduct of pursuing their own goals. The mixed population
results in this project illustrate a mechanism design failure: the social cost yield
mechanism was designed assuming all participants would use emotional regulation, but in a
mixed population, purely self-interested agents exploit the voluntary yielding without
contributing to system fairness. The treatment of normal-form games and the concept of dominant strategies directly explain
why priority aging performs well in mixed populations — "never yield" is a dominant strategy
when the opponent yields unconditionally.

---

## 4. Cooperation and Defection

**Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books.**

Axelrod's landmark study of iterated prisoner's dilemma tournaments shows that cooperative
strategies (specifically tit-for-tat) can be evolutionarily stable if the game is repeated
and agents can condition their behavior on the opponent's history. The mixed population
results in this project replicate the core vulnerability: cooperative strategies (emotion
agents that yield when dominating) are exploited by unconditionally defecting strategies
(priority aging, which never yields). Axelrod's finding that tit-for-tat outperforms
unconditional cooperation points directly to the design refinement identified in this project:
social yielding should be conditional on detecting a cooperative opponent, not unconditional.

---

**Nowak, M. A. (2006). Five rules for the evolution of cooperation. *Science*, 314(5805),
1560–1563.**

Nowak synthesizes five mechanisms by which cooperation can evolve against defection: kin
selection, direct reciprocity, indirect reciprocity, network reciprocity, and group
selection. The mixed population results in this project fail on direct reciprocity (emotion
agents have no way to condition on opponent behavior) and on group selection (the simulation
does not select for groups of cooperators). Nowak's framework predicts that cooperation
should be unstable in the mixed population scenario this project tests — which is exactly
what is observed. Importantly, Nowak's network reciprocity condition suggests that if emotion
agents were clustered (only competing against other emotion agents), cooperation would be
stable — consistent with the homogeneous population results showing 0.94 fairness.

---

## Summary

The project sits at the intersection of four bodies of work. From affective computing it
draws the theoretical grounding for emotion as an internal regulatory variable (Picard,
OCC, Velásquez, Gratch & Marsella). From resource allocation it draws the fairness metric
(Jain) and the baseline scheduling strategies (Corbató for priority aging, Metcalfe & Boggs
for backoff). From multi-agent systems it draws the mechanism design framing for why
voluntary cooperation is exploitable (Shoham & Leyton-Brown). From cooperation theory it
draws the prediction that unconditional cooperation is unstable against defectors and the
design implication toward conditional strategies (Axelrod, Nowak).

The central contribution relative to this literature is the empirical demonstration that
emotion-inspired behavioral regulators produce measurably fairer outcomes than classical
scheduling strategies in homogeneous populations, while quantifying the conditions under
which the cooperative mechanism breaks down in mixed populations.
