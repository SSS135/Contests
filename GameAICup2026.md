# GameAICup 2026 - Space Miners

**Built an attention-based neural network bot trained with PPO + Python/JAX.**

https://github.com/user-attachments/assets/bc9c47e4-21fe-49a0-8d92-27f29dbdcf79

## Introduction

I spent the first two days reimplementing the sim engine in Rust and playing with handcrafted strategies. These got me to around 40th place. Not great, not terrible.

Then I changed approach by introducing Reinforcement Learning. A self-play PPO agent with little handcrafted heuristics. The architecture is an attention-based transformer that processes variable-count game entities (ships, asteroids, bases), producing both a policy and value estimate. Training ran in Python with JAX/Flax, inference and simulation in Rust. At inference time, Gumbel AlphaZero search improved the raw policy output within the 50ms budget.

The training infrastructure was custom-built. A Rust PyO3 module exposed vectorized environments directly to the Python training loop. The Rust simulation was a reimplementation of the Python Box2D engine, validated against golden-file tests and step-equivalence checks. It didn't match 100% - some small divergences remained - but was close enough.

The final submission was a single self-contained Rust file with network weights baked as base64-encoded constants. The attention model forward pass, including AVX2-optimized matrix multiplication, ran in ~115 microseconds on my CPU, leaving headroom for multi-step lookahead search within the 50ms turn budget.

## Model Architecture

The agent used an entity-based transformer architecture, processing entities directly and handling variable asteroid counts through masking.

### Observation Space

The observation was structured as three entity types, each with a dedicated feature vector:

- **6 ships** (3 own + 3 opponent), 41 features each: position, velocity, speed, energy, distance to own base, nearest-asteroid edge distance and push-range flag, push alignment (dot product toward base), push-will-score prediction, bearing to nearest asteroid, closing speed, nearest ally/enemy distances, enemy approach speed, shared-target flag with nearest ally, 4 upgrade levels, team ID, relative position to nearest asteroid, and push counterfactuals for the 2 nearest asteroids (ETA deltas and hit-position shifts before/after a hypothetical push).
- **Up to 20 asteroids**, 32 features each: position, velocity, size one-hot, score, distances to both bases, approach velocities toward each base, nearest own/enemy ship distances, push alignment from nearest own ship, push-will-score, enemy push alignment, base-hit ETAs, will-hit flags, and sinusoidal encoding of predicted hit positions (sin/cos of normalized x,y for both bases — 8 features).
- **1 global token**, 13 features: tick, score differential, both scores, asteroid count, round flags, total energy for both teams.

Running observation normalization (online mean/variance tracking) was applied during training and the final statistics were baked into the exported inference weights.

### Transformer Backbone

Separate linear projections per entity type mapped to a shared 64-dim space, with learned type embeddings. Two self-attention blocks (4 heads, head_dim=16, FFN dim 256, GELU). Asteroid padding with attention masking. ~139K actor parameters total.

```
 Ships [6×41]     Asteroids [≤20×32]     Global [1×13]
      │                   │                    │
 Linear+Bias         Linear+Bias          Linear+Bias
      │                   │                    │
      ▼                   ▼                    ▼
 [6×64]              [≤20×64]              [1×64]
      │                   │                    │
      +── type_embed ──+──── type_embed ───+── type_embed
      │                   │                    │
      └───────────────────┴────────────────────┘
                          │
                    [≤27 × 64]  (concatenated entities)
                          │
                ┌─────────┴─────────┐
                │  Self-Attn Block  │ ×2
                │  (Pre-LN, 4 heads)│
                │  FFN 64→256→64    │
                └─────────┬─────────┘
                          │
                    [≤27 × 64]  (entity representations)
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
         Actor Head             Critic Head
       (separate weights)     (separate weights)
```

### Autoregressive Actor Head

Ships were sorted by x-position for canonical ordering. Ship 0 acted independently; ship 1 conditioned on ship 0's action via a learned embedding table; ship 2 conditioned on both prior ships' actions. This enabled inter-ship coordination without explicit role assignment. Not sure how useful the autoregression actually was, but I added it just in case.

```
Entity representations from backbone
          │
     Sort ships by x-position → [ship_0, ship_1, ship_2]
          │
          ▼
   ┌─────────────┐
   │   Ship 0    │──→ Linear(64, 147) ──→ softmax ──→ action_0
   └─────────────┘                                        │
                                                    ar_embed_slot0
                                                          │
   ┌─────────────┐                                        ▼
   │   Ship 1    │──→ concat(repr, embed_0) ──→ Linear ──→ softmax ──→ action_1
   └─────────────┘                                                        │
                                                                    ar_embed_slot1
                                                                          │
   ┌─────────────┐                                                        ▼
   │   Ship 2    │──→ concat(repr, embed_0, embed_1) ──→ Linear ──→ softmax ──→ action_2
   └─────────────┘
```

### Flat Discrete Action Space

Each ship selected from 147 discrete actions:

- **Idle (1 action):** No acceleration, no push.
- **Push repeat (1 action, duplicated at index 1 and 142):** Repeat previous turn's acceleration with push enabled.
- **Global move (40 actions):** 8 compass directions × 5 speeds (0.5, 1.0, 2.0, 3.0, 5.0), no push.
- **Global move + push (40 actions):** Same 8 × 5 grid, with push enabled.
- **Go-to asteroid (15 actions):** Accelerate toward one of the 3 nearest asteroids at 5 speed levels.
- **Flee asteroid (15 actions):** Accelerate away from one of the 3 nearest asteroids at 5 speed levels.
- **Lateral CW / CCW (30 actions):** Orbit one of the 3 nearest asteroids clockwise or counter-clockwise at 5 speed levels — useful for positioning behind an asteroid before pushing.
- **Upgrades (4 actions):** max_speed, max_accel, push_force, energy_efficiency.

The flat space was a deliberate choice over an earlier factored 5-head design (speed, angle, push, upgrade type, upgrade flag) where the combinatorial explosion made exact search intractable. With a flat categorical, I could enumerate all actions for Gumbel search without approximation.

### Unified Critic

A single scalar value head with mean pooling. The critic shared the architecture with the actor but used separate backbone weights.

```
Entity representations (from critic backbone)
          │
     Mean pool over all entities
          │
          ▼
      [64-dim]
          │
     Linear(64, 1)
          │
          ▼
     scalar value V(s)
```

## Training and Search

Custom PPO in JAX/Flax, no external RL frameworks. Self-play against a pool of frozen past checkpoints, 64 parallel environments, 1000-step rollouts. Reward: per-step score differential `(my_score_delta - opp_score_delta) / 10.0` - simple zero-sum, no shaping needed. Running z-score observation normalization. Separate actor/critic optimizers with independent gradient clipping.

Training used mixed-round sampling (randomly assigned Round 1, Round 2, or Final Round). In 20% of games, each ship started with 0-8 random upgrade levels, forcing the agent to learn against upgraded opponents.

First attempts used behavior cloning from top players' replays, but results were lacking. Pure PPO self-play converged to much stronger play. The iteration cycle was tight: train checkpoint, export to baked Rust weights, TrueSkill tournament against all previous versions, select best, repeat.

Teaching the network to use upgrades was tricky. Since upgrades cost score points, the short-term signal was negative and the agent initially learned to ignore them entirely. The random starting upgrades augmentation helped: by encountering upgraded opponents regularly, the agent learned that upgrades were worth the investment.

At inference time, the submission applied Gumbel AlphaZero search to improve upon the raw policy. For each of the 3 own ships independently:

1. **Candidate generation:** Top 8 actions by logit, perturbed with Gumbel(0,1) noise (sampling without replacement via the Gumbel-Top-k trick).
2. **Phase 0 (depth 1):** Evaluate all 8 candidates with 1-step sim lookahead + critic. Other own ships use their current best action; opponent uses the same NN policy without search. Score: `gumbel_score + 50.0 * (value - mean) / range`. Keep top 4.
3. **Phase 1 (depth 2):** 2-step rollout for the surviving 4. Re-score with critic. Pick the winner.

Total cost per turn: 26 actor forward passes, 36 critic evaluations, 48 sim steps. Search improved win rate to 95-98% vs the same model without search.

## Submission Pipeline and Reflections

**Export pipeline.** `.npz` checkpoint → Python export script → `baked_weights.rs` (FP32 as base64 constants) → Cargo build → custom bundler inlines all crate dependencies into a single `main.rs`. No dependencies beyond serde/serde_json. The final submission was ~4,000 lines of Rust: attention network, physics sim for search, Gumbel search, and all weight data.

**What worked.** Entity-based attention handled variable asteroid counts naturally. Pure self-play PPO converged to creative strategies without any reward engineering - the network learned on its own to prioritize push_force upgrades, the most impactful upgrade in the game. Building the Rust sim from day one made everything else possible: fast training, fast evaluation, fast search. Gumbel search provided a clean improvement over the raw policy, especially for multi-ship coordination.

**What didn't work or was abandoned.** Behavior cloning from top players' replays failed to reach teacher-level quality. A factored 5-head action space required approximate search over the combinatorial joint space; replacing it with flat 147 actions enabled exact Gumbel enumeration. Phasic Policy Gradient (PPG) and Gumbel AlphaZero training both failed to outperform plain PPO - possibly bugs, possibly just not enough time to tune them properly. Handcrafted strategies were interesting but fell short of PPO self-play.
