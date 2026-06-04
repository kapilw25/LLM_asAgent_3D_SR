No RL. The plan contains zero policy optimization (no PPO/GRPO/DPO/RLHF/KTO): those need actions, rewards, or preference pairs, and passive walking video has none. m12g is planning evaluation (best-of-K latent shooting toward a goal, MPC-style) — the "control" half of a world model, not RL training. The only true-RL item is the explicitly deferred TD-JEPA/OGBench offline-RL probe (doc §6, post-AAAI, needs a real RL stack).

Next step: WEBSEARCH to find real Reinforcmeent learning extension of FactorJEPA 
which can add to NOVELTY
but can be implemented [both trianing & Eval on POC 10k Clips] within 1 week