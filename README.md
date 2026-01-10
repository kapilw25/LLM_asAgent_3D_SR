# LLM as Agent for 3D Navigation

🗽🤖 **It takes a village to raise a kid… and a whole city to raise an AI agent**

![NYC AI Agents Sandbox](Literature/proposal/nyc_ai_agents.gif)

We're building a **sandbox for AI agents** — where they learn to move, navigate, coordinate, and survive in complex 3D environments.

This is where we stress-test the real questions:
- 🔀 **How** do agents move, talk, negotiate, and adapt when the world is messy?
- 🌐 **What** emerges when agents share the same streets—cooperate, compete, or collide?
- 🛡️ **How** Do we keep them safe, aligned, and socially aware while they learn?

---

## Quick Start

```bash
# Setup
chmod +x setup_env.sh
./setup_env.sh
source venv_3Denv/bin/activate

# Run tests
python src/m01_shortest_path.py --test    # Run all tests
python src/m01_shortest_path.py --demo    # Show demo with visualization
python src/m02_hybrid_judge.py --test      # Test algorithmic judge
python src/m02_hybrid_judge.py --compare   # Compare algo vs LLM
python src/m02_hybrid_judge.py --demo      # Run demo evaluation
```

---

## References

- [Current Plan](iter/iter1/plan1.md)
- [Proposal](Literature/proposal/Proposal_LLMs_as_Agents.pdf)
- [Experimental Framework](Literature/proposal/experimental_framework.pdf)
