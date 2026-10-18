# FAQ — Anticipated Questions & Evidence-Based Answers

### Q: Did Tesla actually confirm AI is used?
**A:** Tesla confirmed the actuators make **"micro adjustments within a fraction of a second"** and that flight maneuvers are autonomous because humans cannot react fast enough. Tesla has **not** published the algorithm itself. The AI conclusion is an inference from confirmed autonomous high-speed control, not a direct Tesla quote.

### Q: Isn't this just a stunt? Why call it AI?
**A:** The stunt is the *demonstration*, not the technology. The same cold-gas RCS architecture is used on Falcon 9 for **real attitude control during landing**. The Roadster is a terrestrial testbed for the same control problem.

### Q: Where is the proof that AI can control cold-gas thrusters?
**A:** Academic work has already used **reinforcement learning** to land a Falcon 9 model using cold-gas thrusters for balance and attitude. This is a published control strategy, not speculation.

### Q: Has anyone independently verified the Roadster's specs?
**A:** No. Production is scheduled for **2028**, and specifications have **not been independently verified**. This is exactly why the article is framed as a hypothesis about the *control requirements* of demonstrated hardware, not a claim about verified performance.

### Q: What would falsify this hypothesis?
**A:** Any of the following:
- Tesla stating the jump was purely ballistic (no active control)
- A confirmed purely mechanical stabilization system
- A confirmed human-in-the-loop trigger for every thruster firing
- A Tesla patent or paper showing a fundamentally different control approach

### Q: Is this repo affiliated with Tesla or SpaceX?
**A:** No. It is independent open engineering analysis.

### Q: Can I cite this?
**A:** Yes — MIT licensed. See README for citation guidance.

### Q: Where can I learn more about cold-gas thruster control?
**A:** Start with Falcon 9 RCS literature, Starhopper hover test reports, and academic reinforcement-learning-for-landing papers. The full reference list is in `docs/REFERENCES.md`.