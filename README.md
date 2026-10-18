<div align="center">

# 🧠 The Flying Tesla Has an AI Brain

**Everyone saw the jump. Nobody saw the control system.**

[![Live Page](https://img.shields.io/badge/🌐_Live_Page-Visit-8b5cf6?style=for-the-badge)](https://YOUR-USERNAME.github.io/tesla-roadster-ai-brain/)
[![License: MIT](https://img.shields.io/badge/License-MIT-3e6ae1?style=for-the-badge)](LICENSE)
[![Status: Hypothesis](https://img.shields.io/badge/Status-Engineering_Hypothesis-e82127?style=for-the-badge)](#-scope--honesty-note)

</div>

---

## 🚀 The One-Liner

The Tesla Roadster's mid-air barrel roll went viral. But the real engineering story isn't the cold-gas thrusters — it's the **autonomous control layer** that must decide, in milliseconds, when and how hard to fire them.

**This repo is an engineering analysis of that invisible layer.**

---

## 🎯 What This Repo Is

- A **visual explainer** of the AI control stack behind the "flying" Roadster
- A **curated evidence base** linking confirmed hardware to required control systems
- A **hypothesis document** — not a leak, not a claim of insider knowledge
- A **teaching resource** for anyone interested in Physical AI, aerospace control, or cold-gas thruster systems

## 🚫 What This Repo Is Not

- Not affiliated with Tesla or SpaceX
- Not a source of proprietary algorithms (Tesla has not released them)
- Not a claim that the production car will fly as demonstrated
- Not investment or product advice

---

## 📖 The Core Argument in 60 Seconds

| Layer | What Tesla Confirmed | What It Implies |
|---|---|---|
| **Effectors** | 10 cold-gas thrusters, millisecond electric actuators, 600L compressed air | High-frequency electronic control is mandatory |
| **Autonomy** | Human drivers "could not safely manage reaction times or G-forces" | An autonomous controller — not a human — is flying the car |
| **Heritage** | System derived from Falcon 9 / Starship attitude control | Computer-controlled cold-gas RCS is a proven paradigm |
| **Actuation** | Actuators reportedly from Tesla Optimus | Robotics-grade control stack underneath |
| **Missing piece** | Tesla has *not* published the algorithms | The AI layer is inferred from the confirmed autonomy requirement |

👉 **Read the full methodology:** [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
👉 **See the evidence base:** [docs/EVIDENCE.md](docs/EVIDENCE.md)
👉 **Anticipated questions answered:** [docs/FAQ.md](docs/FAQ.md)

---

## 🌐 Live Page

The full visual explainer is a single self-contained HTML file (`index.html`) with:

- Animated sensor-fusion diagram
- Cold-gas thruster flow diagram
- Jump animation
- Color-coded evidence layers
- Full reference list with clickable sources

**View it locally:** open `index.html` in any browser — no build step, no dependencies.

---

## 📚 References

All claims are backed by on-site reporting from the October 2026 launch. See [docs/REFERENCES.md](docs/REFERENCES.md) for the full list with links.

Primary sources include **TechCrunch**, **Engadget**, **AVweb**, **Torque News**, and **Universe Magazine**.

---

## ⚖️ Scope & Honesty Note

> **Tesla has not publicly released the exact AI control algorithms or sensor-fusion architecture used during the jump.** The "AI brain" explanation in this repo is an **accurate synthesis of the system's publicly described behaviour** — an engineering inference from confirmed hardware and confirmed autonomous flight requirements. It is **not** a disclosure of proprietary code.

This repo is published in the spirit of **open engineering analysis**, not speculation.

---

## 🛠️ How to Use This Repo

**Just want to read?** → Open `index.html` or visit the [live page](https://YOUR-USERNAME.github.io/tesla-roadster-ai-brain/)

**Want to cite it?** → See [CITATION.cff](#) *(add if you want formal citation)*

**Want to contribute?** → See [CONTRIBUTING.md](CONTRIBUTING.md) *(add if opening to PRs)*

**Want to fork and remix?** → MIT licensed — go for it. Attribution appreciated.

---

## 🧑‍💻 Local Development

```bash
# Clone
git clone https://github.com/YOUR-USERNAME/tesla-roadster-ai-brain.git
cd tesla-roadster-ai-brain

# Open in browser (no build step required)
open index.html        # macOS
xdg-open index.html    # Linux
start index.html       # Windows