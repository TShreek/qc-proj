# 🧠⚛️ Deadlock Forecast via Quantum Period Analysis

> Predicting deadlocks before they happen by applying Shor-style quantum period-finding to OS-level resource contention patterns.

Modern distributed systems don’t just *hit* deadlocks — they often **repeat** them due to hidden scheduling rhythms (cron bursts, batch jobs, autoscaling waves, GC cycles, etc.).

Traditional deadlock detection triggers **after failure**.  
But what if we could forecast the next deadlock window?

This project simulates a multi-process resource environment, detects real deadlocks classically, and uses a quantum-inspired periodicity estimator (based on Shor’s algorithm’s phase estimation + continued fractions) to **predict future deadlock spikes**.

No mystical “quantum laptop” claims — this is a **systems + quantum algorithms crossover experiment**.

---

## 🎯 Goals

| Goal | Achieved |
|---|---|
Detect real deadlocks | ✅ Tarjan SCC (optimal classical approach)  
Encode system state in modular arithmetic | ✅ Hash + modular exponentiation  
Extract hidden contention periodicity | ✅ Shor-style period-finding  
Forecast future deadlocks | ✅ Prediction windows  
Measure accuracy | ✅ Precision, recall, timeline plot  

---

## 🧩 System Overview

+-----------------------------+
| Process & Resource Simulator|
| (requests, holds, bursts) |
+--------------+--------------+
|
v
+-------------------+
| Wait-For Graph |
+--------+----------+
|
Classical | Tarjan SCC
Deadlock 🧱 Detection
|
v
+---------------------------------+
| Deadlock timestamps (events) |
| + hashed snapshots (y(t)) |
+-----------------+---------------+
|
v
Quantum-Inspired Period Finder
- burst clustering
- Rayleigh phase scan
- continued fractions (Shor-style)
|
v
Estimated period r̂
|
v
+-------------------------------------+
| Forecast upcoming deadlock windows |

---

## ✅ Example Console Output

+-------------------------------------+
▶️ Running simulation...
✔️ Steps=300, deadlocks=46 at times=[21,22,23,...]
🔭 Estimated period r̂ = 24 (true injected = 24)
📊 Forecast: precision=0.67, recall=0.71
📈 Timeline saved to timeline.png

> It didn’t just detect deadlocks — it *anticipated* them.

---

## ⚙️ Install & Run

### Install
uv install # or pip install -r requirements.txt

### Run
uv run deadlock_forecast.py --steps 300 --period 24 --width 4 --window 2

### Useful flags

| Flag | Description |
|---|---|
`--steps` | number of simulation ticks  
`--period` | true injected contention cycle  
`--width` | burst width (how many ticks cluster)  
`--window` | tolerance around forecast ticks  

---

## 📂 Repository Structure

deadlock_forecast.py # simulation + Q-style estimator
README.md # documentation
timeline.png # generated results plot


---

## 🧠 What Makes This "Quantum-Inspired"?

We **do not claim a laptop runs quantum circuits**.

Instead we **simulate the logic** behind Shor's algorithm’s period-finding:

- repeated system states → cyclic structure  
- modular encoding of states  
- phase-like samples → continued fractions  
- integer period extraction (r̂)

This explores **how quantum structure can aid predictive analysis** in concurrent systems.

---

## 📊 Results

Outputs:

- red ❌ marks = real deadlocks
- shaded bands = predicted future deadlock windows
- final accuracy metrics (precision/recall)
- `timeline.png` visualizing system evolution + forecast

---

## 🛠 Future Extensions

- Qiskit version using a real QFT + phase estimation (small N)
- Apply to Kubernetes trace logs
- FFT/autocorrelation comparison baseline
- RL agent that avoids deadlock windows

---

## ⚠️ Disclaimer

This is a **research demo**, not a production fault-prediction engine.  
(Though the idea is genuinely promising — especially for heavy-periodicity workloads.)

No real servers were harmed in the simulation.  
Simulated processes, however, fought bravely over imaginary mutexes 🧵🔒

---

## 👤 Author

Built by Shree (AI Engineer, footballer, quantum adventurer)  
Let’s just say: this time, **deadlocks had it coming**.

---

## ⭐ If you liked this project

Star the repo, share it, or tell your SRE team:

> “The deadlocks are coming — I have quantum feelings about it.”