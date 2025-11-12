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

## 🧩 System Workflow

1. **Simulation**: The system simulates multiple processes and resources. Processes request and release resources in bursty, periodic patterns. Resource allocation and wait-for graphs are tracked at each time step.
2. **Deadlock Detection**: At each tick, the system builds a wait-for graph and uses Tarjan's algorithm to detect strongly connected components (SCCs), identifying deadlocks.
3. **State Encoding**: Each snapshot of the wait-for graph is hashed and encoded using modular exponentiation, producing a time series that reflects contention patterns.
4. **Quantum-Inspired Period Estimation**:
    - **Event-Driven**: If deadlock timestamps are available, the algorithm analyzes gaps between events using GCD, mode, and autocorrelation to estimate the period.
    - **Phase Samples & Refinement**: Synthetic phase samples (fractions k/r) are generated and refined using continued fractions and LCM to confirm the period.
    - **Fallback**: If no events, the algorithm analyzes the hashed state sequence for repeating patterns, using the same refinement process.
5. **Forecasting**: Using the estimated period and last deadlock time, the system predicts future deadlock windows and evaluates forecast accuracy (precision/recall).

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

## 🧠 How Does the Quantum-Inspired Algorithm Work?

### Simulation & Deadlock Detection
- Simulates processes and resources with periodic contention bursts.
- Detects deadlocks using Tarjan's SCC algorithm on the wait-for graph.

### State Encoding
- Each system state (wait-for graph) is hashed and encoded as a modular integer, producing a time series.

### Quantum-Inspired Period Estimation
- **Step 1: Event-Driven Estimation**
    - Analyzes deadlock timestamps to find gaps between events.
    - Uses GCD, mode, and autocorrelation to estimate the period.
- **Step 2: Phase Samples & Refinement**
    - Generates synthetic phase samples (fractions k/r) for candidate period r.
    - Uses continued fractions and LCM to refine and confirm the period.
- **Step 3: Fallback to Hashed-Series Heuristic**
    - If no deadlock events, analyzes the hashed state sequence for repeating patterns.
    - Uses the same phase sample and refinement process to estimate the period.

### Forecasting
- Predicts future deadlock windows using the estimated period and last deadlock time.
- Evaluates forecast accuracy (precision, recall, hits, misses).

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