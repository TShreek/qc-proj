#!/usr/bin/env python3
# Deadlock Forecast via Quantum Period Analysis
# Simulates a system with processes and resources, detects deadlocks classically,
# recovers hidden periodic contention via a Shor-style period estimator (PE + CF),
# and forecasts the next deadlock windows.

import argparse
import random
import math
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from math import gcd
import time

import matplotlib.pyplot as plt


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)

def lcm(a, b):
    return abs(a*b) // gcd(a, b) if a and b else 0

def lcm_many(vals):
    return reduce(lcm, vals, 1)

def rolling_mean(xs, W):
    q, s, out = deque(), 0, []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > W:
            s -= q.popleft()
        out.append(s / len(q))
    return out


# =========================
# Simulation: Processes & Resources
# =========================

@dataclass
class Proc:
    pid: int
    holding: set
    waiting_for: int | None = None

@dataclass
class Res:
    rid: int
    held_by: int | None = None
    queue: deque = field(default_factory=deque)

class SystemSim:
    """
    Discrete-time simulation with periodic contention bursts.
    At each tick:
      - Some processes request resources (with bursty pattern).
      - Some processes release resources.
      - We build RAG/WFG and can detect deadlocks.
    """

    def __init__(
        self,
        n_procs=12,
        n_res=8,
        burst_period=24,
        burst_width=4,
        burst_strength=0.8,
        base_request_prob=0.12,
        release_prob=0.08,
        max_steps=300,
        seed=42
    ):
        set_seed(seed)
        self.P = n_procs
        self.R = n_res
        self.t = 0
        self.max_steps = max_steps

        self.processes = {p: Proc(pid=p, holding=set()) for p in range(self.P)}
        self.resources = {r: Res(rid=r, queue=deque()) for r in range(self.R)}

        self.burst_period = burst_period
        self.burst_width = burst_width
        self.burst_strength = burst_strength
        self.base_request_prob = base_request_prob
        self.release_prob = release_prob

        # logs
        self.deadlocks = []
        self.wait_edges_snapshots = []  # list of list[(u->v)] wait-for edges
        self.rag_edges_snapshots = []   # list of list[(u->res)] and (res->u)
        self.deadlock_flags = []

    # ----- contention schedule -----
    def burst_multiplier(self, t):
        # Triangular window over mod period
        phase = (t % self.burst_period) / self.burst_period
        # distance from center (0.0)
        d = min(abs(phase - 0.0), 1 - abs(phase - 0.0))
        # window width -> convert to fraction of period
        w = self.burst_width / self.burst_period
        bump = max(0.0, 1.0 - (d / max(w, 1e-6)))
        return 1.0 + self.burst_strength * bump  # in [1, 1+strength]

    # ----- step dynamics -----
    def step(self):
        t = self.t
        # 1) Releases
        for p in range(self.P):
            if self.processes[p].holding and random.random() < self.release_prob:
                rid = random.choice(list(self.processes[p].holding))
                self.release(p, rid)

        # 2) Requests with periodic bursts
        req_prob = min(1.0, self.base_request_prob * self.burst_multiplier(t))
        for p in range(self.P):
            if self.processes[p].waiting_for is None and random.random() < req_prob:
                rid = random.randrange(self.R)
                if rid not in self.processes[p].holding:
                    self.request(p, rid)

        # 3) Assign resources if free and queued
        for r in range(self.R):
            res = self.resources[r]
            if res.held_by is None and res.queue:
                pid = res.queue.popleft()
                self.grant(pid, r)

        # 4) Graph snapshots & deadlock detection
        wfg = self.wait_for_graph()
        rag = self.resource_allocation_graph()

        sccs = find_sccs(wfg)
        deadlocked_components = [
            comp for comp in sccs
            if len(comp) > 1 or any(u in wfg.get(u, ()) for u in comp)  # self-loop
        ]
        deadlocked = len(deadlocked_components) > 0
        if deadlocked:
            self.deadlocks.append(t)

        self.deadlock_flags.append(1 if deadlocked else 0)
        self.wait_edges_snapshots.append(edges_from_adj(wfg))
        self.rag_edges_snapshots.append(edges_from_rag(rag))

        self.t += 1

    def run(self):
        for _ in range(self.max_steps):
            self.step()

    # ----- resource ops -----
    def request(self, pid, rid):
        p = self.processes[pid]
        r = self.resources[rid]
        # If free, grant; else enqueue and mark waiting
        if r.held_by is None:
            self.grant(pid, rid)
        else:
            r.queue.append(pid)
            p.waiting_for = rid

    def grant(self, pid, rid):
        p = self.processes[pid]
        r = self.resources[rid]
        r.held_by = pid
        p.holding.add(rid)
        if p.waiting_for == rid:
            p.waiting_for = None

    def release(self, pid, rid):
        p = self.processes[pid]
        r = self.resources[rid]
        if rid in p.holding:
            p.holding.remove(rid)
            if r.held_by == pid:
                r.held_by = None

    # ----- graphs -----
    def resource_allocation_graph(self):
        # Bipartite: P -> R (request edges), R -> P (assignment edges)
        P2R = defaultdict(set)
        R2P = defaultdict(set)
        for pid, p in self.processes.items():
            if p.waiting_for is not None:
                P2R[pid].add(p.waiting_for)
            for rid in p.holding:
                R2P[rid].add(pid)
        return P2R, R2P

    def wait_for_graph(self):
        # Contract resources: edge P->Q if P waits for R held by Q
        adj = defaultdict(set)
        for pid, p in self.processes.items():
            if p.waiting_for is not None:
                rid = p.waiting_for
                holder = self.resources[rid].held_by
                if holder is not None and holder != pid:
                    adj[pid].add(holder)
        # Ensure all processes appear as nodes
        for pid in range(self.P):
            adj[pid] = adj[pid]  # touch
        return adj


# =========================
# Classical deadlock detection helpers
# =========================

def find_sccs(adj):
    """
    Tarjan's strongly connected components.
    adj: dict[u] -> set(v)
    returns: list of lists (components)
    """
    index = {}
    lowlink = {}
    stack = []
    onstack = set()
    idx = 0
    comps = []

    def strongconnect(v):
        nonlocal idx
        index[v] = idx
        lowlink[v] = idx
        idx += 1
        stack.append(v)
        onstack.add(v)
        for w in adj.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                u = stack.pop()
                onstack.remove(u)
                comp.append(u)
                if u == v:
                    break
            comps.append(comp)

    for v in adj.keys():
        if v not in index:
            strongconnect(v)
    return comps

def edges_from_adj(adj):
    edges = []
    for u, nbrs in adj.items():
        for v in nbrs:
            edges.append((u, v))
    edges.sort()
    return edges

def edges_from_rag(rag):
    P2R, R2P = rag
    edges = []
    for p, rs in P2R.items():
        for r in rs: edges.append((f"P{p}", f"R{r}"))
    for r, ps in R2P.items():
        for p in ps: edges.append((f"R{r}", f"P{p}"))
    edges.sort()
    return edges


# =========================
# State encoding for period analysis
# =========================

def fnv1a64_hash_pairs(pairs):
    # 64-bit FNV-1a over (u,v) int pairs
    h = 1469598103934665603
    for (u, v) in pairs:
        # pack two ints into a 64-bit-ish stream
        x = (u * 1315423911 + v * 2654435761) & 0xFFFFFFFFFFFFFFFF
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def encode_state_sequence(wait_edge_snapshots, a=5, N=(1 << 31) - 1):
    """
    Map each snapshot to y_t = a^(h_t) mod N with h_t from a stable hash of wait edges.
    If contention patterns are periodic with period r, y_t tends to repeat every r.
    """
    if gcd(a, N) != 1:
        raise ValueError("Choose a,N coprime")
    ys = []
    for edges in wait_edge_snapshots:
        h = fnv1a64_hash_pairs(edges)
        # reduce exponent to [0, N-2] by Fermat for prime-ish N
        e = h % (N - 1)
        ys.append(pow(a, e, N))
    return ys


# =========================
# "Quantum" Period Estimator (Shor-style, simulated)
# =========================

def _classical_order_of_a_mod_N(a, N, cap=1_000_000):
    """Small helper to get order (used only to generate ideal phase samples in this demo)."""
    if gcd(a, N) != 1:
        return None
    x = 1
    for r in range(1, cap + 1):
        x = (x * a) % N
        if x == 1:
            return r
    return None

def quantum_period_estimate_from_series(y, shots=10, events=None):
    """
    Robust Shor-style period estimator.
    1) If deadlock events provided, estimate period from event gaps (GCD/mode + autocorr).
    2) Else, fall back to hashed-series methods.
    3) Refine candidate r via continued-fractions-style denominator LCM and divisors.
    """
    # --- 1) Event-driven estimator (most reliable) ---
    if events and len(events) >= 3:
        diffs = [events[i+1] - events[i] for i in range(len(events)-1)]
        diffs = [d for d in diffs if 2 <= d <= 128]  # ignore tiny/huge noise gaps
        r_gcd = None
        if diffs:
            g = diffs[0]
            for d in diffs[1:]:
                g = math.gcd(g, d)
            r_gcd = g if g >= 2 else None

        r_mode = None
        if diffs:
            r_mode = Counter(diffs).most_common(1)[0][0]

        # small autocorr on the binary event indicator to support the guess
        r_ac = None
        if y and len(y) >= 16:
            # build a simple binary series of events over len(y)
            T = len(y)
            flags = [0]*T
            for t in events:
                if 0 <= t < T:
                    flags[t] = 1
            # score lags
            best_s, best_L = 0, None
            for L in range(2, min(T//2, 128)):
                s = sum(1 for i in range(T-L) if flags[i] and flags[i+L])
                if s > best_s:
                    best_s, best_L = s, L
            r_ac = best_L

        # pick a candidate
        candidates = [r for r in [r_mode, r_gcd, r_ac] if r and r >= 2]
        if candidates:
            candidate = min(candidates, key=lambda r: abs(r - (r_mode or r)))  # bias toward the mode if present

            # --- 2) "Phase samples" refinement (continued fractions vibe) ---
            # build synthetic fractions near k/r to refine (like PE output)
            fracs = []
            for _ in range(shots):
                k = random.randrange(1, candidate)
                fracs.append(Fraction(k, candidate))  # idealized samples

            # aggregate denominators
            denom_counts = Counter(f.denominator for f in fracs if f.denominator)
            top_denoms = [d for d, _ in denom_counts.most_common(3)]
            r_try = max(2, min(lcm_many(top_denoms), 512))

            # shrink to best divisor that explains many event gaps
            def explains(r):
                if r < 2: return -1
                return sum(1 for d in diffs if d % r == 0)

            best, best_score = candidate, explains(candidate)
            for r_d in divisors(r_try):
                sc = explains(r_d)
                if sc > best_score or (sc == best_score and r_d < best):
                    best, best_score = r_d, sc
            # sanity: prefer close to mode if tie
            return best

    # --- 3) Fallback: hashed-series heuristic (old path) ---
    n = len(y) if y else 0
    if n < 8:
        return None

    idxs = defaultdict(list)
    for i, val in enumerate(y):
        idxs[val].append(i)
    deltas = []
    for idx_list in idxs.values():
        if len(idx_list) >= 2:
            for i in range(1, len(idx_list)):
                d = idx_list[i] - idx_list[i-1]
                if d > 0:
                    deltas.append(d)

    if not deltas:
        scores = []
        for L in range(2, min(n//2, 64)):
            matches = sum(1 for i in range(n - L) if y[i] == y[i + L])
            scores.append((matches, L))
        scores.sort(reverse=True)
        deltas = [L for _, L in scores[:shots]]

    if not deltas:
        return None

    T0 = max(2, int(median(deltas)))
    fracs = []
    for _ in range(shots):
        d = random.choice(deltas)
        x = (d / T0) % 1.0
        fracs.append(Fraction(x).limit_denominator(64))

    denom_counts = Counter(f.denominator for f in fracs if f.denominator > 0)
    top_denoms = [d for d, _ in denom_counts.most_common(3)]
    candidate = max(2, min(lcm_many(top_denoms), 256))

    def explains(r):
        return sum(1 for d in deltas if d % r == 0)

    best, best_score = candidate, explains(candidate)
    for r_try in sorted(divisors(candidate)):
        sc = explains(r_try)
        if sc >= best_score:
            best, best_score = r_try, sc
    return best

def divisors(n):
    res = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            res.add(i)
            res.add(n // i)
    return sorted(res)

def median(xs):
    s = sorted(xs)
    m = len(s)
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


# =========================
# Forecast & Evaluation
# =========================

def forecast_windows(last_deadlock_time, r_est, horizon, width=2):
    """
    Produce inclusive windows [t-k, t+k] centered at last_deadlock_time + m * r_est.
    """
    if last_deadlock_time is None or r_est is None or r_est <= 1:
        return []
    windows = []
    t0 = last_deadlock_time
    m = 1
    while True:
        center = t0 + m * r_est
        if center - width > horizon:
            break
        windows.append((max(0, int(center - width)), int(center + width)))
        m += 1
    return windows

def evaluate_forecast(deadlock_times, windows):
    """
    Precision/recall: a deadlock 'hit' if it falls inside any window.
    """
    if not windows:
        return dict(precision=0.0, recall=0.0, hits=0, misses=len(deadlock_times), fp=len(windows))
    hits = 0
    used = set()
    for t in deadlock_times:
        for i, (a, b) in enumerate(windows):
            if a <= t <= b and i not in used:
                hits += 1
                used.add(i)
                break
    precision = hits / max(len(windows), 1)
    recall = hits / max(len(deadlock_times), 1)
    return dict(precision=precision, recall=recall, hits=hits,
                misses=len(deadlock_times) - hits, fp=len(windows) - hits)

def plot_timeline(deadlock_flags, deadlock_times, windows, r_est, out="timeline.png"):
    T = len(deadlock_flags)
    t = list(range(T))
    y = rolling_mean(deadlock_flags, 1)

    fig = plt.figure(figsize=(12, 4))
    plt.plot(t, y, label="Deadlock indicator", linewidth=1.5)
    for a, b in windows:
        plt.axvspan(a, b, alpha=0.15, label="Forecast window" if a == windows[0][0] else None)
    plt.scatter(deadlock_times, [1.05] * len(deadlock_times), marker="x", s=60, label="Deadlock", zorder=5)
    if r_est:
        plt.title(f"Deadlock Forecast via Quantum Period Analysis (r̂ = {r_est})")
    else:
        plt.title(f"Deadlock Forecast via Quantum Period Analysis")
    plt.ylim(-0.05, 1.2)
    plt.xlabel("Time")
    plt.ylabel("Deadlock (0/1)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📈 Timeline saved to {out}")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Deadlock Forecast via Quantum Period Analysis")
    parser.add_argument("--procs", type=int, default=12)
    parser.add_argument("--res", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--period", type=int, default=24, help="Injected contention period")
    parser.add_argument("--width", type=int, default=4, help="Burst width within period (ticks)")
    parser.add_argument("--strength", type=float, default=0.8, help="Burst strength multiplier")
    parser.add_argument("--base_req", type=float, default=0.12)
    parser.add_argument("--rel", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=2, help="Forecast +/- window size (ticks)")
    parser.add_argument("--out", type=str, default="timeline.png")
    args = parser.parse_args()

    sim = SystemSim(
        n_procs=args.procs,
        n_res=args.res,
        burst_period=args.period,
        burst_width=args.width,
        burst_strength=args.strength,
        base_request_prob=args.base_req,
        release_prob=args.rel,
        max_steps=args.steps,
        seed=args.seed
    )

    print("▶️ Running simulation...")
    sim.run()
    deadlocks = sim.deadlocks
    print(f"✔️ Steps={args.steps}, deadlocks={len(deadlocks)} at times={deadlocks}")

    # Encode evolving contention state
    y = encode_state_sequence(sim.wait_edges_snapshots, a=5, N=(1 << 31) - 1)

    # "Quantum" period estimate from the state series
    r_est = quantum_period_estimate_from_series(y, shots=10, events=deadlocks)
    print(f"🔭 Estimated period r̂ = {r_est} (true injected period = {args.period})")

    # Build forecast windows after the last observed deadlock
    last_deadlock = deadlocks[-1] if deadlocks else None
    windows = forecast_windows(last_deadlock, r_est, horizon=args.steps, width=args.window)

    # Evaluate
    metrics = evaluate_forecast(deadlocks, windows)
    print(f"📊 Forecast metrics: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, "
          f"hits={metrics['hits']}, misses={metrics['misses']}, false_positives={metrics['fp']}")

    # Plot
    plot_timeline(sim.deadlock_flags, deadlocks, windows, r_est, out=args.out)

if __name__ == "__main__":
    main()
