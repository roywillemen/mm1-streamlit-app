"""
Streamlit app for simulating a simple M/M/1 queue using SimPy.
Discussed in Operations Research 6 / Week 7's class (2025-2026)
Fontys University of Applied Sciences, Eindhoven, The Netherlands
"""

# if necessary, install missing packages
# pip install streamlit simpy pandas matplotlib openpyxl


import io
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import pandas as pd
import simpy
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =========================
# Model / Monitor
# =========================

@dataclass
class Params:
    mean_interarrival: float  # minutes
    mean_service: float       # minutes
    sim_time: float           # minutes
    warmup: float             # minutes
    sample_dt: float          # minutes


@dataclass
class CustRecord:
    cid: int
    arrival: float
    start: float
    service: float
    depart: float

    @property
    def waitq(self) -> float:
        return self.start - self.arrival

    @property
    def sojourn(self) -> float:
        return self.depart - self.arrival


class Monitor:
    def __init__(self):
        self.customers: List[CustRecord] = []
        self.q_trace: List[Tuple[float, int]] = []  # (time, N(t)=queue+in service)

    def customers_df(self) -> pd.DataFrame:
        rows = []
        prev_arrival = None
        for r in self.customers:
            ia = (r.arrival - prev_arrival) if prev_arrival is not None else r.arrival
            prev_arrival = r.arrival
            rows.append({
                "Customer": r.cid,
                "Interarrival": ia,
                "Arrival": r.arrival,
                "ServiceStart": r.start,
                "WaitQ": r.waitq,
                "Service": r.service,
                "SojournW": r.sojourn,
                "Departure": r.depart,
            })
        return pd.DataFrame(rows)

    def trace_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.q_trace, columns=["Time", "N_system"])


def expovariate_mean(rng: random.Random, mean: float) -> float:
    # random.expovariate expects rate = 1/mean
    return rng.expovariate(1.0 / mean)


# =========================
# SimPy processes
# =========================

def customer(env: simpy.Environment, cid: int, server: simpy.Resource, p: Params, mon: Monitor, rng: random.Random):
    t_arr = env.now
    with server.request() as req:
        yield req
        t_start = env.now
        s = expovariate_mean(rng, p.mean_service)
        yield env.timeout(s)
        t_dep = env.now

    mon.customers.append(CustRecord(cid=cid, arrival=t_arr, start=t_start, service=s, depart=t_dep))


def arrivals(env: simpy.Environment, server: simpy.Resource, p: Params, mon: Monitor, rng: random.Random):
    cid = 0
    while True:
        ia = expovariate_mean(rng, p.mean_interarrival)
        yield env.timeout(ia)
        cid += 1
        env.process(customer(env, cid, server, p, mon, rng))


def sampler(env: simpy.Environment, server: simpy.Resource, p: Params, mon: Monitor):
    while True:
        n_system = len(server.queue) + server.count
        mon.q_trace.append((env.now, n_system))
        yield env.timeout(p.sample_dt)


# =========================
# Statistics
# =========================

def time_average_L_from_trace(q_trace: List[Tuple[float, int]], warmup: float) -> float:
    q = [(t, n) for (t, n) in q_trace if t >= warmup]
    if len(q) < 2:
        return float("nan")

    area = 0.0
    for (t0, n0), (t1, _) in zip(q[:-1], q[1:]):
        area += n0 * (t1 - t0)

    denom = max(q[-1][0] - q[0][0], 1e-12)
    return area / denom


def summarize_run(mon: Monitor, p: Params) -> Dict[str, float]:
    t0, t1 = p.warmup, p.sim_time
    obs_time = max(t1 - t0, 1e-12)

    # Customers counted for W and λ: departures in observation window
    obs = [r for r in mon.customers if (r.depart >= t0 and r.depart <= t1)]
    n_served_obs = len(obs)

    lam_hat = n_served_obs / obs_time  # throughput rate per minute

    W_hat = sum(r.sojourn for r in obs) / n_served_obs if n_served_obs > 0 else float("nan")
    L_hat = time_average_L_from_trace(mon.q_trace, p.warmup)

    ll_rhs = lam_hat * W_hat if not math.isnan(W_hat) else float("nan")
    ll_delta = (L_hat - ll_rhs) if (not math.isnan(L_hat) and not math.isnan(ll_rhs)) else float("nan")

    return {
        "L_hat": L_hat,
        "lambda_hat_per_min": lam_hat,
        "W_hat": W_hat,
        "Little_delta": ll_delta,
        "served_obs": n_served_obs,
        "obs_time": obs_time,
    }


def run_one_replication(p: Params, seed: int) -> Tuple[Monitor, Dict[str, float]]:
    rng = random.Random(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    mon = Monitor()

    env.process(arrivals(env, server, p, mon, rng))
    env.process(sampler(env, server, p, mon))
    env.run(until=p.sim_time)

    stats = summarize_run(mon, p)
    return mon, stats


def aggregate_stats(stats_list: List[Dict[str, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build a dataframe of run-level stats
    df = pd.DataFrame(stats_list)
    # Provide mean + std for main KPIs
    summary = pd.DataFrame({
        "metric": ["L_hat", "lambda_hat_per_min", "W_hat", "Little_delta"],
        "mean": [
            df["L_hat"].mean(),
            df["lambda_hat_per_min"].mean(),
            df["W_hat"].mean(),
            df["Little_delta"].mean(),
        ],
        "std": [
            df["L_hat"].std(ddof=1) if len(df) > 1 else float("nan"),
            df["lambda_hat_per_min"].std(ddof=1) if len(df) > 1 else float("nan"),
            df["W_hat"].std(ddof=1) if len(df) > 1 else float("nan"),
            df["Little_delta"].std(ddof=1) if len(df) > 1 else float("nan"),
        ],
    })
    return df, summary


def plot_last_trace(mon: Monitor, p: Params):
    fig, ax = plt.subplots(figsize=(6, 3))  # already scaled height

    t = [x for (x, _) in mon.q_trace]
    n = [y for (_, y) in mon.q_trace]

    ax.plot(t, n, linewidth=0.5)
    ax.axvline(p.warmup, linestyle="--")

    # ↓↓↓ ADD smaller font sizes here ↓↓↓
    ax.set_title("Number in system (queue + in service) — last run", fontsize=6)
    ax.set_xlabel("time (min)", fontsize=6)
    ax.set_ylabel("number in system N(t)", fontsize=6)

    ax.tick_params(axis="both", labelsize=6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    return fig


def build_excel_bytes(mon: Monitor, p: Params) -> bytes:
    cust_df = mon.customers_df()
    trace_df = mon.trace_df()

    # Add a small "params" table
    params_df = pd.DataFrame([{
        "mean_interarrival": p.mean_interarrival,
        "mean_service": p.mean_service,
        "sim_time": p.sim_time,
        "warmup": p.warmup,
        "sample_dt": p.sample_dt,
    }])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        params_df.to_excel(writer, sheet_name="params", index=False)
        cust_df.to_excel(writer, sheet_name="customers_last_run", index=False)
        trace_df.to_excel(writer, sheet_name="trace_last_run", index=False)

    return output.getvalue()


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Simple Queue Simulator", layout="wide")

st.title("Simple Queue Simulator")

with st.sidebar:
    st.header("Inputs")

    mean_interarrival = st.number_input("Mean interarrival time (minutes)", min_value=0.001, value=5.0, step=0.1)
    mean_service = st.number_input("Mean service time (minutes)", min_value=0.001, value=4.0, step=0.1)

    n_runs = st.number_input("Number of simulation runs", min_value=1, value=10, step=1)

    st.divider()

    # checkbox to toggle optional settings
    show_optional = st.checkbox("Show optional settings", value=False)

    # Defaults (used when optional settings are hidden)
    sim_time = 24 * 60.0
    warmup = 60.0
    sample_dt = 0.1
    base_seed = 123

    # Only show controls if box is checked
    if show_optional:
        st.subheader("Run settings (optional)")
        sim_time = st.number_input("Simulation time (minutes)", min_value=1.0, value=float(sim_time), step=60.0)
        warmup = st.number_input("Warm-up (minutes)", min_value=0.0, value=float(warmup), step=10.0)
        sample_dt = st.number_input("Sampling interval (minutes)", min_value=0.01, value=float(sample_dt), step=0.01)
        base_seed = st.number_input("Base seed", min_value=0, value=int(base_seed), step=1)

    run_button = st.button("Run simulation", type="primary")


# Basic validation
if warmup >= sim_time:
    st.error("Warm-up must be smaller than simulation time.")
    st.stop()

p = Params(
    mean_interarrival=float(mean_interarrival),
    mean_service=float(mean_service),
    sim_time=float(sim_time),
    warmup=float(warmup),
    sample_dt=float(sample_dt),
)

rho = p.mean_service / p.mean_interarrival  # because lambda=1/IA, mu=1/S => rho=lambda/mu = S/IA
st.caption(f"Implied utilization estimate (theoretical): ρ ≈ mean_service / mean_interarrival = **{rho:.3f}**")

if run_button:
    stats_list: List[Dict[str, float]] = []
    last_mon: Optional[Monitor] = None

    # Run replications
    for r in range(int(n_runs)):
        seed = int(base_seed) + r
        mon, stats = run_one_replication(p, seed)
        stats_list.append(stats)
        last_mon = mon  # keep only the last run monitor for plotting/export

    run_df, summary_df = aggregate_stats(stats_list)

    st.subheader("Results (sample averages across runs)")

    # Show summary table
    st.dataframe(summary_df, use_container_width=True)

    # Interpret Little's Law
    ll_mean = summary_df.loc[summary_df["metric"] == "Little_delta", "mean"].values[0]
    st.write(
        f"**Little’s Law check (mean Δ across runs):** "
        f"Δ = L − λW = **{ll_mean:.4f}** (closer to 0 is better; expect larger deviations for short runs / heavy traffic)."
    )

    # Also show run-level stats if desired
    with st.expander("Run-level stats (each replication)"):
        st.dataframe(run_df, use_container_width=True)

    st.subheader("Last run: number-in-system plot and monitor")

    if last_mon is not None:
        fig = plot_last_trace(last_mon, p)
        st.pyplot(fig, clear_figure=True)

        # Show last run tables
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Last run: customers table (first 200 rows)**")
            st.dataframe(last_mon.customers_df().head(200), use_container_width=True, height=350)
        with col2:
            st.markdown("**Last run: trace table (first 200 rows)**")
            st.dataframe(last_mon.trace_df().head(200), use_container_width=True, height=350)

        # Export button
        excel_bytes = build_excel_bytes(last_mon, p)
        st.download_button(
            label="Download last run monitor as Excel",
            data=excel_bytes,
            file_name="mm1_last_run_monitor.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.info("Set parameters in the sidebar and click **Run simulation**.")
