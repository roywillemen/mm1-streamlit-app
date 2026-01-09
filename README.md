# M/M/1 Queue Simulator (Streamlit + SimPy)

This repository contains a Streamlit web app for experimenting with a single-server queue  
(M/M/1: Poisson arrivals, exponential service times) using discrete-event simulation.

The app is intended for teaching purposes in an Operations Research / Queueing Theory course.

---

## What the app does

Users can:
- Set the **mean interarrival time** and **mean service time**
- Choose the **number of simulation runs (replications)**
- Run a **discrete-event simulation** of an M/M/1 queue
- Observe:
  - Time-average number of customers in the system (L)
  - Long-run throughput (λ̂)
  - Mean time in system (W)
  - A check of **Little’s Law** (L ≈ λW)
- View the **number-in-system trajectory** for the last run
- Download simulation data from the last run as an **Excel file**

Optional advanced settings (simulation time, warm-up, etc.) can be shown or hidden in the UI.

---

## How to run locally

Install dependencies:

```bash
pip install -r requirements.txt
