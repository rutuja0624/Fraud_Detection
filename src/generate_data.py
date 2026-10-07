#!/usr/bin/env python3
"""
Synthetic mobile-money transaction generator (PaySim-compatible schema).

Why synthetic?  Real financial transaction data is confidential.  This generator
produces a labelled dataset with the SAME columns as the public PaySim dataset
(Lopez-Rojas et al., 2016) so every script in this repo also runs unchanged on
the real PaySim CSV (see README, "Using real PaySim data").

Fraud scenario modelled (account-takeover + mule cash-out):
  1. Fraudster drains (or partially drains) a victim account via TRANSFER to a mule.
  2. With prob. 0.6 the mule immediately CASH_OUTs the received money.
Hard cases are injected on purpose so the task is not trivial:
  * ~6% of legitimate TRANSFER / CASH_OUT transactions also drain the whole balance
  * 25% of frauds are partial drains
  * mules also appear as ordinary customers in legitimate traffic
  * ~20% of legitimate transfers go to empty / newly created destination accounts
  * heavy-tailed, overlapping amount distributions for fraud and legitimate traffic
Balances are per-transaction snapshots (not a full ledger simulation) - a
documented simplification.
"""
import argparse, os
import numpy as np
import pandas as pd

HOUR_W_LEGIT = np.array([.4, .3, .25, .25, .3, .5, .9, 1.3, 1.6, 1.7, 1.7, 1.7,
                         1.7, 1.6, 1.6, 1.6, 1.7, 1.8, 1.8, 1.7, 1.5, 1.2, .9, .6])
HOUR_W_FRAUD = np.array([2., 2.2, 2.4, 2.4, 2.2, 1.8, 1.2, .9, .8, .8, .8, .8,
                         .8, .8, .8, .8, .8, .8, .9, .9, 1., 1.2, 1.5, 1.8])


def sample_steps(rng, n, w, n_days):
    day = rng.integers(0, n_days, n)
    hour = rng.choice(24, n, p=w / w.sum())
    return day * 24 + hour


def generate(n_rows=1_000_000, fraud_rate=0.004, seed=42, n_cust=150_000,
             n_merch=15_000, n_mules=1_500, n_agents=3_000, n_days=31):
    rng = np.random.default_rng(seed)
    cust = np.array([f"C{i:09d}" for i in range(n_cust)])
    merch = np.array([f"M{i:08d}" for i in range(n_merch)])
    base = rng.lognormal(9.5, 1.6, n_cust)               # account-level typical balance
    act = rng.gamma(0.8, 1.0, n_cust)
    act /= act.sum()                                      # heterogeneous activity
    mules = rng.choice(n_cust, n_mules, replace=False)
    mw = 1.0 / np.arange(1, n_mules + 1) ** 0.5
    mw /= mw.sum()
    agents = rng.choice(n_cust, n_agents, replace=False)

    def bal(idx):
        return base[idx] * rng.lognormal(0, 0.5, len(idx))

    # ------------------------------------------------------------------ fraud
    n_fraud_target = int(n_rows * fraud_rate)
    n_att = int(n_fraud_target / 1.6)
    cand = rng.integers(0, n_cust, (n_att, 3))
    victims = cand[np.arange(n_att), base[cand].argmax(1)]  # fraudsters prefer richer accounts
    v_old = bal(victims)
    drain = rng.random(n_att) < 0.75
    f_amt = np.where(drain, v_old, v_old * rng.uniform(0.4, 0.95, n_att)).round(2).clip(1)
    f_step = sample_steps(rng, n_att, HOUR_W_FRAUD, n_days - 1)
    m_idx = rng.choice(mules, n_att, p=mw)
    m_old = np.where(rng.random(n_att) < 0.7, 0.0, rng.lognormal(7, 1.2, n_att)).round(2)
    f_tr = pd.DataFrame(dict(
        step=f_step, type="TRANSFER", amount=f_amt,
        nameOrig=cust[victims], oldbalanceOrg=v_old.round(2),
        newbalanceOrig=(v_old - f_amt).clip(0).round(2),
        nameDest=cust[m_idx], oldbalanceDest=m_old,
        newbalanceDest=(m_old + f_amt).round(2), isFraud=1))
    co = rng.random(n_att) < 0.6
    k = int(co.sum())
    mo_old = (m_old[co] + f_amt[co]).round(2)
    full = rng.random(k) < 0.9
    co_amt = np.where(full, mo_old, mo_old * rng.uniform(0.5, 0.95, k)).round(2)
    ag = rng.choice(agents, k)
    ag_old = bal(ag).round(2)
    f_co = pd.DataFrame(dict(
        step=f_step[co] + rng.integers(0, 4, k), type="CASH_OUT",
        amount=co_amt, nameOrig=cust[m_idx[co]], oldbalanceOrg=mo_old,
        newbalanceOrig=(mo_old - co_amt).clip(0).round(2),
        nameDest=cust[ag], oldbalanceDest=ag_old,
        newbalanceDest=(ag_old + co_amt).round(2), isFraud=1))
    fraud = pd.concat([f_tr, f_co], ignore_index=True)
    fraud["step"] = fraud["step"].clip(upper=n_days * 24 - 1)

    # ------------------------------------------------------------- legitimate
    n_leg = n_rows - len(fraud)
    types = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], n_leg,
                       p=[.34, .22, .35, .08, .01])
    step = sample_steps(rng, n_leg, HOUR_W_LEGIT, n_days)
    o = rng.choice(n_cust, n_leg, p=act)
    o_old = bal(o)
    amount = np.empty(n_leg)
    mu = {"PAYMENT": (8.0, 1.0), "CASH_IN": (10.5, 1.2), "CASH_OUT": (10.5, 1.3),
          "TRANSFER": (10.8, 1.4), "DEBIT": (6.5, 0.8)}
    for t, (m_, s_) in mu.items():
        mk = types == t
        amount[mk] = rng.lognormal(m_, s_, int(mk.sum()))
    outflow = types != "CASH_IN"
    amount = np.where(outflow, np.minimum(amount, o_old * rng.uniform(0.05, 1.0, n_leg)), amount)
    legit_drain = outflow & np.isin(types, ["TRANSFER", "CASH_OUT"]) & (rng.random(n_leg) < 0.06)
    amount = np.where(legit_drain, o_old, amount).round(2).clip(1)
    o_new = np.where(outflow, o_old - amount, o_old + amount).clip(0).round(2)

    d_name = np.empty(n_leg, dtype=object)
    d_old = np.zeros(n_leg)
    d_new = np.zeros(n_leg)
    mm = np.isin(types, ["PAYMENT", "DEBIT"])
    d_name[mm] = merch[rng.integers(0, n_merch, int(mm.sum()))]
    ca = types == "CASH_OUT"
    d_idx_ca = rng.choice(agents, int(ca.sum()))
    d_name[ca] = cust[d_idx_ca]
    d_old[ca] = bal(d_idx_ca)
    d_new[ca] = d_old[ca] + amount[ca]
    ci = types == "CASH_IN"
    d_idx_ci = rng.choice(agents, int(ci.sum()))
    d_name[ci] = cust[d_idx_ci]
    d_old[ci] = bal(d_idx_ci)
    d_new[ci] = (d_old[ci] - amount[ci]).clip(0)
    tr = types == "TRANSFER"
    d_idx_tr = rng.choice(n_cust, int(tr.sum()), p=act)
    d_name[tr] = cust[d_idx_tr]
    d_old[tr] = np.where(rng.random(int(tr.sum())) < 0.2, 0.0, bal(d_idx_tr))
    d_new[tr] = d_old[tr] + amount[tr]
    legit = pd.DataFrame(dict(
        step=step, type=types, amount=amount, nameOrig=cust[o],
        oldbalanceOrg=o_old.round(2), newbalanceOrig=o_new, nameDest=d_name,
        oldbalanceDest=d_old.round(2), newbalanceDest=d_new.round(2), isFraud=0))

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sort_values(["step"], kind="stable").reset_index(drop=True)
    df.insert(0, "txnId", np.arange(len(df)))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--fraud-rate", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/transactions.csv")
    ap.add_argument("--sample-out", default="data/sample_10k.csv")
    a = ap.parse_args()
    d = generate(a.rows, a.fraud_rate, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d.to_csv(a.out, index=False)
    d.head(10_000).to_csv(a.sample_out, index=False)
    print(f"rows={len(d):,} fraud={d.isFraud.sum():,} ({d.isFraud.mean():.4%}) -> {a.out}")
    print(d.groupby("type").agg(n=("isFraud", "size"), fraud=("isFraud", "sum")))
