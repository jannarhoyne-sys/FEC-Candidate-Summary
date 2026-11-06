from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent

# Load App
model = joblib.load(BASE / "Models" / "vote_share_model.pkl")
with open(BASE / "Models" / "defaults.json") as f:
    defaults = json.load(f)
FEATURES = defaults["features"]
races = pd.read_csv(BASE / "Data" / "model_data_2024.csv")

# Dropdown menu
race_menu = {}
for key in races["key"].unique():
    state, dist = key.split("-")
    race_menu.setdefault(state, []).append((int(dist), key))
race_menu = {st: [k for _, k in sorted(v)] for st, v in sorted(race_menu.items())}

PARTY_CLASS = {"DEM": "dem", "REP": "rep", "OTHER": "oth"}
MAX_CANDS = 4                          # rows in the hypothetical form

app = Flask(__name__)


# format helpers
def fmt_money(dollars):
    if dollars >= 1_000_000:
        return f"${dollars / 1_000_000:.1f}M"
    if dollars >= 1_000:
        return f"${dollars / 1_000:.0f}K"
    return f"${dollars:,.0f}"


def clean_name(name):
    return str(name).replace('\\"', '"').title()


def rating(margin, party):
    if margin < 5:
        return "TOO CLOSE TO CALL", "tossup"
    level = "LEANS" if margin < 15 else "SAFE"
    return f"{level} {party}", f"{level.lower()}-{PARTY_CLASS[party]}"


# prediction
def predict_race(rows):
    raw = model.predict(rows[FEATURES])
    pred = raw / raw.sum() * 100      # scale so the race adds up to 100%

    cands = []
    for (_, r), p in zip(rows.iterrows(), pred):
        cands.append({
            "name": clean_name(r["candidate"]),
            "party": r["party_group"],
            "cls": PARTY_CLASS[r["party_group"]],
            "status": r["status"].title(),
            "raised": fmt_money(10 ** r["log_receipts"] - 1),
            "money_share": r["money_share"] * 100,
            "pred": p,
            "actual": r["vote_share"] * 100 if "vote_share" in r else None,
        })
    cands.sort(key=lambda c: c["pred"], reverse=True)

    leader, runner_up = cands[0], cands[1]
    margin = leader["pred"] - runner_up["pred"]
    label, chip = rating(margin, leader["party"])

    # Order: leader on the left, runner-up on the right, everyone else in the middle
    bar = [leader] + cands[2:] + [runner_up]

    result = {"cands": cands, "leader": leader, "runner_up": runner_up, "bar": bar,
              "margin": margin, "label": label, "chip": chip, "close": margin < 5}

    # Flag races with two+ candidates from the same major party (top-two / open primaries).
    # The model is less reliable there (see "Version 2" in the notebook).
    majors = rows.loc[rows["party_group"].isin(["DEM", "REP"]), "party_group"]
    result["same_party"] = bool(majors.duplicated().any())

    if cands[0]["actual"] is not None:
        actual_winner = max(cands, key=lambda c: c["actual"])
        actual_winner["won"] = True
        result["called_right"] = actual_winner is leader
    return result


def typical(party, status):
    return defaults["by_group"].get(f"{party}|{status}", defaults["overall"])


def build_race(state, cands):
    df = pd.DataFrame(cands)
    r = df["raised"].astype(float).clip(lower=0)
    n, total = len(df), r.sum()

    df["candidate"] = df["name"]
    df["state_po"] = state
    df["party_group"] = df["party"]
    df["n_candidates"] = n
    df["money_share"] = r / total if total > 0 else 1 / n
    df["opp_money"] = total - r

    top, second = r.max(), r.nlargest(2).iloc[-1]
    top_opp = np.where(r == top, second, top)             # biggest opponent's money
    df["log_receipts"] = np.log10(1 + r)
    df["log_opp_money"] = np.log10(1 + df["opp_money"])
    df["money_gap"] = df["log_receipts"] - np.log10(1 + top_opp)

    def given(c, key):
        v = c.get(key)
        return None if v is None or pd.isna(v) else float(v)

    for col in ["small_donor_pct", "pac_pct", "self_fund_pct", "party_money_pct", "log_cash_start"]:
        df[col] = np.nan
    for i, c in df.iterrows():
        t = typical(c["party"], c["status"])
        for col, key in [("small_donor_pct", "small"), ("pac_pct", "pac"), ("self_fund_pct", "self")]:
            v = given(c, key)
            df.at[i, col] = v / 100 if v is not None else t[col]
        df.at[i, "party_money_pct"] = t["party_money_pct"]
        cash = given(c, "cash")
        df.at[i, "log_cash_start"] = np.log10(1 + cash) if cash is not None else t["log_cash_start"]
    return df


# hypothetical form
def to_number(text):
    s = str(text or "").replace(",", "").replace("$", "").replace("%", "").strip().upper()
    if s == "":
        return None
    mult = 1
    if s.endswith("M"):
        s, mult = s[:-1], 1_000_000
    elif s.endswith("K"):
        s, mult = s[:-1], 1_000
    return float(s) * mult


def blank_rows():
    rows = [
        {"name": "Candidate A", "party": "DEM", "status": "INCUMBENT", "raised": "2M"},
        {"name": "Candidate B", "party": "REP", "status": "CHALLENGER", "raised": "500K"},
    ]
    rows += [{"name": "", "party": "OTHER", "status": "CHALLENGER", "raised": ""}
             for _ in range(MAX_CANDS - len(rows))]
    return rows


def rows_from_race(key):
    real = races[races["key"] == key].sort_values("log_receipts", ascending=False).head(MAX_CANDS)
    rows = []
    for _, c in real.iterrows():
        rows.append({
            "name": clean_name(c["candidate"]), "party": c["party_group"], "status": c["status"],
            "raised": f"{10 ** c['log_receipts'] - 1:,.0f}",
            "small": f"{c['small_donor_pct'] * 100:.0f}", "pac": f"{c['pac_pct'] * 100:.0f}",
            "self": f"{c['self_fund_pct'] * 100:.0f}", "cash": f"{10 ** c['log_cash_start'] - 1:,.0f}",
        })
    rows += [{"name": "", "party": "OTHER", "status": "CHALLENGER", "raised": ""}
             for _ in range(MAX_CANDS - len(rows))]
    return key.split("-")[0], rows


def rows_from_form(form):
    fields = ["name", "party", "status", "raised", "small", "pac", "self", "cash"]
    rows = [{f: form.get(f"c{i}_{f}", "") for f in fields} for i in range(MAX_CANDS)]
    for row in rows:
        row["party"] = row["party"] or "OTHER"
        row["status"] = row["status"] or "CHALLENGER"
    return rows


def parse_rows(rows):
    cands, errors = [], []
    for i, row in enumerate(rows):
        if not row.get("name", "").strip() and not str(row.get("raised", "")).strip():
            continue
        label = row.get("name", "").strip() or f"Candidate {i + 1}"
        try:
            c = {"name": label, "party": row.get("party"), "status": row.get("status"),
                 "raised": to_number(row.get("raised")) or 0,
                 "small": to_number(row.get("small")), "pac": to_number(row.get("pac")),
                 "self": to_number(row.get("self")), "cash": to_number(row.get("cash"))}
        except ValueError:
            errors.append(f"{label}: use numbers like 250000, 250K or 1.5M.")
            continue
        if c["party"] not in defaults["party_groups"] or c["status"] not in defaults["statuses"]:
            errors.append(f"{label}: pick a party and a status.")
        if c["raised"] < 0:
            errors.append(f"{label}: amount raised can't be negative.")
        for key in ["small", "pac", "self"]:
            if c[key] is not None and not 0 <= c[key] <= 100:
                errors.append(f"{label}: percentages must be between 0 and 100.")
                break
        cands.append(c)
    if len(cands) < 2 and not errors:
        errors.append("Enter at least two candidates.")
    return cands, errors


# pages
@app.route("/")
def lookup():
    key = request.args.get("race", "")
    if key not in races["key"].values:
        key = next(iter(race_menu.values()))[0]          # default - first race in the menu
    result = predict_race(races[races["key"] == key])
    return render_template("index.html", mode="lookup", race_menu=race_menu,
                           key=key, title=key, r=result)


@app.route("/hypothetical", methods=["GET", "POST"])
def hypothetical():
    state, rows, result, errors, source = "CA", blank_rows(), None, [], None

    if request.method == "POST":                          # predict was clicked
        state = request.form.get("state", "CA")
        rows = rows_from_form(request.form)
    elif request.args.get("from") in races["key"].values:  # "Edit as a what-if" link
        source = request.args["from"]
        state, rows = rows_from_race(source)

    if request.method == "POST" or source:
        cands, errors = parse_rows(rows)
        if state not in defaults["states"]:
            errors.append("Pick a state.")
        if not errors:
            result = predict_race(build_race(state, cands))

    title = f"What-if: {source}" if source else f"Hypothetical race ({state})"
    return render_template("index.html", mode="hypo", states=defaults["states"],
                           state=state, rows=rows, errors=errors, title=title, r=result,
                           parties=defaults["party_groups"], statuses=defaults["statuses"])


# machine-to-machine routes
@app.route("/ping")
def ping():
    return jsonify(status="ok")


@app.route("/invocations", methods=["POST"])
def invocations():
    data = request.get_json(force=True, silent=True) or {}
    rows = [{k: str(v) for k, v in c.items()} for c in data.get("candidates", [])]
    cands, errors = parse_rows(rows)
    state = data.get("state", "CA")
    if state not in defaults["states"]:
        errors.append(f"Unknown state: {state}")
    if errors:
        return jsonify(errors=errors), 400
    result = predict_race(build_race(state, cands))
    return jsonify(state=state, rating=result["label"], predicted_margin=round(result["margin"], 1),
                   predictions=[{"name": c["name"], "party": c["party"],
                                 "predicted_vote_share": round(c["pred"], 1)} for c in result["cands"]])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))   # 8080 locally
    app.run(host="0.0.0.0", port=port, debug=True)