from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_DAYS, DEFAULT_SEED, DEFAULT_USERS, SIMULATION_END

CHANNELS = ("organic", "paid_social", "search", "referral", "influencer")
COUNTRIES = ("US", "GB", "DE", "FR", "IT", "ES", "CA", "AU")
DEVICES = ("ios", "android", "web")
MONTHLY_PRICES = {"US": 9.99, "GB": 8.99, "DE": 8.99, "FR": 8.99, "IT": 7.99, "ES": 7.99, "CA": 9.99, "AU": 12.99}
CHANNEL_CONVERSION = {"organic": 0.43, "paid_social": 0.35, "search": 0.48, "referral": 0.50, "influencer": 0.38}
CHANNEL_CAC = {"organic": 0.22, "paid_social": 3.8, "search": 4.4, "referral": 0.8, "influencer": 2.6}


def generate_dataset(
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    n_users: int = DEFAULT_USERS,
    days: int = DEFAULT_DAYS,
    treatment_lift: float = 0.03,
) -> dict[str, Path]:
    """Generate reproducible user, event, subscription, spend and A/B data."""
    if n_users < 1:
        raise ValueError("n_users must be at least 1")
    if days < 30:
        raise ValueError("days must be at least 30")
    if not 0 <= treatment_lift <= 1:
        raise ValueError("treatment_lift must be between 0 and 1")

    rng = np.random.default_rng(seed)
    end_date = pd.Timestamp(SIMULATION_END)
    first_date = end_date - pd.Timedelta(days - 1, unit="D")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_ids = np.array([f"u_{i:07d}" for i in range(1, n_users + 1)])
    signup_offsets = rng.integers(0, days, size=n_users)
    signup_dates = first_date + pd.to_timedelta(signup_offsets, unit="D")
    channels = rng.choice(CHANNELS, size=n_users, p=[0.34, 0.27, 0.18, 0.13, 0.08])
    countries = rng.choice(COUNTRIES, size=n_users, p=[0.32, 0.12, 0.13, 0.11, 0.10, 0.08, 0.08, 0.06])
    devices = rng.choice(DEVICES, size=n_users, p=[0.48, 0.41, 0.11])
    users = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": signup_dates.date.astype(str),
        "acquisition_channel": channels,
        "country": countries,
        "device": devices,
    }).sort_values("user_id")

    assignments = rng.choice(["control", "treatment"], size=n_users)
    experiment = pd.DataFrame({
        "assignment_id": [f"a_{i:07d}" for i in range(1, n_users + 1)],
        "user_id": user_ids,
        "variant": assignments,
        "assigned_at": signup_dates.astype(str),
        "experiment_name": "onboarding_v2",
    }).sort_values("assignment_id")

    event_rows: list[dict[str, object]] = []
    subscription_rows: list[dict[str, object]] = []
    event_counter = 0
    for i, user_id in enumerate(user_ids):
        signup = pd.Timestamp(signup_dates[i])
        active_days = days - int(signup_offsets[i])
        age = np.arange(active_days)
        daily_lambda = 0.70 * np.exp(-age / 90.0) + 0.12
        active = rng.random(active_days) < np.minimum(daily_lambda, 0.85)
        for offset in np.flatnonzero(active):
            event_date = signup + pd.Timedelta(int(offset), unit="D")
            n_sessions = int(rng.choice([1, 2, 3], p=[0.79, 0.18, 0.03]))
            for _ in range(n_sessions):
                session_minute = int(rng.integers(0, 24 * 60))
                ts = event_date + pd.Timedelta(session_minute, unit="m")
                for event_type in ("session_start", "content_view"):
                    event_counter += 1
                    event_rows.append({
                        "event_id": f"e_{event_counter:09d}",
                        "user_id": user_id,
                        "event_ts": ts.isoformat(),
                        "event_type": event_type,
                        "feature_name": "",
                    })
                if rng.random() < 0.28:
                    event_counter += 1
                    event_rows.append({
                        "event_id": f"e_{event_counter:09d}",
                        "user_id": user_id,
                        "event_ts": (ts + pd.Timedelta(30, unit="s")).isoformat(),
                        "event_type": "feature_exposure",
                        "feature_name": "onboarding_v2" if assignments[i] == "treatment" else "onboarding_v1",
                    })

        # 72% try a 7-day trial; remaining users buy immediately.
        trial = bool(rng.random() < 0.72)
        trial_start = signup if trial else pd.NaT
        trial_end = signup + pd.Timedelta(7, unit="D") if trial else pd.NaT
        base_rate = CHANNEL_CONVERSION[str(channels[i])]
        conversion_probability = min(1.0, base_rate + (treatment_lift if assignments[i] == "treatment" else 0.0))
        converted = bool(rng.random() < conversion_probability) if trial else True
        paid_start = signup + pd.Timedelta(7, unit="D") if trial and converted else signup if not trial else pd.NaT
        if pd.notna(paid_start) and paid_start > end_date:
            paid_start = pd.NaT
        canceled_at = pd.NaT
        status = "trial" if trial and not converted else "active"
        if pd.notna(paid_start):
            # Cancellation hazard rises for lower engagement and paid social users.
            cancel_probability = 0.16 + (0.10 if channels[i] == "paid_social" else 0.0)
            if rng.random() < cancel_probability:
                cancel_offset = int(rng.integers(25, max(26, min(150, (end_date - paid_start).days + 1))))
                cancel_date = paid_start + pd.Timedelta(cancel_offset, unit="D")
                if cancel_date <= end_date:
                    canceled_at = cancel_date
                    status = "canceled"
        subscription_rows.append({
            "subscription_id": f"s_{i + 1:07d}",
            "user_id": user_id,
            "plan": "monthly",
            "trial_start_at": trial_start.isoformat() if pd.notna(trial_start) else "",
            "trial_end_at": trial_end.isoformat() if pd.notna(trial_end) else "",
            "paid_start_at": paid_start.isoformat() if pd.notna(paid_start) else "",
            "canceled_at": canceled_at.isoformat() if pd.notna(canceled_at) else "",
            "status": status,
            "monthly_price": MONTHLY_PRICES[str(countries[i])],
        })

    events = pd.DataFrame(event_rows, columns=["event_id", "user_id", "event_ts", "event_type", "feature_name"])
    if not events.empty:
        events = events.sort_values(["event_ts", "event_id"])
    subscriptions = pd.DataFrame(subscription_rows).sort_values("subscription_id")

    spend_rows: list[dict[str, object]] = []
    for date in pd.date_range(first_date, end_date, freq="D"):
        for channel in CHANNELS:
            acquired = int(((signup_dates.date == date.date()) & (channels == channel)).sum())
            spend_rows.append({
                "spend_date": date.date().isoformat(),
                "acquisition_channel": channel,
                "new_users": acquired,
                "spend": round(acquired * CHANNEL_CAC[channel], 2),
            })
    acquisition = pd.DataFrame(spend_rows)

    tables = {
        "users": users,
        "events": events,
        "subscriptions": subscriptions,
        "acquisition": acquisition,
        "experiment_assignments": experiment,
    }
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        paths[name] = path
    return paths
