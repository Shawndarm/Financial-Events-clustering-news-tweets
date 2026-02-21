import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_ground_truth_events(price_df, threshold=0.02, gap_tolerance=3):
    """
    Formule (2) du papier : Delta_d = |close(d+7) - close(d)| / close(d)
    """
    df = price_df.copy().sort_values("date")

    # Calcul de la variation à d+7 (on regarde 7 jours calendaires après)
    # Note: On utilise shift(-7) si vos données sont quotidiennes
    df["close_plus_7"] = df["close"].shift(-7)
    df["delta_d"] = (df["close_plus_7"] - df["close"]).abs() / df["close"]

    # Marquage des jours d'événements
    event_days = df[df["delta_d"] > threshold]["date"].dt.date.tolist()
    event_days = sorted(list(set(event_days)))

    if not event_days:
        return []

    # Agrégation en intervalles avec tolérance de gap (3 jours)
    gt_intervals = []
    if event_days:
        start_date = event_days[0]
        prev_date = event_days[0]

        for i in range(1, len(event_days)):
            # Si l'écart est > gap_tolerance, on ferme l'intervalle
            if (event_days[i] - prev_date).days > gap_tolerance:
                gt_intervals.append(
                    {"start": pd.Timestamp(start_date), "end": pd.Timestamp(prev_date)}
                )
                start_date = event_days[i]
            prev_date = event_days[i]

        # Fermeture du dernier intervalle
        gt_intervals.append(
            {"start": pd.Timestamp(start_date), "end": pd.Timestamp(prev_date)}
        )

    return gt_intervals


def plot_sp500_ground_truth_simplified(price_df, gt_intervals):
    """
    Version simplifiée : S&P 500 avec zones GT et jalons clés (SVB & ARM).
    """
    fig = go.Figure()

    # 1. Courbe du S&P 500
    fig.add_trace(
        go.Scatter(
            x=price_df["date"],
            y=price_df["close"],
            mode="lines",
            name="S&P 500",
            line=dict(color="#3498db", width=2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Prix: %{y:.2f}$<extra></extra>",
        )
    )

    # 2. Zones d'événements Ground Truth (Surlignage rouge sans texte)
    for interval in gt_intervals:
        fig.add_vrect(
            x0=interval["start"],
            x1=interval["end"],
            fillcolor="red",
            opacity=0.15,
            layer="below",
            line_width=0,
        )

    # 3. Barres verticales discontinues (Jalons spécifiques)
    # --- Crise SVB ---
    fig.add_vline(x="2023-03-10", line_width=2, line_dash="dash", line_color="#e67e22")
    fig.add_annotation(
        x="2023-03-10",
        y=1.02,
        yref="paper",
        text="Crise SVB",
        showarrow=False,
        font=dict(color="#e67e22", size=11),
    )

    # --- IPO ARM ---
    fig.add_vline(x="2023-09-14", line_width=2, line_dash="dash", line_color="#27ae60")
    fig.add_annotation(
        x="2023-09-14",
        y=1.02,
        yref="paper",
        text="IPO ARM",
        showarrow=False,
        font=dict(color="#27ae60", size=11),
    )

    # 4. Mise en page épurée (Pas de slider)
    fig.update_layout(
        title="<b>S&P 500 (2023) — Analyse de Volatilité et Événements Clés</b>",
        xaxis_title="Date",
        yaxis_title="Prix de clôture ($)",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(t=80),
    )

    fig.show()


def generate_alerts_for_period(
    all_tweets, assigned_tweets, start_date, end_date, alert_threshold_ratio=0.20
):
    # Filtrage temporel
    mask_all = (all_tweets["date"] >= start_date) & (all_tweets["date"] <= end_date)
    mask_assigned = (assigned_tweets["date"] >= start_date) & (
        assigned_tweets["date"] <= end_date
    )

    # Agrégation
    daily_total = all_tweets[mask_all].groupby(all_tweets["date"].dt.date).size()
    daily_assigned = (
        assigned_tweets[mask_assigned].groupby(assigned_tweets["date"].dt.date).size()
    )

    # Calcul
    alert_df = pd.DataFrame({"total": daily_total, "assigned": daily_assigned}).fillna(
        0
    )
    alert_df["ratio"] = alert_df["assigned"] / alert_df["total"]
    alert_df["is_alert"] = alert_df["ratio"] >= alert_threshold_ratio

    return alert_df.reset_index().rename(columns={"index": "date"})


def evaluate_pipeline_for_period(alert_df, gt_intervals, start_date, end_date):
    """
    Évalue les performances sur une période précise.
    """
    start_obs = pd.Timestamp(start_date)
    end_obs = pd.Timestamp(end_date)

    # 1. On ne garde que les alertes produites par le système dans cette période
    # (alert_df est normalement déjà filtré, mais on sécurise)
    alerts_in_period = alert_df[
        (alert_df["is_alert"] == True)
        & (pd.to_datetime(alert_df["date"]) >= start_obs)
        & (pd.to_datetime(alert_df["date"]) <= end_obs)
    ]["date"].tolist()
    alerts_in_period = [pd.Timestamp(d) for d in alerts_in_period]

    # 2. On ne garde que les événements réels (GT) qui chevauchent la période
    relevant_gt = [
        event
        for event in gt_intervals
        if (event["start"] <= end_obs) and (event["end"] >= start_obs)
    ]

    if not relevant_gt:
        return (
            "Aucun événement de marché (GT) sur cette période pour le calcul du Recall."
        )

    # --- RECALL (Sensibilité) ---
    spotted_count = 0
    for event in relevant_gt:
        # On vérifie si une alerte tombe dans l'événement
        for alert_date in alerts_in_period:
            if event["start"] <= alert_date <= event["end"]:
                spotted_count += 1
                break

    recall = spotted_count / len(relevant_gt)

    # --- PRECISION (Fiabilité) ---
    hits = 0
    for alert_date in alerts_in_period:
        is_hit = False
        for event in relevant_gt:
            if event["start"] <= alert_date <= event["end"]:
                is_hit = True
                break
        if is_hit:
            hits += 1

    precision = hits / len(alerts_in_period) if alerts_in_period else 0

    # --- F-SCORE ---
    f_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "Période": f"{start_date} au {end_date}",
        "Événements réels (GT)": len(relevant_gt),
        "Alertes générées": len(alerts_in_period),
        "Hits (Alertes justes)": hits,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F-Score": round(f_score, 4),
    }


def plot_social_heat_curve(alert_df, gt_intervals, threshold):
    fig = px.line(
        alert_df,
        x="date",
        y="ratio",
        title="<b>Social Heat & Market Alerts</b> (Fig. 16 adaptation)",
        labels={"ratio": "% Assigned Tweets", "date": "Date"},
    )

    # Ajout du seuil d'alerte
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Alert Threshold",
    )

    # Ajout des zones d'événements réels (GT) pour voir la corrélation
    for event in gt_intervals:
        fig.add_vrect(
            x0=event["start"],
            x1=event["end"],
            fillcolor="green",
            opacity=0.1,
            line_width=0,
        )

    fig.update_layout(template="plotly_white")
    fig.show()
