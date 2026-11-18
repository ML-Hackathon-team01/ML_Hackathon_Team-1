from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "results" / "pathsnap_student_test.csv"
MODEL_FILE = ROOT / "saved_models" / "sohel_gradient_boosting.joblib"
OUT_DIR = ROOT / "results" / "hmm"

RESET_GAP_SECONDS = 60.0
SAME_ARC_WEIGHT = 0.85
CONNECTED_WEIGHT = 0.14
DISCONNECTED_WEIGHT = 0.01
EPS = 1e-12


def decode_segment(observations, probabilities):
    """Viterbi-decode one continuous sequence of GPS observations."""
    if not observations:
        return []

    # Each item is a candidate table for one GPS observation.
    log_scores = []
    backpointers = []

    first_ids = observations[0]["row_id"].to_numpy(dtype=int)
    log_scores.append(np.log(np.clip(probabilities[first_ids], EPS, 1.0)))
    backpointers.append(None)

    for t in range(1, len(observations)):
        previous = observations[t - 1]
        current = observations[t]

        prev_ids = previous["row_id"].to_numpy(dtype=int)
        curr_ids = current["row_id"].to_numpy(dtype=int)
        prev_scores = log_scores[-1]
        curr_emissions = np.log(np.clip(probabilities[curr_ids], EPS, 1.0))

        prev_arc = previous["arc_id"].to_numpy()
        prev_to = previous["to_node"].to_numpy()
        curr_arc = current["arc_id"].to_numpy()
        curr_from = current["from_node"].to_numpy()

        next_scores = np.empty(len(current), dtype=float)
        back = np.empty(len(current), dtype=int)

        for j in range(len(current)):
            # A same-arc transition is preferred; otherwise prefer a directed
            # connection from the previous arc's end node to this arc's start.
            weights = np.full(len(previous), DISCONNECTED_WEIGHT, dtype=float)
            weights[prev_to == curr_from[j]] = CONNECTED_WEIGHT
            weights[prev_arc == curr_arc[j]] = SAME_ARC_WEIGHT
            weights /= weights.sum()

            candidates = prev_scores + np.log(np.clip(weights, EPS, 1.0))
            best_previous = int(np.argmax(candidates))
            next_scores[j] = candidates[best_previous] + curr_emissions[j]
            back[j] = best_previous

        log_scores.append(next_scores)
        backpointers.append(back)

    state = int(np.argmax(log_scores[-1]))
    chosen_states = [state]

    for t in range(len(observations) - 1, 0, -1):
        state = int(backpointers[t][state])
        chosen_states.append(state)

    chosen_states.reverse()

    return [
        int(observations[t].iloc[chosen_states[t]]["row_id"])
        for t in range(len(observations))
    ]


def decode_trip(trip, probabilities):
    """Split a trip at missing observation IDs and large GPS time gaps."""
    trip = trip.sort_values(["observation_id", "candidate_rank"])
    obs_tables = [
        group.copy()
        for _, group in trip.groupby("observation_id", sort=True)
    ]

    decoded_row_ids = []
    current_segment = []
    previous_observation_id = None

    for candidates in obs_tables:
        obs_id = int(candidates["observation_id"].iloc[0])
        time_gap = float(candidates["time_gap_s"].iloc[0])

        missing_observation = (
            previous_observation_id is not None
            and obs_id > previous_observation_id + 1
        )
        long_gap = (
            previous_observation_id is not None
            and time_gap > RESET_GAP_SECONDS
        )

        if missing_observation or long_gap:
            decoded_row_ids.extend(
                decode_segment(current_segment, probabilities)
            )
            current_segment = []

        current_segment.append(candidates)
        previous_observation_id = obs_id

    decoded_row_ids.extend(decode_segment(current_segment, probabilities))
    return decoded_row_ids


def continuity_metrics(selected):
    """Calculate continuity using both all adjacent observations and a gap filter."""
    total_all = good_all = 0
    total_filtered = good_filtered = 0

    for _, trip in selected.groupby("trip_id", sort=False):
        trip = trip.sort_values("observation_id")
        rows = list(trip.itertuples(index=False))

        for previous, current in zip(rows, rows[1:]):
            # Do not score across absent observations.
            if current.observation_id != previous.observation_id + 1:
                continue

            continuous = (
                current.arc_id == previous.arc_id
                or current.from_node == previous.to_node
            )

            total_all += 1
            good_all += int(continuous)

            # Supplemental continuity score excluding long time gaps.
            if current.time_gap_s <= RESET_GAP_SECONDS:
                total_filtered += 1
                good_filtered += int(continuous)

    return {
        "route_continuity_all_adjacent_observations": (
            round(good_all / total_all, 5) if total_all else None
        ),
        "route_continuity_excluding_long_gaps": (
            round(good_filtered / total_filtered, 5)
            if total_filtered else None
        ),
    }


def summarize(selected, method):
    return {
        "method": method,
        "observations": int(len(selected)),
        # This is provisional because the current labels are route-membership
        # labels, not verified point-by-point ground truth.
        "provisional_top1_match_rate": round(
            float(selected["label"].mean()), 5
        ),
        **continuity_metrics(selected),
    }


def main():
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing student-test candidates: {TEST_CSV}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing saved model: {MODEL_FILE}")

    data = pd.read_csv(TEST_CSV, dtype={"trip_id": str})
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    features = bundle["features"]

    required = {
        "trip_id", "observation_id", "candidate_rank",
        "arc_id", "from_node", "to_node",
        "time_gap_s", "label", *features,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Student-test CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.reset_index(drop=True)
    data["row_id"] = np.arange(len(data))

    # Classifier probabilities are the HMM emission scores.
    probabilities = model.predict_proba(data[features])[:, 1]

    # Classifier-only: choose the highest-probability road for each point.
    classifier_ids = (
        data.assign(score=probabilities)
        .groupby(["trip_id", "observation_id"], sort=False)["score"]
        .idxmax()
    )
    classifier_only = data.loc[classifier_ids].copy()

    # HMM/Viterbi: choose the highest-probability sequence using road links.
    hmm_ids = []
    for _, trip in data.groupby("trip_id", sort=False):
        hmm_ids.extend(decode_trip(trip, probabilities))
    hmm_selected = data.loc[hmm_ids].copy()

    classifier_metrics = summarize(classifier_only, "classifier_only")
    hmm_metrics = summarize(hmm_selected, "classifier_plus_hmm_viterbi")

    result = {
        "reset_gap_seconds": RESET_GAP_SECONDS,
        "transition_weights": {
            "same_arc": SAME_ARC_WEIGHT,
            "directed_connected_arc": CONNECTED_WEIGHT,
            "disconnected_fallback": DISCONNECTED_WEIGHT,
        },
        "classifier_only": classifier_metrics,
        "classifier_plus_hmm_viterbi": hmm_metrics,
        "label_warning": (
            "The current labels are provisional: label 1 means the arc appears "
            "somewhere in that trip's reference route, not necessarily at that "
            "specific GPS observation. Treat match rates as provisional."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classifier_only["method"] = "classifier_only"
    hmm_selected["method"] = "classifier_plus_hmm_viterbi"
    pd.concat([classifier_only, hmm_selected], ignore_index=True).to_csv(
        OUT_DIR / "gb_classifier_vs_hmm_predictions.csv",
        index=False,
    )

    with open(
        OUT_DIR / "gb_classifier_vs_hmm_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nPredictions saved: {OUT_DIR / 'gb_classifier_vs_hmm_predictions.csv'}")
    print(f"Metrics saved:     {OUT_DIR / 'gb_classifier_vs_hmm_metrics.json'}")


if __name__ == "__main__":
    main()
