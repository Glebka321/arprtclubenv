"""Simulate a 5-email campaign and optimize send times for opens/CTR/purchases."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from clothing_brand_ctr_env import ClothingBrandCtrAction
from clothing_brand_ctr_env.server.clothing_brand_ctr_env_environment import (
    ClothingBrandCtrEnvironment,
)
from simulate_brand_campaign import (
    MLEngineerJudge,
    TenXMarketerJudge,
    AudiencePersona,
    clamp,
    load_or_create_persona_dataset,
    parse_send_hours,
    send_time_alignment,
)

WEEKDAY_TO_INDEX = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}

INDEX_TO_WEEKDAY = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


@dataclass(frozen=True)
class EmailStepConfig:
    """Definition of each campaign email."""

    step_idx: int
    step_name: str
    brand_voice: str
    key_value_prop: str
    call_to_action: str


@dataclass
class EmailStepGenerated:
    """Generated copy + quality info for a campaign step."""

    step_idx: int
    step_name: str
    brand_voice: str
    key_value_prop: str
    call_to_action: str
    subject_line: str
    preview_text: str
    email_copy: str
    ctr_proxy_score: float
    marketer_score: float
    marketer_rationale: str
    generation_source: str


def default_email_steps() -> List[EmailStepConfig]:
    """Default 5-email sequence for brand launch lifecycle."""
    return [
        EmailStepConfig(
            step_idx=1,
            step_name="teaser",
            brand_voice="playful",
            key_value_prop="travel tees that stay crisp from gate to dinner",
            call_to_action="Get on the early-access list",
        ),
        EmailStepConfig(
            step_idx=2,
            step_name="launch",
            brand_voice="bold",
            key_value_prop="premium airport-ready staples built for repeat wear",
            call_to_action="Shop the launch drop",
        ),
        EmailStepConfig(
            step_idx=3,
            step_name="proof",
            brand_voice="minimal",
            key_value_prop="easy travel fits that real flyers keep reaching for",
            call_to_action="See bestsellers",
        ),
        EmailStepConfig(
            step_idx=4,
            step_name="offer",
            brand_voice="luxury",
            key_value_prop="elevated basics with lounge-level comfort",
            call_to_action="Unlock your limited offer",
        ),
        EmailStepConfig(
            step_idx=5,
            step_name="last_call",
            brand_voice="bold",
            key_value_prop="the last restock for your carry-on uniform",
            call_to_action="Claim your size now",
        ),
    ]


def generate_email_steps(
    brand_name: str,
    target_audience: str,
    steps: List[EmailStepConfig],
) -> List[EmailStepGenerated]:
    """Generate campaign email copy for each step."""
    env = ClothingBrandCtrEnvironment()
    env.reset()
    marketer = TenXMarketerJudge()
    generated: List[EmailStepGenerated] = []

    for step in steps:
        action = ClothingBrandCtrAction(
            brand_name=brand_name,
            target_audience=target_audience,
            brand_voice=step.brand_voice,
            key_value_prop=step.key_value_prop,
            call_to_action=step.call_to_action,
            metadata={"campaign_step": step.step_name},
        )
        observation = env.step(action)

        step_generated = EmailStepGenerated(
            step_idx=step.step_idx,
            step_name=step.step_name,
            brand_voice=step.brand_voice,
            key_value_prop=step.key_value_prop,
            call_to_action=step.call_to_action,
            subject_line=observation.subject_line,
            preview_text=observation.preview_text,
            email_copy=observation.email_copy,
            ctr_proxy_score=observation.ctr_proxy_score,
            marketer_score=0.0,
            marketer_rationale="",
            generation_source=observation.metadata.get("generation_source", "unknown"),
        )
        score, rationale = marketer.score(
            arm=type(
                "TempArm",
                (),
                {
                    "validation": observation.validation,
                    "ctr_proxy_score": observation.ctr_proxy_score,
                    "email_copy": observation.email_copy,
                    "preview_text": observation.preview_text,
                    "subject_line": observation.subject_line,
                    "key_value_prop": step.key_value_prop,
                },
            )(),
            brand_name=brand_name,
        )
        step_generated.marketer_score = score
        step_generated.marketer_rationale = rationale
        generated.append(step_generated)
    return generated


def _schedule_seed(schedule: Sequence[Tuple[int, int]], seed: int) -> int:
    """Create deterministic seed from schedule."""
    flattened = ",".join(f"{day}-{hour}" for day, hour in schedule)
    digest = hashlib.sha256((flattened + f"|{seed}").encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def parse_send_days(raw: str) -> List[int]:
    """Parse comma-separated weekday tokens into day indices [0=Mon..6=Sun]."""
    days: List[int] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in WEEKDAY_TO_INDEX:
            raise ValueError(f"Invalid send day: {token}")
        days.append(WEEKDAY_TO_INDEX[key])
    if not days:
        raise ValueError("At least one send day is required.")
    return sorted(set(days))


def _slot_label(day_idx: int, hour: int) -> str:
    """Format one send slot."""
    return f"{INDEX_TO_WEEKDAY.get(day_idx, '?')} {hour:02d}:00"


def _format_schedule(schedule: Sequence[Tuple[int, int]]) -> str:
    """Format schedule into a compact label string."""
    return " | ".join(_slot_label(day, hour) for day, hour in schedule)


def _persona_preferred_day(persona: AudiencePersona) -> int:
    """Infer a stable preferred day-of-week from persona attributes."""
    digest = hashlib.sha256(persona.persona_id.encode("utf-8")).hexdigest()
    seed_val = int(digest[:12], 16)
    rng = random.Random(seed_val)

    segment = persona.segment.lower()
    if any(term in segment for term in ["student", "artist", "creative", "freelance"]):
        choices = [5, 6]
    elif any(term in segment for term in ["manager", "engineer", "director", "consultant"]):
        choices = [1, 2, 3]
    elif persona.travel_intensity >= 9:
        choices = [3, 4, 5, 6]
    else:
        choices = [0, 1, 2, 3, 4, 5, 6]
    return rng.choice(choices)


def send_day_alignment(send_day: int, preferred_day: int) -> float:
    """Return alignment score in [-1, 1] based on weekday distance."""
    diff = abs(send_day - preferred_day) % 7
    wrapped = min(diff, 7 - diff)
    capped = min(wrapped, 3)
    return ((3 - capped) / 3.0) * 2.0 - 1.0


def open_prob_for_step(
    step: EmailStepGenerated,
    send_day: int,
    send_hour: int,
    persona: AudiencePersona,
    engagement: float,
    streak_unopened: int,
) -> float:
    """Probability of open for one persona on one campaign step."""
    hour_alignment = send_time_alignment(send_hour, persona.preferred_send_hour)
    day_alignment = send_day_alignment(send_day, _persona_preferred_day(persona))
    style_match = 1.0 if step.brand_voice == persona.style_preference else 0.0
    travel_signal = 1.0 if "travel" in (step.preview_text + step.email_copy).lower() else 0.0
    fatigue = max(0.0, streak_unopened * 0.015)

    prob = persona.open_base
    prob += 0.13 * ((step.marketer_score / 100.0) - 0.5)
    prob += 0.10 * (step.ctr_proxy_score - 0.5)
    prob += 0.08 * hour_alignment
    prob += 0.07 * day_alignment
    prob += 0.05 * style_match
    prob += 0.07 * engagement
    prob += 0.03 * travel_signal * min(persona.travel_intensity / 10.0, 1.0)
    prob -= fatigue
    return clamp(prob, 0.01, 0.95)


def click_prob_for_step(
    step: EmailStepGenerated,
    persona: AudiencePersona,
    opened: bool,
    engagement: float,
) -> float:
    """Probability of click after open."""
    if not opened:
        return 0.0
    style_match = 1.0 if step.brand_voice == persona.style_preference else 0.0

    prob = persona.click_base
    prob += 0.13 * ((step.marketer_score / 100.0) - 0.5)
    prob += 0.12 * (step.ctr_proxy_score - 0.5)
    prob += 0.05 * style_match
    prob += 0.04 * engagement
    return clamp(prob, 0.005, 0.85)


def purchase_prob_for_step(
    step_idx: int,
    step: EmailStepGenerated,
    persona: AudiencePersona,
    clicked: bool,
    engagement: float,
) -> float:
    """Probability of purchase after click."""
    if not clicked:
        return 0.0
    style_match = 1.0 if step.brand_voice == persona.style_preference else 0.0
    stage_lift = [0.75, 0.9, 1.0, 1.15, 1.3][min(max(step_idx - 1, 0), 4)]
    income_lift = {"premium": 0.05, "mid": 0.02}.get(persona.income_band, 0.0)

    prob = persona.purchase_base * stage_lift
    prob += 0.08 * ((step.marketer_score / 100.0) - 0.5)
    prob += 0.05 * style_match
    prob += 0.03 * engagement
    prob += income_lift
    return clamp(prob, 0.002, 0.60)


def simulate_schedule(
    schedule: Sequence[Tuple[int, int]],
    generated_steps: List[EmailStepGenerated],
    personas: List[AudiencePersona],
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    """Simulate entire 5-email sequence for one day+time schedule."""
    if len(schedule) != len(generated_steps):
        raise ValueError("Schedule length must match number of campaign emails.")

    rng = random.Random(_schedule_seed(schedule, seed))
    states: Dict[str, Dict[str, float | int | bool]] = {
        p.persona_id: {"engagement": 0.0, "streak_unopened": 0, "purchased": False}
        for p in personas
    }
    persona_outcomes: Dict[str, Dict[str, object]] = {
        p.persona_id: {
            "persona_id": p.persona_id,
            "segment": p.segment,
            "style_preference": p.style_preference,
            "income_band": p.income_band,
            "travel_intensity": p.travel_intensity,
            "preferred_send_hour": p.preferred_send_hour,
            "opens": 0,
            "clicks": 0,
            "purchased": False,
            "purchase_step_idx": None,
            "purchase_step_name": "",
            "purchase_day": "",
            "purchase_hour": None,
        }
        for p in personas
    }

    total_sent = total_opens = total_clicks = total_purchases = 0
    step_metrics: List[Dict[str, object]] = []

    for idx, step in enumerate(generated_steps):
        day_idx, hour = schedule[idx]
        sent = opens = clicks = purchases = 0

        for persona in personas:
            state = states[persona.persona_id]
            if bool(state["purchased"]):
                continue

            sent += 1
            open_prob = open_prob_for_step(
                step=step,
                send_day=day_idx,
                send_hour=hour,
                persona=persona,
                engagement=float(state["engagement"]),
                streak_unopened=int(state["streak_unopened"]),
            )
            opened = rng.random() < open_prob

            click_prob = click_prob_for_step(
                step=step,
                persona=persona,
                opened=opened,
                engagement=float(state["engagement"]),
            )
            clicked = opened and (rng.random() < click_prob)

            purchase_prob = purchase_prob_for_step(
                step_idx=idx + 1,
                step=step,
                persona=persona,
                clicked=clicked,
                engagement=float(state["engagement"]),
            )
            purchased = clicked and (rng.random() < purchase_prob)

            if opened:
                opens += 1
                persona_outcomes[persona.persona_id]["opens"] = (
                    int(persona_outcomes[persona.persona_id]["opens"]) + 1
                )
                state["streak_unopened"] = 0
                state["engagement"] = clamp(float(state["engagement"]) + 0.08, -0.2, 0.6)
            else:
                state["streak_unopened"] = int(state["streak_unopened"]) + 1
                state["engagement"] = clamp(float(state["engagement"]) - 0.03, -0.2, 0.6)

            if clicked:
                clicks += 1
                persona_outcomes[persona.persona_id]["clicks"] = (
                    int(persona_outcomes[persona.persona_id]["clicks"]) + 1
                )
                state["engagement"] = clamp(float(state["engagement"]) + 0.07, -0.2, 0.7)

            if purchased:
                purchases += 1
                state["purchased"] = True
                persona_outcomes[persona.persona_id]["purchased"] = True
                persona_outcomes[persona.persona_id]["purchase_step_idx"] = step.step_idx
                persona_outcomes[persona.persona_id]["purchase_step_name"] = step.step_name
                persona_outcomes[persona.persona_id]["purchase_day"] = INDEX_TO_WEEKDAY.get(
                    day_idx,
                    "?",
                )
                persona_outcomes[persona.persona_id]["purchase_hour"] = hour

        total_sent += sent
        total_opens += opens
        total_clicks += clicks
        total_purchases += purchases

        step_metrics.append(
            {
                "step_idx": step.step_idx,
                "step_name": step.step_name,
                "send_day_idx": day_idx,
                "send_day": INDEX_TO_WEEKDAY.get(day_idx, "?"),
                "send_hour": hour,
                "sent": sent,
                "opens": opens,
                "clicks": clicks,
                "purchases": purchases,
                "open_rate": round(opens / sent, 4) if sent else 0.0,
                "ctr": round(clicks / sent, 4) if sent else 0.0,
                "purchase_rate": round(purchases / sent, 4) if sent else 0.0,
                "subject_line": step.subject_line,
                "generation_source": step.generation_source,
            }
        )

    audience_size = len(personas)
    open_rate = (total_opens / total_sent) if total_sent else 0.0
    ctr = (total_clicks / total_sent) if total_sent else 0.0
    purchase_rate = (total_purchases / audience_size) if audience_size else 0.0
    click_to_open = (total_clicks / total_opens) if total_opens else 0.0

    campaign_score = (0.15 * open_rate) + (0.25 * ctr) + (0.60 * purchase_rate)
    summary = {
        "arm_id": "schedule_" + "_".join(f"{day}_{hour:02d}" for day, hour in schedule),
        "schedule": _format_schedule(schedule),
        "sent": total_sent,
        "opens": total_opens,
        "clicks": total_clicks,
        "purchases": total_purchases,
        "open_rate": round(open_rate, 4),
        "ctr": round(ctr, 4),
        "click_to_open_rate": round(click_to_open, 4),
        "purchase_rate": round(purchase_rate, 4),
        "composite_score": round(campaign_score, 4),
        "generation_source": generated_steps[0].generation_source if generated_steps else "unknown",
        "marketer_score": round(
            sum(step.marketer_score for step in generated_steps) / max(1, len(generated_steps)),
            2,
        ),
    }
    for i, (day_idx, hour) in enumerate(schedule, start=1):
        summary[f"email_{i}_day_idx"] = day_idx
        summary[f"email_{i}_day"] = INDEX_TO_WEEKDAY.get(day_idx, "?")
        summary[f"email_{i}_hour"] = hour

    purchaser_rows = [
        row for row in persona_outcomes.values() if bool(row["purchased"])
    ]
    purchaser_rows.sort(
        key=lambda row: (
            int(row["purchase_step_idx"]) if row["purchase_step_idx"] is not None else 999,
            -int(row["clicks"]),
            -int(row["opens"]),
            str(row["persona_id"]),
        )
    )
    return summary, step_metrics, purchaser_rows


def candidate_schedules(
    send_slots: List[Tuple[int, int]],
    num_emails: int,
    max_exhaustive: int,
    sampled_count: int,
    seed: int,
) -> Tuple[List[Tuple[Tuple[int, int], ...]], str, int]:
    """Build schedule candidates via exhaustive or sampled search."""
    total = len(send_slots) ** num_emails
    if total <= max_exhaustive:
        return list(itertools.product(send_slots, repeat=num_emails)), "exhaustive", total

    rng = random.Random(seed)
    needed = min(total, sampled_count)
    schedules = set()
    while len(schedules) < needed:
        schedules.add(tuple(rng.choice(send_slots) for _ in range(num_emails)))
    return list(schedules), "sampled", total


def save_step_metrics(step_metrics: List[Dict[str, object]], output_path: Path) -> None:
    """Persist per-email metrics for best schedule."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "step_idx",
            "step_name",
            "send_day_idx",
            "send_day",
            "send_hour",
            "sent",
            "opens",
            "clicks",
            "purchases",
            "open_rate",
            "ctr",
            "purchase_rate",
            "generation_source",
            "subject_line",
        ]
        writer = __import__("csv").DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in step_metrics:
            writer.writerow(row)


def save_schedule_results(
    ranked: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """Persist ranked schedule-level metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank",
            "arm_id",
            "schedule",
            "email_1_day_idx",
            "email_1_day",
            "email_1_hour",
            "email_2_day_idx",
            "email_2_day",
            "email_2_hour",
            "email_3_day_idx",
            "email_3_day",
            "email_3_hour",
            "email_4_day_idx",
            "email_4_day",
            "email_4_hour",
            "email_5_day_idx",
            "email_5_day",
            "email_5_hour",
            "sent",
            "opens",
            "clicks",
            "purchases",
            "open_rate",
            "ctr",
            "click_to_open_rate",
            "purchase_rate",
            "composite_score",
            "generation_source",
            "marketer_score",
        ]
        writer = __import__("csv").DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "arm_id": row["arm_id"],
                    "schedule": row["schedule"],
                    "email_1_day_idx": row["email_1_day_idx"],
                    "email_1_day": row["email_1_day"],
                    "email_1_hour": row["email_1_hour"],
                    "email_2_day_idx": row["email_2_day_idx"],
                    "email_2_day": row["email_2_day"],
                    "email_2_hour": row["email_2_hour"],
                    "email_3_day_idx": row["email_3_day_idx"],
                    "email_3_day": row["email_3_day"],
                    "email_3_hour": row["email_3_hour"],
                    "email_4_day_idx": row["email_4_day_idx"],
                    "email_4_day": row["email_4_day"],
                    "email_4_hour": row["email_4_hour"],
                    "email_5_day_idx": row["email_5_day_idx"],
                    "email_5_day": row["email_5_day"],
                    "email_5_hour": row["email_5_hour"],
                    "sent": row["sent"],
                    "opens": row["opens"],
                    "clicks": row["clicks"],
                    "purchases": row["purchases"],
                    "open_rate": row["open_rate"],
                    "ctr": row["ctr"],
                    "click_to_open_rate": row["click_to_open_rate"],
                    "purchase_rate": row["purchase_rate"],
                    "composite_score": row["composite_score"],
                    "generation_source": row["generation_source"],
                    "marketer_score": row["marketer_score"],
                }
            )


def print_schedule_results(
    ranked: List[Dict[str, object]],
    top_steps: List[Dict[str, object]],
    top_purchasers: List[Dict[str, object]],
    search_mode: str,
    searched_count: int,
    total_count: int,
    ml_score: float,
    ml_rationale: str,
) -> None:
    """Print summary leaderboard and winning schedule details."""
    print("\n5-Email Schedule Leaderboard")
    print("-" * 170)
    print(
        f"{'Rank':<5} {'Schedule':<70} {'Score':<7} {'Open':<7} {'CTR':<7} "
        f"{'Purchase':<9} {'Opens':<7} {'Clicks':<7} {'Buys':<7}"
    )
    print("-" * 170)
    for rank, row in enumerate(ranked[:10], start=1):
        print(
            f"{rank:<5} {str(row['schedule']):<70} {float(row['composite_score']):<7.3f} "
            f"{float(row['open_rate']):<7.3f} {float(row['ctr']):<7.3f} "
            f"{float(row['purchase_rate']):<9.3f} {int(row['opens']):<7} {int(row['clicks']):<7} {int(row['purchases']):<7}"
        )

    best = ranked[0]
    print("\nBest 5-Email Schedule")
    print("-" * 170)
    print(f"Schedule: {best['schedule']}")
    print(f"Composite score: {best['composite_score']}")
    print(f"Open rate: {best['open_rate']}")
    print(f"CTR: {best['ctr']}")
    print(f"Purchase rate (per audience): {best['purchase_rate']}")

    print("\nBest Schedule Step Breakdown")
    print("-" * 170)
    for row in top_steps:
        print(
            f"Email {row['step_idx']} ({row['step_name']}), {row['send_day']} {int(row['send_hour']):02d}:00 -> "
            f"open {float(row['open_rate']):.3f}, ctr {float(row['ctr']):.3f}, purchase {float(row['purchase_rate']):.3f}"
        )

    print("\nJudge: ML Engineer (A/B Testing)")
    print("-" * 170)
    print(f"Experiment score: {ml_score}/100")
    print(ml_rationale)

    print("\nTop 10 Personas That Purchased")
    print("-" * 170)
    if not top_purchasers:
        print("No purchases in this run.")
    else:
        for rank, row in enumerate(top_purchasers[:10], start=1):
            print(
                f"{rank}. {row['persona_id']} | {row['segment']} | "
                f"Email {row['purchase_step_idx']} ({row['purchase_step_name']}) at "
                f"{row['purchase_day']} {int(row['purchase_hour']):02d}:00 | "
                f"opens={row['opens']} clicks={row['clicks']} "
                f"style={row['style_preference']} income={row['income_band']}"
            )

    print("\nSearch Details")
    print("-" * 170)
    print(f"Mode: {search_mode}")
    print(f"Schedules evaluated: {searched_count}")
    print(f"Total possible schedules: {total_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate 5-email campaign and optimize send day+time schedule.",
    )
    parser.add_argument("--brand-name", default="AIRPORT CLUB")
    parser.add_argument("--target-audience", default="fellow traveler")
    parser.add_argument(
        "--send-days",
        default="mon,tue,wed,thu,fri,sat,sun",
        help="Candidate send days (e.g., mon,tue,wed).",
    )
    parser.add_argument(
        "--send-hours",
        default="8,10,12,15,18,21",
        help="Candidate send hours to optimize across.",
    )
    parser.add_argument("--num-emails", type=int, default=5)
    parser.add_argument(
        "--persona-dataset",
        default="outputs/evals/persona_dataset.csv",
    )
    parser.add_argument("--persona-count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--persona-source", choices=["hf", "synthetic"], default="hf")
    parser.add_argument(
        "--hf-dataset-name",
        default="nvidia/Nemotron-Personas-USA",
    )
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--refresh-personas", action="store_true")
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--max-exhaustive", type=int, default=5000)
    parser.add_argument("--schedule-samples", type=int, default=2500)
    parser.add_argument(
        "--schedule-results-csv",
        default="outputs/evals/five_email_schedule_results.csv",
    )
    parser.add_argument(
        "--best-step-results-csv",
        default="outputs/evals/five_email_best_schedule_steps.csv",
    )
    args = parser.parse_args()

    if args.num_emails != 5:
        raise ValueError("This simulator is configured specifically for 5-email campaigns.")

    send_days = parse_send_days(args.send_days)
    send_hours = parse_send_hours(args.send_hours)
    send_slots = [(day_idx, hour) for day_idx in send_days for hour in send_hours]
    personas = load_or_create_persona_dataset(
        dataset_path=Path(args.persona_dataset),
        size=args.persona_count,
        seed=args.seed,
        source=args.persona_source,
        hf_dataset_name=args.hf_dataset_name,
        hf_split=args.hf_split,
        refresh=args.refresh_personas,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )

    steps = generate_email_steps(
        brand_name=args.brand_name,
        target_audience=args.target_audience,
        steps=default_email_steps(),
    )
    schedules, search_mode, total_count = candidate_schedules(
        send_slots=send_slots,
        num_emails=args.num_emails,
        max_exhaustive=args.max_exhaustive,
        sampled_count=args.schedule_samples,
        seed=args.seed,
    )

    summaries: List[Dict[str, object]] = []
    best_step_metrics: List[Dict[str, object]] = []
    best_purchasers: List[Dict[str, object]] = []
    for sched_idx, schedule in enumerate(schedules):
        summary, step_metrics, purchaser_rows = simulate_schedule(
            schedule=schedule,
            generated_steps=steps,
            personas=personas,
            seed=args.seed + (sched_idx * 3571),
        )
        summaries.append(summary)
        if not best_step_metrics:
            best_step_metrics = step_metrics
        if not best_purchasers:
            best_purchasers = purchaser_rows

    ranked = sorted(
        summaries,
        key=lambda row: (
            float(row["composite_score"]),
            float(row["purchase_rate"]),
            float(row["ctr"]),
            float(row["open_rate"]),
        ),
        reverse=True,
    )

    best_schedule = tuple(
        (
            int(ranked[0][f"email_{i}_day_idx"]),
            int(ranked[0][f"email_{i}_hour"]),
        )
        for i in range(1, args.num_emails + 1)
    )
    _, best_step_metrics, best_purchasers = simulate_schedule(
        schedule=best_schedule,
        generated_steps=steps,
        personas=personas,
        seed=args.seed + 999983,
    )

    ml_judge = MLEngineerJudge()
    ml_score, ml_rationale = ml_judge.score_experiment(
        ranked_results=ranked,
        audience_size=len(personas),
    )

    save_schedule_results(ranked, Path(args.schedule_results_csv))
    save_step_metrics(best_step_metrics, Path(args.best_step_results_csv))

    print_schedule_results(
        ranked=ranked,
        top_steps=best_step_metrics,
        top_purchasers=best_purchasers,
        search_mode=search_mode,
        searched_count=len(schedules),
        total_count=total_count,
        ml_score=ml_score,
        ml_rationale=ml_rationale,
    )
    print(f"\nPersona source: {personas[0].persona_source if personas else 'unknown'}")
    print(
        "Candidate send slots: "
        + ", ".join(_slot_label(day_idx, hour) for day_idx, hour in send_slots)
    )
    print(f"Persona dataset: {args.persona_dataset}")
    print(f"Schedule results: {args.schedule_results_csv}")
    print(f"Best schedule step results: {args.best_step_results_csv}")


if __name__ == "__main__":
    main()
