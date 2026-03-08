"""Simulate brand email campaign performance with judge personas + A/B testing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from clothing_brand_ctr_env import ClothingBrandCtrAction
from clothing_brand_ctr_env.server.clothing_brand_ctr_env_environment import (
    ClothingBrandCtrEnvironment,
)


@dataclass(frozen=True)
class EmailVariant:
    """Single copy variant configuration."""

    name: str
    brand_voice: str
    key_value_prop: str
    call_to_action: str


@dataclass(frozen=True)
class AudiencePersona:
    """Audience member with behavior priors used in simulation."""

    persona_id: str
    segment: str
    style_preference: str
    travel_intensity: int
    income_band: str
    preferred_send_hour: int
    open_base: float
    click_base: float
    purchase_base: float
    persona_source: str = "synthetic"


@dataclass
class CampaignArm:
    """Variant + send-time arm with generated copy and judge scores."""

    arm_id: str
    variant_name: str
    brand_voice: str
    send_hour: int
    key_value_prop: str
    call_to_action: str
    subject_line: str
    preview_text: str
    email_copy: str
    word_count: int
    validation: Dict[str, bool]
    ctr_proxy_score: float
    generation_source: str
    marketer_score: float
    marketer_rationale: str


def default_variants() -> List[EmailVariant]:
    """Default set of copy variants for campaign testing."""
    return [
        EmailVariant(
            name="Bold Launch",
            brand_voice="bold",
            key_value_prop="premium t-shirts for standing out when you travel",
            call_to_action="Shop the launch collection now",
        ),
        EmailVariant(
            name="Minimal Core",
            brand_voice="minimal",
            key_value_prop="clean wardrobe staples designed for bringing to the airport",
            call_to_action="Explore the core collection",
        ),
        EmailVariant(
            name="Quiet Luxury",
            brand_voice="luxury",
            key_value_prop="soft luxury items for traveling",
            call_to_action="View the curated edit",
        ),
        EmailVariant(
            name="Playful Drop",
            brand_voice="playful",
            key_value_prop="the only t-shirt you need for a 14 hour flight",
            call_to_action="See what's new today",
        ),
    ]


class TenXMarketerJudge:
    """Judge persona: growth marketer with successful fashion-brand launches."""

    name = "10x_marketer"

    def score(self, arm: CampaignArm, brand_name: str) -> Tuple[float, str]:
        """Score copy quality and brand positioning consistency from 0-100."""
        score = 40.0
        notes: List[str] = []

        if arm.validation.get("subject_has_brand"):
            score += 10
            notes.append("Subject includes brand")
        if arm.validation.get("subject_length_ok"):
            score += 8
            notes.append("Subject length is healthy")
        if arm.validation.get("preview_length_ok"):
            score += 8
            notes.append("Preview length supports opens")
        if arm.validation.get("body_has_cta"):
            score += 10
            notes.append("CTA is present")
        if arm.validation.get("body_word_count_ok"):
            score += 8
            notes.append("Body length is concise")

        score += arm.ctr_proxy_score * 10

        lower_copy = arm.email_copy.lower()
        lower_preview = arm.preview_text.lower()
        lower_subject = arm.subject_line.lower()
        quirky_terms = {
            "lounge",
            "flight",
            "airport",
            "carry-on",
            "gate",
            "jet",
            "runway",
            "passport",
        }
        if any(term in lower_copy or term in lower_preview for term in quirky_terms):
            score += 8
            notes.append("Quirky travel lexicon present")

        positioning_text = f"{arm.key_value_prop} {arm.preview_text}"
        if len(positioning_text.split()) <= 24:
            score += 6
            notes.append("Positioning is short")

        if brand_name.lower() in lower_subject and brand_name.lower() in lower_copy:
            score += 5
            notes.append("Brand consistency is strong")

        return max(0.0, min(100.0, round(score, 2))), "; ".join(notes[:4])


class MLEngineerJudge:
    """Judge persona: ML engineer focused on experiment quality and reliability."""

    name = "ml_engineer_ab_tester"

    def score_experiment(
        self,
        ranked_results: List[Dict[str, object]],
        audience_size: int,
    ) -> Tuple[float, str]:
        """Score A/B experiment rigor from 0-100."""
        if len(ranked_results) < 2:
            return 0.0, "Need at least two arms to evaluate A/B quality."

        best = ranked_results[0]
        runner_up = ranked_results[1]

        p_purchase = two_proportion_p_value(
            int(best["purchases"]),
            int(best["sent"]),
            int(runner_up["purchases"]),
            int(runner_up["sent"]),
        )
        p_ctr = two_proportion_p_value(
            int(best["clicks"]),
            int(best["sent"]),
            int(runner_up["clicks"]),
            int(runner_up["sent"]),
        )

        score = 55.0
        notes: List[str] = []

        if audience_size >= 1000:
            score += 15
            notes.append("Strong sample size")
        elif audience_size >= 500:
            score += 10
            notes.append("Good sample size")
        else:
            score -= 5
            notes.append("Limited sample size")

        if p_purchase < 0.05:
            score += 18
            notes.append("Purchase lift appears significant")
        elif p_purchase < 0.1:
            score += 10
            notes.append("Purchase lift is directional")
        else:
            score -= 6
            notes.append("Purchase lift not significant")

        if p_ctr < 0.05:
            score += 10
            notes.append("CTR lift appears significant")
        elif p_ctr >= 0.2:
            score -= 4
            notes.append("CTR separation is weak")

        rationale = (
            f"{'; '.join(notes[:4])}. "
            f"p_purchase={p_purchase:.4f}, p_ctr={p_ctr:.4f}. "
            f"Recommended arm={best['arm_id']}."
        )
        return max(0.0, min(100.0, round(score, 2))), rationale


def generate_persona_dataset(size: int, seed: int) -> List[AudiencePersona]:
    """Generate synthetic audience personas used to simulate campaign outcomes."""
    rng = random.Random(seed)
    archetypes = [
        {
            "segment": "frequent_flyer_creative",
            "styles": ["bold", "playful"],
            "travel": (6, 14),
            "income": "premium",
            "open": (0.27, 0.42),
            "click": (0.09, 0.18),
            "purchase": (0.03, 0.09),
            "hours": [7, 8, 20, 21],
        },
        {
            "segment": "minimal_urban_professional",
            "styles": ["minimal", "luxury"],
            "travel": (3, 8),
            "income": "premium",
            "open": (0.22, 0.36),
            "click": (0.07, 0.14),
            "purchase": (0.025, 0.07),
            "hours": [8, 12, 18],
        },
        {
            "segment": "budget_style_hunter",
            "styles": ["playful", "bold", "minimal"],
            "travel": (1, 5),
            "income": "mid",
            "open": (0.18, 0.3),
            "click": (0.05, 0.11),
            "purchase": (0.01, 0.04),
            "hours": [12, 18, 19],
        },
        {
            "segment": "luxury_fashion_reader",
            "styles": ["luxury", "minimal"],
            "travel": (4, 10),
            "income": "premium",
            "open": (0.24, 0.38),
            "click": (0.08, 0.15),
            "purchase": (0.03, 0.08),
            "hours": [9, 13, 20],
        },
    ]
    weights = [0.28, 0.32, 0.22, 0.18]

    personas: List[AudiencePersona] = []
    for idx in range(size):
        archetype = rng.choices(archetypes, weights=weights, k=1)[0]
        personas.append(
            AudiencePersona(
                persona_id=f"p_{idx:05d}",
                segment=archetype["segment"],
                style_preference=rng.choice(archetype["styles"]),
                travel_intensity=rng.randint(*archetype["travel"]),
                income_band=archetype["income"],
                preferred_send_hour=rng.choice(archetype["hours"]),
                open_base=round(rng.uniform(*archetype["open"]), 4),
                click_base=round(rng.uniform(*archetype["click"]), 4),
                purchase_base=round(rng.uniform(*archetype["purchase"]), 4),
                persona_source="synthetic",
            )
        )
    return personas


def _clamp_int(value: int, low: int, high: int) -> int:
    """Clamp integer value to [low, high]."""
    return max(low, min(high, value))


def _infer_style_preference(text: str, rng: random.Random) -> str:
    """Infer style preference label from Nemotron persona text."""
    groups = {
        "luxury": ["luxury", "couture", "designer", "high-end", "refined", "elegant"],
        "minimal": ["minimal", "clean", "understated", "simple", "neutral"],
        "playful": ["playful", "fun", "quirky", "vibrant", "whimsical", "colorful"],
        "bold": ["bold", "edgy", "statement", "daring", "confident"],
    }
    scores = {
        label: sum(text.count(keyword) for keyword in keywords)
        for label, keywords in groups.items()
    }
    best_score = max(scores.values())
    if best_score <= 0:
        return rng.choice(["minimal", "bold", "playful", "luxury"])
    winners = [label for label, score in scores.items() if score == best_score]
    return rng.choice(winners)


def _infer_income_band(occupation: str, education_level: str) -> str:
    """Map occupation/education to simplified income band."""
    text = f"{occupation} {education_level}".lower()
    premium_terms = [
        "physician",
        "surgeon",
        "lawyer",
        "attorney",
        "executive",
        "director",
        "manager",
        "engineer",
        "founder",
        "consultant",
        "architect",
        "master",
        "doctorate",
        "phd",
    ]
    if any(term in text for term in premium_terms):
        return "premium"
    return "mid"


def _infer_preferred_send_hour(age: int, occupation_text: str, rng: random.Random) -> int:
    """Infer likely preferred send hour from age/occupation."""
    if age < 28:
        choices = [9, 12, 20, 21]
    elif age < 45:
        choices = [8, 12, 18, 20]
    else:
        choices = [7, 11, 17, 19]

    occupation_lower = occupation_text.lower()
    if any(term in occupation_lower for term in ["nurse", "doctor", "hospital", "shift"]):
        choices = [7, 15, 21]
    elif any(term in occupation_lower for term in ["student", "intern"]):
        choices = [12, 19, 21]

    return rng.choice(choices)


def _nemotron_row_to_persona(row: Dict[str, object], seed: int, idx: int) -> AudiencePersona:
    """Convert Nemotron dataset row into campaign simulation persona."""
    persona_id = str(row.get("uuid") or f"hf_{idx:06d}")
    row_seed = int(
        hashlib.sha256(f"{persona_id}|{seed}".encode("utf-8")).hexdigest()[:16],
        16,
    )
    rng = random.Random(row_seed)

    professional = str(row.get("professional_persona", ""))
    sports = str(row.get("sports_persona", ""))
    arts = str(row.get("arts_persona", ""))
    travel = str(row.get("travel_persona", ""))
    culinary = str(row.get("culinary_persona", ""))
    persona = str(row.get("persona", ""))
    occupation = str(row.get("occupation", ""))
    education_level = str(row.get("education_level", ""))

    text_blob = " ".join(
        [
            professional,
            sports,
            arts,
            travel,
            culinary,
            persona,
            str(row.get("hobbies_and_interests", "")),
            str(row.get("hobbies_and_interests_list", "")),
            occupation,
            education_level,
            str(row.get("career_goals_and_ambitions", "")),
        ]
    ).lower()

    style_preference = _infer_style_preference(text_blob, rng=rng)
    income_band = _infer_income_band(occupation=occupation, education_level=education_level)

    travel_terms = [
        "travel",
        "trip",
        "flight",
        "airport",
        "journey",
        "vacation",
        "passport",
        "nomad",
        "explore",
    ]
    travel_hits = sum(text_blob.count(term) for term in travel_terms)
    travel_intensity = _clamp_int(1 + travel_hits + rng.randint(0, 2), 1, 14)

    age = int(row.get("age", 34) or 34)
    preferred_send_hour = _infer_preferred_send_hour(
        age=age,
        occupation_text=occupation,
        rng=rng,
    )

    open_base = 0.19 + 0.06 * (travel_intensity / 14)
    open_base += 0.05 if income_band == "premium" else 0.0
    open_base += rng.uniform(-0.04, 0.04)

    click_base = 0.055 + 0.04 * (style_preference in {"bold", "playful"})
    click_base += 0.02 if income_band == "premium" else 0.0
    click_base += rng.uniform(-0.02, 0.02)

    purchase_base = 0.01 + 0.015 * (travel_intensity / 14)
    purchase_base += 0.022 if income_band == "premium" else 0.0
    purchase_base += rng.uniform(-0.01, 0.01)

    return AudiencePersona(
        persona_id=persona_id,
        segment=(occupation or professional or "general_consumer").strip().lower().replace(" ", "_"),
        style_preference=style_preference,
        travel_intensity=travel_intensity,
        income_band=income_band,
        preferred_send_hour=preferred_send_hour,
        open_base=round(clamp(open_base, 0.08, 0.65), 4),
        click_base=round(clamp(click_base, 0.02, 0.35), 4),
        purchase_base=round(clamp(purchase_base, 0.003, 0.12), 4),
        persona_source="hf_nemotron",
    )


def load_hf_persona_dataset(
    dataset_name: str,
    split: str,
    size: int,
    seed: int,
    token: str | None,
) -> List[AudiencePersona]:
    """Load and map personas from Hugging Face Nemotron dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("`datasets` package is required for HF persona loading.") from exc

    kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if token:
        kwargs["token"] = token

    try:
        dataset_stream = load_dataset(**kwargs)
    except TypeError:
        # Compatibility with older `datasets` argument naming.
        kwargs.pop("token", None)
        kwargs["use_auth_token"] = token
        dataset_stream = load_dataset(**kwargs)

    try:
        dataset_stream = dataset_stream.shuffle(seed=seed, buffer_size=min(5000, max(1000, size * 3)))
    except Exception:
        # Some iterable backends may not support shuffle; keep original order.
        pass

    personas: List[AudiencePersona] = []
    for idx, row in enumerate(dataset_stream):
        personas.append(_nemotron_row_to_persona(row=row, seed=seed, idx=idx))
        if len(personas) >= size:
            break
    if not personas:
        raise RuntimeError("No personas loaded from HF dataset.")
    return personas


def save_persona_dataset(personas: List[AudiencePersona], dataset_path: Path) -> None:
    """Persist mapped persona dataset as CSV."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "persona_id",
                "segment",
                "style_preference",
                "travel_intensity",
                "income_band",
                "preferred_send_hour",
                "open_base",
                "click_base",
                "purchase_base",
                "persona_source",
            ],
        )
        writer.writeheader()
        for persona in personas:
            writer.writerow(
                {
                    "persona_id": persona.persona_id,
                    "segment": persona.segment,
                    "style_preference": persona.style_preference,
                    "travel_intensity": persona.travel_intensity,
                    "income_band": persona.income_band,
                    "preferred_send_hour": persona.preferred_send_hour,
                    "open_base": persona.open_base,
                    "click_base": persona.click_base,
                    "purchase_base": persona.purchase_base,
                    "persona_source": persona.persona_source,
                }
            )


def load_or_create_persona_dataset(
    dataset_path: Path,
    size: int,
    seed: int,
    source: str,
    hf_dataset_name: str,
    hf_split: str,
    refresh: bool,
    allow_synthetic_fallback: bool,
) -> List[AudiencePersona]:
    """Load existing persona dataset or create from requested source."""
    if dataset_path.exists() and not refresh:
        return load_persona_dataset(dataset_path)

    if source == "hf":
        try:
            personas = load_hf_persona_dataset(
                dataset_name=hf_dataset_name,
                split=hf_split,
                size=size,
                seed=seed,
                token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            )
        except Exception as exc:
            if not allow_synthetic_fallback:
                raise RuntimeError(
                    "Failed to load personas from Hugging Face dataset. "
                    "Use --allow-synthetic-fallback to continue with synthetic personas."
                ) from exc
            print(f"Warning: HF dataset load failed ({exc}); using synthetic personas.")
            personas = generate_persona_dataset(size=size, seed=seed)
    else:
        personas = generate_persona_dataset(size=size, seed=seed)

    save_persona_dataset(personas, dataset_path=dataset_path)
    return personas


def load_persona_dataset(dataset_path: Path) -> List[AudiencePersona]:
    """Load persona dataset CSV."""
    personas: List[AudiencePersona] = []
    with dataset_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            personas.append(
                AudiencePersona(
                    persona_id=row["persona_id"],
                    segment=row["segment"],
                    style_preference=row["style_preference"],
                    travel_intensity=int(row["travel_intensity"]),
                    income_band=row["income_band"],
                    preferred_send_hour=int(row["preferred_send_hour"]),
                    open_base=float(row["open_base"]),
                    click_base=float(row["click_base"]),
                    purchase_base=float(row["purchase_base"]),
                    persona_source=row.get("persona_source", "unknown"),
                )
            )
    return personas


def build_campaign_arms(
    brand_name: str,
    target_audience: str,
    send_hours: List[int],
) -> List[CampaignArm]:
    """Generate copy with OpenEnv and expand into variant x send-time arms."""
    env = ClothingBrandCtrEnvironment()
    env.reset()
    marketer = TenXMarketerJudge()

    arms: List[CampaignArm] = []
    for variant in default_variants():
        action = ClothingBrandCtrAction(
            brand_name=brand_name,
            target_audience=target_audience,
            brand_voice=variant.brand_voice,
            key_value_prop=variant.key_value_prop,
            call_to_action=variant.call_to_action,
            metadata={"variant_name": variant.name},
        )
        observation = env.step(action)

        for hour in send_hours:
            arm = CampaignArm(
                arm_id=f"{variant.name.replace(' ', '_').lower()}__{hour:02d}00",
                variant_name=variant.name,
                brand_voice=variant.brand_voice,
                send_hour=hour,
                key_value_prop=variant.key_value_prop,
                call_to_action=variant.call_to_action,
                subject_line=observation.subject_line,
                preview_text=observation.preview_text,
                email_copy=observation.email_copy,
                word_count=observation.word_count,
                validation=observation.validation,
                ctr_proxy_score=observation.ctr_proxy_score,
                generation_source=observation.metadata.get(
                    "generation_source",
                    "unknown",
                ),
                marketer_score=0.0,
                marketer_rationale="",
            )
            score, rationale = marketer.score(arm=arm, brand_name=brand_name)
            arm.marketer_score = score
            arm.marketer_rationale = rationale
            arms.append(arm)
    return arms


def simulate_arm(
    arm: CampaignArm,
    personas: List[AudiencePersona],
    seed: int,
    capture_outcomes: bool = False,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Simulate one campaign arm across all personas."""
    rng = random.Random(seed)

    sent = len(personas)
    opens = clicks = purchases = 0
    outcomes: List[Dict[str, object]] = []

    for persona in personas:
        open_prob = calc_open_probability(arm=arm, persona=persona)
        opened = rng.random() < open_prob

        click_prob = calc_click_probability(arm=arm, persona=persona, opened=opened)
        clicked = opened and rng.random() < click_prob

        purchase_prob = calc_purchase_probability(
            arm=arm,
            persona=persona,
            clicked=clicked,
        )
        purchased = clicked and rng.random() < purchase_prob

        opens += int(opened)
        clicks += int(clicked)
        purchases += int(purchased)

        if capture_outcomes:
            outcomes.append(
                {
                    "persona_id": persona.persona_id,
                    "segment": persona.segment,
                    "persona_source": persona.persona_source,
                    "style_preference": persona.style_preference,
                    "preferred_send_hour": persona.preferred_send_hour,
                    "opened": int(opened),
                    "clicked": int(clicked),
                    "purchased": int(purchased),
                }
            )

    open_rate = opens / sent if sent else 0.0
    ctr = clicks / sent if sent else 0.0
    click_to_open_rate = clicks / opens if opens else 0.0
    purchase_rate = purchases / sent if sent else 0.0
    composite_score = (0.2 * open_rate) + (0.3 * ctr) + (0.5 * purchase_rate)

    metrics = {
        "arm_id": arm.arm_id,
        "variant_name": arm.variant_name,
        "brand_voice": arm.brand_voice,
        "send_hour": arm.send_hour,
        "generation_source": arm.generation_source,
        "marketer_score": arm.marketer_score,
        "marketer_rationale": arm.marketer_rationale,
        "subject_line": arm.subject_line,
        "sent": sent,
        "opens": opens,
        "clicks": clicks,
        "purchases": purchases,
        "open_rate": round(open_rate, 4),
        "ctr": round(ctr, 4),
        "click_to_open_rate": round(click_to_open_rate, 4),
        "purchase_rate": round(purchase_rate, 4),
        "composite_score": round(composite_score, 4),
    }
    return metrics, outcomes


def calc_open_probability(arm: CampaignArm, persona: AudiencePersona) -> float:
    """Estimate probability that persona opens this email."""
    alignment = send_time_alignment(arm.send_hour, persona.preferred_send_hour)
    style_match = 1.0 if arm.brand_voice == persona.style_preference else 0.0
    travel_signal = 1.0 if "travel" in (arm.preview_text + arm.email_copy).lower() else 0.0

    prob = persona.open_base
    prob += 0.16 * ((arm.marketer_score / 100.0) - 0.5)
    prob += 0.10 * (arm.ctr_proxy_score - 0.5)
    prob += 0.09 * alignment
    prob += 0.05 * style_match
    prob += 0.03 * travel_signal * min(persona.travel_intensity / 10.0, 1.0)
    return clamp(prob, 0.01, 0.95)


def calc_click_probability(
    arm: CampaignArm,
    persona: AudiencePersona,
    opened: bool,
) -> float:
    """Estimate probability that persona clicks after open."""
    if not opened:
        return 0.0
    style_match = 1.0 if arm.brand_voice == persona.style_preference else 0.0

    prob = persona.click_base
    prob += 0.15 * ((arm.marketer_score / 100.0) - 0.5)
    prob += 0.14 * (arm.ctr_proxy_score - 0.5)
    prob += 0.05 if arm.validation.get("body_has_cta") else -0.02
    prob += 0.02 if arm.validation.get("preview_length_ok") else -0.01
    prob += 0.03 * style_match
    return clamp(prob, 0.005, 0.85)


def calc_purchase_probability(
    arm: CampaignArm,
    persona: AudiencePersona,
    clicked: bool,
) -> float:
    """Estimate probability that persona purchases after click."""
    if not clicked:
        return 0.0
    income_boost = {"premium": 0.05, "mid": 0.02}.get(persona.income_band, 0.0)
    style_match = 1.0 if arm.brand_voice == persona.style_preference else 0.0
    travel_signal = 1.0 if "travel" in arm.email_copy.lower() else 0.0

    prob = persona.purchase_base
    prob += 0.10 * ((arm.marketer_score / 100.0) - 0.5)
    prob += 0.05 * style_match
    prob += 0.02 * travel_signal * min(persona.travel_intensity / 10.0, 1.0)
    prob += income_boost
    return clamp(prob, 0.002, 0.55)


def send_time_alignment(send_hour: int, preferred_hour: int) -> float:
    """Return alignment score in [-1, 1] based on send-hour distance."""
    diff = abs(send_hour - preferred_hour) % 24
    wrapped = min(diff, 24 - diff)
    return (1.0 - (wrapped / 12.0)) * 2.0 - 1.0


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to [low, high]."""
    return max(low, min(high, value))


def two_proportion_p_value(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
) -> float:
    """Two-sided p-value for difference between two Bernoulli proportions."""
    if total_a <= 0 or total_b <= 0:
        return 1.0
    p_a = success_a / total_a
    p_b = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    variance = pooled * (1 - pooled) * ((1 / total_a) + (1 / total_b))
    if variance <= 0:
        return 1.0
    z = (p_a - p_b) / math.sqrt(variance)
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    return max(0.0, min(1.0, 2 * (1 - cdf)))


def save_arm_metrics(results: List[Dict[str, object]], output_path: Path) -> None:
    """Write arm-level simulation metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "arm_id",
                "variant_name",
                "brand_voice",
                "send_hour",
                "generation_source",
                "marketer_score",
                "open_rate",
                "ctr",
                "click_to_open_rate",
                "purchase_rate",
                "composite_score",
                "opens",
                "clicks",
                "purchases",
                "sent",
                "subject_line",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "arm_id": row["arm_id"],
                    "variant_name": row["variant_name"],
                    "brand_voice": row["brand_voice"],
                    "send_hour": row["send_hour"],
                    "generation_source": row["generation_source"],
                    "marketer_score": row["marketer_score"],
                    "open_rate": row["open_rate"],
                    "ctr": row["ctr"],
                    "click_to_open_rate": row["click_to_open_rate"],
                    "purchase_rate": row["purchase_rate"],
                    "composite_score": row["composite_score"],
                    "opens": row["opens"],
                    "clicks": row["clicks"],
                    "purchases": row["purchases"],
                    "sent": row["sent"],
                    "subject_line": row["subject_line"],
                }
            )


def save_top_arm_outcomes(outcomes: List[Dict[str, object]], output_path: Path) -> None:
    """Write per-persona outcomes for the winning arm."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "persona_id",
                "segment",
                "persona_source",
                "style_preference",
                "preferred_send_hour",
                "opened",
                "clicked",
                "purchased",
            ],
        )
        writer.writeheader()
        for row in outcomes:
            writer.writerow(row)


def print_results(
    ranked_results: List[Dict[str, object]],
    marketer_name: str,
    ml_score: float,
    ml_rationale: str,
) -> None:
    """Print top-line simulation leaderboard + judge outputs."""
    print("\nCampaign Arms Leaderboard")
    print("-" * 130)
    print(
        f"{'Rank':<5} {'Arm':<26} {'Score':<7} {'Open':<7} {'CTR':<7} {'Purchase':<9} "
        f"{'MktScore':<9} {'Source':<17} {'Send'}"
    )
    print("-" * 130)
    for rank, row in enumerate(ranked_results[:10], start=1):
        print(
            f"{rank:<5} {str(row['arm_id']):<26} {float(row['composite_score']):<7.3f} "
            f"{float(row['open_rate']):<7.3f} {float(row['ctr']):<7.3f} "
            f"{float(row['purchase_rate']):<9.3f} {float(row['marketer_score']):<9.1f} "
            f"{str(row['generation_source']):<17} {int(row['send_hour']):02d}:00"
        )

    top = ranked_results[0]
    print("\nWinning Campaign Arm")
    print("-" * 130)
    print(f"Arm: {top['arm_id']}")
    print(f"Subject: {top['subject_line']}")
    print(f"Open Rate: {top['open_rate']}")
    print(f"CTR: {top['ctr']}")
    print(f"Purchase Rate: {top['purchase_rate']}")
    print(f"{marketer_name} score: {top['marketer_score']} ({top['marketer_rationale']})")

    print("\nJudge: ML Engineer (A/B Testing)")
    print("-" * 130)
    print(f"Experiment score: {ml_score}/100")
    print(ml_rationale)


def parse_send_hours(raw: str) -> List[int]:
    """Parse comma-separated hour list into integers [0,23]."""
    hours: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        hour = int(token)
        if hour < 0 or hour > 23:
            raise ValueError(f"Invalid send hour: {hour}")
        hours.append(hour)
    if not hours:
        raise ValueError("At least one send hour is required.")
    return sorted(set(hours))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate brand email campaign performance with judge personas.",
    )
    parser.add_argument("--brand-name", default="AIRPORT CLUB")
    parser.add_argument("--target-audience", default="fellow traveler")
    parser.add_argument(
        "--send-hours",
        default="8,12,18,21",
        help="Comma-separated send hours (0-23).",
    )
    parser.add_argument(
        "--persona-dataset",
        default="outputs/evals/persona_dataset.csv",
        help="CSV path for audience persona dataset.",
    )
    parser.add_argument(
        "--persona-source",
        choices=["hf", "synthetic"],
        default="hf",
        help="Persona source. Default uses Hugging Face Nemotron dataset.",
    )
    parser.add_argument(
        "--hf-dataset-name",
        default=os.getenv("HF_PERSONA_DATASET", "nvidia/Nemotron-Personas-USA"),
        help="Hugging Face dataset id used when --persona-source=hf.",
    )
    parser.add_argument(
        "--hf-split",
        default=os.getenv("HF_PERSONA_SPLIT", "train"),
        help="Hugging Face split used when --persona-source=hf.",
    )
    parser.add_argument(
        "--refresh-personas",
        action="store_true",
        help="Rebuild persona CSV from source even if cached CSV exists.",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow fallback to synthetic personas if HF dataset load fails.",
    )
    parser.add_argument(
        "--persona-count",
        type=int,
        default=1200,
        help="How many personas to generate when dataset does not exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--results-csv",
        default="outputs/evals/brand_campaign_arm_results.csv",
        help="Output CSV path for arm-level metrics.",
    )
    parser.add_argument(
        "--top-arm-outcomes-csv",
        default="outputs/evals/top_arm_persona_outcomes.csv",
        help="Output CSV path for winner-arm per-persona outcomes.",
    )
    args = parser.parse_args()

    send_hours = parse_send_hours(args.send_hours)
    persona_path = Path(args.persona_dataset)
    personas = load_or_create_persona_dataset(
        dataset_path=persona_path,
        size=args.persona_count,
        seed=args.seed,
        source=args.persona_source,
        hf_dataset_name=args.hf_dataset_name,
        hf_split=args.hf_split,
        refresh=args.refresh_personas,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )

    arms = build_campaign_arms(
        brand_name=args.brand_name,
        target_audience=args.target_audience,
        send_hours=send_hours,
    )

    metrics: List[Dict[str, object]] = []
    for idx, arm in enumerate(arms):
        arm_seed = args.seed + (idx * 7919)
        arm_metrics, _ = simulate_arm(
            arm=arm,
            personas=personas,
            seed=arm_seed,
            capture_outcomes=False,
        )
        metrics.append(arm_metrics)

    ranked_results = sorted(
        metrics,
        key=lambda row: (
            float(row["composite_score"]),
            float(row["purchase_rate"]),
            float(row["ctr"]),
            float(row["open_rate"]),
        ),
        reverse=True,
    )

    # Re-run winning arm with deterministic seed to export per-persona outcomes.
    best_arm_id = ranked_results[0]["arm_id"]
    best_arm = next(arm for arm in arms if arm.arm_id == best_arm_id)
    _, outcomes = simulate_arm(
        arm=best_arm,
        personas=personas,
        seed=args.seed + 999999,
        capture_outcomes=True,
    )

    ml_judge = MLEngineerJudge()
    ml_score, ml_rationale = ml_judge.score_experiment(
        ranked_results=ranked_results,
        audience_size=len(personas),
    )

    save_arm_metrics(ranked_results, Path(args.results_csv))
    save_top_arm_outcomes(outcomes, Path(args.top_arm_outcomes_csv))
    print_results(
        ranked_results=ranked_results,
        marketer_name=TenXMarketerJudge.name,
        ml_score=ml_score,
        ml_rationale=ml_rationale,
    )
    print(f"\nPersona source: {personas[0].persona_source if personas else 'unknown'}")
    print(f"\nPersona dataset: {persona_path}")
    print(f"Arm results: {args.results_csv}")
    print(f"Top-arm outcomes: {args.top_arm_outcomes_csv}")


if __name__ == "__main__":
    main()
