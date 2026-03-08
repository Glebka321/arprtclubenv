"""Run a simple multi-step simulation to compare and rank email variants."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from clothing_brand_ctr_env import ClothingBrandCtrAction
from clothing_brand_ctr_env.server.clothing_brand_ctr_env_environment import (
    ClothingBrandCtrEnvironment,
)


@dataclass(frozen=True)
class EmailVariant:
    """Single email variant configuration."""

    name: str
    brand_voice: str
    key_value_prop: str
    call_to_action: str


def default_variants() -> List[EmailVariant]:
    """Default set of variants for quick comparisons."""
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


def run_simulation(
    brand_name: str,
    target_audience: str,
    variants: List[EmailVariant],
) -> List[Dict[str, object]]:
    """Evaluate variants over multiple steps and return ranked results."""
    env = ClothingBrandCtrEnvironment()
    env.reset()

    results: List[Dict[str, object]] = []
    for variant in variants:
        action = ClothingBrandCtrAction(
            brand_name=brand_name,
            target_audience=target_audience,
            brand_voice=variant.brand_voice,
            key_value_prop=variant.key_value_prop,
            call_to_action=variant.call_to_action,
            metadata={"variant_name": variant.name},
        )
        observation = env.step(action)

        passed_rules = sum(1 for passed in observation.validation.values() if passed)
        results.append(
            {
                "variant_name": variant.name,
                "brand_voice": variant.brand_voice,
                "ctr_proxy_score": observation.ctr_proxy_score,
                "validation_passed": observation.validation_passed,
                "passed_rules": passed_rules,
                "generation_source": observation.metadata.get(
                    "generation_source",
                    "unknown",
                ),
                "subject_line": observation.subject_line,
                "preview_text": observation.preview_text,
                "email_copy": observation.email_copy,
                "word_count": observation.word_count,
                "validation": observation.validation,
            }
        )

    return sorted(
        results,
        key=lambda item: (
            float(item["ctr_proxy_score"]),
            int(item["passed_rules"]),
            int(item["word_count"]),
        ),
        reverse=True,
    )


def print_rankings(rankings: List[Dict[str, object]]) -> None:
    """Print a compact leaderboard for quick comparison."""
    print("\nEmail Variant Leaderboard")
    print("-" * 116)
    print(
        f"{'Rank':<5} {'Variant':<16} {'Voice':<9} {'Score':<7} {'Checks':<7} {'Source':<18} {'Subject Line'}"
    )
    print("-" * 116)

    for rank, row in enumerate(rankings, start=1):
        subject = str(row["subject_line"])
        if len(subject) > 46:
            subject = subject[:43] + "..."
        checks = f"{row['passed_rules']}/{len(row['validation'])}"
        print(
            f"{rank:<5} {str(row['variant_name']):<16} {str(row['brand_voice']):<9} "
            f"{float(row['ctr_proxy_score']):<7.2f} {checks:<7} "
            f"{str(row['generation_source']):<18} {subject}"
        )

    best = rankings[0]
    print("\nTop Variant")
    print("-" * 96)
    print(f"Name: {best['variant_name']}")
    print(f"Voice: {best['brand_voice']}")
    print(f"Score: {best['ctr_proxy_score']}")
    print(f"Validation Passed: {best['validation_passed']}")
    print(f"Generation Source: {best['generation_source']}")
    print(f"Preview: {best['preview_text']}")
    print("\nEmail Copy:\n")
    print(best["email_copy"])


def save_rankings_csv(rankings: List[Dict[str, object]], output_path: Path) -> None:
    """Persist ranking output for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "variant_name",
                "brand_voice",
                "ctr_proxy_score",
                "validation_passed",
                "passed_rules",
                "generation_source",
                "word_count",
                "subject_line",
                "preview_text",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rankings, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "variant_name": row["variant_name"],
                    "brand_voice": row["brand_voice"],
                    "ctr_proxy_score": row["ctr_proxy_score"],
                    "validation_passed": row["validation_passed"],
                    "passed_rules": row["passed_rules"],
                    "generation_source": row["generation_source"],
                    "word_count": row["word_count"],
                    "subject_line": row["subject_line"],
                    "preview_text": row["preview_text"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-step simulation and rank email campaign variants."
    )
    parser.add_argument("--brand-name", default="AIRPORT CLUB")
    parser.add_argument(
        "--target-audience",
        default="fellow traveler",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/evals/email_variant_rankings.csv",
        help="Path to save CSV ranking output",
    )
    parser.add_argument(
        "--hf-model-id",
        default=None,
        help="Optional override for HF model id (e.g., deepseek-ai/DeepSeek-V3).",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable HF generation and use deterministic template fallback.",
    )
    args = parser.parse_args()

    if args.hf_model_id:
        os.environ["HF_MODEL_ID"] = args.hf_model_id
    if args.disable_llm:
        os.environ["USE_HF_LLM"] = "false"

    rankings = run_simulation(
        brand_name=args.brand_name,
        target_audience=args.target_audience,
        variants=default_variants(),
    )
    print_rankings(rankings)
    save_rankings_csv(rankings, Path(args.output_csv))
    print(f"\nSaved rankings to: {args.output_csv}")


if __name__ == "__main__":
    main()
