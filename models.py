# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Clothing Brand Ctr Env Environment.

The clothing_brand_ctr_env environment generates and validates marketing copy.
"""

from typing import Dict, Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class ClothingBrandCtrAction(Action):
    """Action that asks the environment to generate launch email copy."""

    brand_name: str = Field(..., description="Brand name to feature in the email")
    target_audience: str = Field(
        ...,
        description="Who this campaign is aimed at",
    )
    brand_voice: Literal["minimal", "bold", "playful", "luxury"] = Field(
        default="bold",
        description="Tone style for generated copy",
    )
    key_value_prop: str = Field(
        ...,
        description="Main value proposition to communicate",
    )
    call_to_action: str = Field(
        ...,
        description="Primary CTA text to include in the email body",
    )


class ClothingBrandCtrObservation(Observation):
    """Observation with generated email copy and validation results."""

    subject_line: str = Field(default="", description="Generated subject line")
    preview_text: str = Field(default="", description="Generated preview/snippet text")
    email_copy: str = Field(default="", description="Generated full email body copy")
    word_count: int = Field(default=0, description="Word count for generated body")
    validation: Dict[str, bool] = Field(
        default_factory=dict,
        description="Rule-by-rule validation outcomes",
    )
    validation_passed: bool = Field(
        default=False,
        description="True when all validation checks pass",
    )
    ctr_proxy_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Simple proxy score (0-1) for likely CTR quality",
    )
