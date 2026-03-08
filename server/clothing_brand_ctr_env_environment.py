# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Clothing Brand Ctr Env Environment Implementation.

Simple environment that generates launch email copy and validates quality checks.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ClothingBrandCtrAction, ClothingBrandCtrObservation
except ImportError:  # pragma: no cover - supports direct server.app imports
    from models import ClothingBrandCtrAction, ClothingBrandCtrObservation

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover
    InferenceClient = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


class ClothingBrandCtrEnvironment(Environment):
    """
    Generate introductory email campaign copy for a clothing brand.

    The environment returns copy + validation checks so an agent can optimize
    prompts/actions for better CTR proxy quality over multiple steps.

    Example:
        >>> env = ClothingBrandCtrEnvironment()
        >>> obs = env.reset()
        >>> print(obs.preview_text)  # "Environment ready for email-copy simulation."
        >>>
        >>> obs = env.step(
        ...     ClothingBrandCtrAction(
        ...         brand_name="ARPRT CLUB",
        ...         target_audience="urban professionals",
        ...         brand_voice="bold",
        ...         key_value_prop="clothing brand for people that travel",
        ...         call_to_action="Shop the first drop",
        ...     )
        ... )
        >>> print(obs.subject_line)
        >>> print(obs.validation_passed)
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the clothing_brand_ctr_env environment."""
        if load_dotenv is not None:
            load_dotenv()

        self._project_root = Path(__file__).resolve().parents[1]
        self._brand_persona_json = self._load_brand_persona_json(
            os.getenv("BRAND_PERSONA_FILE", "config/brand_persona.json")
        )
        self._brand_copy_instructions = (
            os.getenv("BRAND_COPY_INSTRUCTIONS", "").strip()
            or self._load_brand_copy_instructions(
                os.getenv("BRAND_INSTRUCTIONS_FILE", "config/brand_instructions.txt")
            )
            or (
                "Keep copy consistent with the brand persona. "
                "Keep positioning short, playful, and quirky."
            )
        )

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._hf_model_id = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-V3")
        self._use_hf_llm = os.getenv("USE_HF_LLM", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self._hf_client = None
        if self._use_hf_llm and self._hf_token and InferenceClient is not None:
            self._hf_client = InferenceClient(api_key=self._hf_token)

    def reset(self) -> ClothingBrandCtrObservation:
        """
        Reset the environment.

        Returns:
            ClothingBrandCtrObservation with initial empty copy state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return ClothingBrandCtrObservation(
            subject_line="",
            preview_text="Environment ready for email-copy simulation.",
            email_copy="",
            word_count=0,
            validation={},
            validation_passed=False,
            ctr_proxy_score=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: ClothingBrandCtrAction) -> ClothingBrandCtrObservation:  # type: ignore[override]
        """
        Generate intro email copy and run quality validation.

        Args:
            action: Email generation request

        Returns:
            ClothingBrandCtrObservation with generated copy and validation
        """
        self._state.step_count += 1

        subject_line, preview_text, email_copy, generation_source = self._generate_email_copy(
            action
        )
        word_count = len(email_copy.split())
        validation = self._validate_email_copy(
            action=action,
            subject_line=subject_line,
            preview_text=preview_text,
            email_copy=email_copy,
            word_count=word_count,
        )
        ctr_proxy_score = self._compute_ctr_proxy_score(validation)
        validation_passed = all(validation.values())

        return ClothingBrandCtrObservation(
            subject_line=subject_line,
            preview_text=preview_text,
            email_copy=email_copy,
            word_count=word_count,
            validation=validation,
            validation_passed=validation_passed,
            ctr_proxy_score=ctr_proxy_score,
            done=False,
            reward=ctr_proxy_score,
            metadata={
                "step": self._state.step_count,
                "brand_name": action.brand_name,
                "brand_voice": action.brand_voice,
                "generation_source": generation_source,
                "hf_model_id": self._hf_model_id if generation_source == "hf_llm" else "",
            },
        )

    def _generate_email_copy(
        self,
        action: ClothingBrandCtrAction,
    ) -> tuple[str, str, str, str]:
        """Generate copy via HF LLM when available, else fallback to template."""
        llm_output = self._generate_email_copy_with_hf(action)
        if llm_output is not None:
            return llm_output[0], llm_output[1], llm_output[2], "hf_llm"

        subject_line, preview_text, email_copy = self._generate_email_copy_template(action)
        return subject_line, preview_text, email_copy, "template_fallback"

    def _generate_email_copy_template(
        self,
        action: ClothingBrandCtrAction,
    ) -> tuple[str, str, str]:
        """Create deterministic fallback copy from the action fields."""
        voice_openers = {
            "minimal": "clean wardrobe staples designed for bringing to the airport",
            "bold": "premium t-shirts for standing out when you travel",
            "playful": "the only t-shirt you need for a 14 hour flight",
            "luxury": "soft luxury items for traveling",
        }
        opener = voice_openers[action.brand_voice]

        subject_line = f"Meet {action.brand_name}: Your New T-Shirt For Travel"
        preview_text = (
            f"{action.key_value_prop}. Built for {action.target_audience}. "
            f"{action.call_to_action}."
        )

        email_copy = (
            f"Hi {action.target_audience},\n\n"
            f"We're excited to introduce {action.brand_name}.\n"
            f"{opener}\n\n"
            f"At {action.brand_name}, our focus is simple: {action.key_value_prop}.\n"
            f"Every piece is designed so you can look put together without overthinking it.\n\n"
            f"Ready to see the collection? {action.call_to_action}.\n\n"
            f"See you at the lounge,\n"
            f"Team {action.brand_name}"
        )

        return subject_line, preview_text, email_copy

    def _generate_email_copy_with_hf(
        self,
        action: ClothingBrandCtrAction,
    ) -> Optional[tuple[str, str, str]]:
        """Generate copy via Hugging Face Inference API using a DeepSeek model."""
        if not self._use_hf_llm or self._hf_client is None:
            return None

        system_prompt = (
            "You are an expert lifecycle email marketer for fashion brands. "
            "Return only strict JSON with keys: subject_line, preview_text, email_copy."
        )
        user_prompt = (
            "Create intro campaign email copy using these inputs.\n"
            f"brand_name: {action.brand_name}\n"
            f"target_audience: {action.target_audience}\n"
            f"brand_voice: {action.brand_voice}\n"
            f"key_value_prop: {action.key_value_prop}\n"
            f"call_to_action: {action.call_to_action}\n\n"
            f"brand_persona_json: {self._brand_persona_json}\n"
            f"additional_brand_instructions: {self._brand_copy_instructions}\n\n"
            "Rules:\n"
            "1) Subject line 24-70 chars and include brand_name.\n"
            "2) Preview text 45-140 chars.\n"
            "3) Email body 60-170 words.\n"
            "4) Body must include brand_name, key_value_prop, and call_to_action.\n"
            "5) Keep travel context and no markdown.\n"
            "6) Ensure voice and references are consistent with brand_persona_json.\n"
            "7) Positioning language must be short and quirky."
        )

        try:
            response = self._hf_client.chat.completions.create(
                model=self._hf_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=500,
            )
            content = self._extract_hf_content(response)
            if not content:
                return None
            parsed = self._parse_llm_json(content)
            if parsed is None:
                return None
            subject_line = str(parsed.get("subject_line", "")).strip()
            preview_text = str(parsed.get("preview_text", "")).strip()
            email_copy = str(parsed.get("email_copy", "")).strip()

            if not subject_line or not preview_text or not email_copy:
                return None
            return subject_line, preview_text, email_copy
        except Exception:
            return None

    def _extract_hf_content(self, response: object) -> str:
        """Extract text content from HF chat completion response."""
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message")

        content = ""
        if message is not None:
            content = getattr(message, "content", "")
            if not content and isinstance(message, dict):
                content = message.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(str(item))
            return "".join(text_parts).strip()

        return str(content).strip()

    def _parse_llm_json(self, content: str) -> Optional[Dict[str, object]]:
        """Parse JSON response, tolerating fenced blocks."""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _load_brand_persona_json(self, persona_path: str) -> str:
        """Load persona JSON from file and normalize to a compact JSON string."""
        path = Path(persona_path)
        if not path.is_absolute():
            path = self._project_root / path
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return "{}"
        return json.dumps(parsed, separators=(",", ":"))

    def _load_brand_copy_instructions(self, instructions_path: str) -> str:
        """Load additional brand instruction text from file."""
        path = Path(instructions_path)
        if not path.is_absolute():
            path = self._project_root / path
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _validate_email_copy(
        self,
        action: ClothingBrandCtrAction,
        subject_line: str,
        preview_text: str,
        email_copy: str,
        word_count: int,
    ) -> Dict[str, bool]:
        """Validate copy quality checks for basic launch-email best practices."""
        brand_lc = action.brand_name.lower()
        cta_lc = action.call_to_action.lower()
        value_prop_lc = action.key_value_prop.lower()

        return {
            "subject_has_brand": brand_lc in subject_line.lower(),
            "subject_length_ok": 24 <= len(subject_line) <= 70,
            "preview_length_ok": 45 <= len(preview_text) <= 140,
            "body_has_brand": brand_lc in email_copy.lower(),
            "body_mentions_value_prop": value_prop_lc in email_copy.lower(),
            "body_has_cta": cta_lc in email_copy.lower(),
            "body_word_count_ok": 60 <= word_count <= 180,
        }

    def _compute_ctr_proxy_score(self, validation: Dict[str, bool]) -> float:
        """Aggregate validation outcomes into a 0-1 proxy CTR score."""
        weights = {
            "subject_has_brand": 0.16,
            "subject_length_ok": 0.17,
            "preview_length_ok": 0.14,
            "body_has_brand": 0.12,
            "body_mentions_value_prop": 0.14,
            "body_has_cta": 0.17,
            "body_word_count_ok": 0.10,
        }
        score = sum(weights[key] for key, passed in validation.items() if passed)
        return round(score, 4)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
