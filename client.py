# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Clothing Brand Ctr Env Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import ClothingBrandCtrAction, ClothingBrandCtrObservation


class ClothingBrandCtrEnv(
    EnvClient[ClothingBrandCtrAction, ClothingBrandCtrObservation, State]
):
    """
    Client for the Clothing Brand Ctr Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ClothingBrandCtrEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.preview_text)
        ...
        ...     result = client.step(
        ...         ClothingBrandCtrAction(
        ...             brand_name="AIRPORT CLUB",
        ...             target_audience="style-focused people that travel",
        ...             brand_voice="bold, easy-going, fun, wholesome",
        ...             key_value_prop="premium t-shirts when traveling by plane",
        ...             call_to_action="Shop the launch collection now",
        ...         )
        ...     )
        ...     print(result.observation.validation_passed)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ClothingBrandCtrEnv.from_docker_image("clothing_brand_ctr_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(
        ...         ClothingBrandCtrAction(
        ...             brand_name="ARPRT CLUB",
        ...             target_audience="traveler",
        ...             brand_voice="bold",
        ...             key_value_prop="premium essentials with runway-level polish",
        ...             call_to_action="Shop the launch collection now",
        ...         )
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ClothingBrandCtrAction) -> Dict:
        """
        Convert ClothingBrandCtrAction to JSON payload for step message.

        Args:
            action: ClothingBrandCtrAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "brand_name": action.brand_name,
            "target_audience": action.target_audience,
            "brand_voice": action.brand_voice,
            "key_value_prop": action.key_value_prop,
            "call_to_action": action.call_to_action,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ClothingBrandCtrObservation]:
        """
        Parse server response into StepResult[ClothingBrandCtrObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ClothingBrandCtrObservation
        """
        obs_data = payload.get("observation", {})
        observation = ClothingBrandCtrObservation(
            subject_line=obs_data.get("subject_line", ""),
            preview_text=obs_data.get("preview_text", ""),
            email_copy=obs_data.get("email_copy", ""),
            word_count=obs_data.get("word_count", 0),
            validation=obs_data.get("validation", {}),
            validation_passed=obs_data.get("validation_passed", False),
            ctr_proxy_score=obs_data.get("ctr_proxy_score", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
