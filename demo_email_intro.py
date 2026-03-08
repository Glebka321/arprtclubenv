"""Small local smoke demo for intro-email copy generation + validation."""

from clothing_brand_ctr_env import ClothingBrandCtrAction
from clothing_brand_ctr_env.server.clothing_brand_ctr_env_environment import (
    ClothingBrandCtrEnvironment,
)


def main() -> None:
    env = ClothingBrandCtrEnvironment()
    env.reset()

    action = ClothingBrandCtrAction(
        brand_name="AIRPORT CLUB",
        target_audience="for working millenials and Gen Z that travel",
        brand_voice="bold, easy-going, wholesome",
        key_value_prop="quiet luxury items to where to the airport",
        call_to_action="Shop the launch collection now",
    )

    observation = env.step(action)

    print("Subject:", observation.subject_line)
    print("Preview:", observation.preview_text)
    print("CTR Proxy Score:", observation.ctr_proxy_score)
    print("Validation Passed:", observation.validation_passed)
    print("Validation:", observation.validation)
    print("\nEmail Copy:\n")
    print(observation.email_copy)

    # Minimal pass/fail check for this demo run.
    assert observation.validation_passed, "Validation failed for generated email copy."


if __name__ == "__main__":
    main()
