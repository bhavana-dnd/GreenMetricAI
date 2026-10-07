from greenmetricai.mbpp import load_mbpp
from greenmetricai.features import extract_code_metrics
from greenmetricai.optimizer import (
    GeminiProvider,
    GreenMetricEnergyModel,
    optimize_mbpp_task,
)


TASK_ID = 2


def main():

    tasks = load_mbpp()

    task = next(
        task for task in tasks
        if int(task["task_id"]) == TASK_ID
    )

    print("=" * 60)
    print("GREENMETRICAI LLM OPTIMIZATION PILOT")
    print("=" * 60)

    print(f"Task ID: {TASK_ID}")
    print(f"Task description: {task['prompt']}")

    print("\nOriginal code:")
    print("-" * 60)
    print(task["code"])

    model = GreenMetricEnergyModel()
    provider = GeminiProvider()

    original_metrics = extract_code_metrics(
        task["code"]
    )

    original_energy = model.predict(
        original_metrics
    )

    print("\nOriginal predicted energy:")
    print(
        f"{original_energy:.12g} "
        "kWh/execution"
    )

    print("\nRunning Gemini optimization...")
    print("One Gemini call will be made.")

    result = optimize_mbpp_task(
        task=task,
        model=model,
        provider=provider,
        rounds=1,
        min_improvement=0.03,
        validation_timeout=30,
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print("\nFinal predicted energy:")
    print(
        f"{result['predicted_energy_kwh']:.12g} "
        "kWh/execution"
    )

    print("\nOptimized code:")
    print("-" * 60)
    print(result["optimized_code"])

    print("\nOptimization history:")
    print("-" * 60)

    for item in result["history"]:
        print(item)


if __name__ == "__main__":
    main()