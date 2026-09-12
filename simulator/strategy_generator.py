import numpy as np
def generate_strategies():

    strategies = [
    # One-stop

    ["SOFT", "HARD"],
    ["MEDIUM", "HARD"],
    ["HARD", "MEDIUM"],

    # Two-stop

    ["SOFT", "HARD", "HARD"],
    ["MEDIUM", "HARD", "HARD"],

    ["SOFT", "SOFT", "HARD"],
    ["MEDIUM", "MEDIUM", "HARD"],

    ["SOFT", "HARD", "SOFT"],
    ["MEDIUM", "HARD", "SOFT"],

    ["SOFT", "MEDIUM", "HARD"],
    ["MEDIUM", "SOFT", "HARD"]
]

    return strategies


if __name__ == "__main__":
    strategies = generate_strategies()
    print("Generated Strategies:")
    for strategy in strategies:
        print(f" - {strategy}")

def evaluate_strategies(
    simulator,
    race_context,
    top_n=5
):

    strategies = generate_strategies()

    results = []

    for strategy in strategies:

        result = simulator.simulate_strategy(
            strategy=strategy,
            race_context=race_context
        )

        balance_score = float(
            round(
                np.std(result["predicted_stints"]),
                2
            )
        )

        result["balance_score"] = balance_score

        results.append(result)

    results.sort(
        key=lambda x: (
            x["balance_score"],
            x["predicted_stints"][-1]
        )
    )

    return results[:top_n]


    