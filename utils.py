import sys


def run_tests(agent, trials):
    results = []
    for _ in trials:
        result = agent.train_agent()
        results.append(result)

    return results


