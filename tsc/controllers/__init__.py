"""
Controller registry.  make(method) -> Controller   (learning methods load models/<method>.*)
"""

from tsc.controllers.base import Controller

RULE_BASED = ["Fixed", "Webster", "Actuated", "SOTL", "MaxPressure"]
LEARNING = ["QLearning", "D3QN", "PPO", "PressLight", "FRAP"]
ALL_METHODS = RULE_BASED + LEARNING


def make(method: str) -> Controller:
    from tsc.controllers import rule_based as rb

    rule = {
        "Fixed": rb.FixedController,
        "Webster": rb.WebsterController,
        "Actuated": rb.ActuatedController,
        "SOTL": rb.SOTLController,
        "MaxPressure": rb.MaxPressureController,
    }
    if method in rule:
        return rule[method]()
    if method in LEARNING:
        from tsc.controllers.learning import load

        return load(method)
    raise ValueError(f"Unknown method '{method}'. Choose from {ALL_METHODS}")
