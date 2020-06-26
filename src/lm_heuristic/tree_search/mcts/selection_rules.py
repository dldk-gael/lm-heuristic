import math

from .counter_node import CounterNode


def standart_ucb(child: CounterNode, parent: CounterNode, k=1) -> float:
    # Compute vanilla UCB as proposed by
    # Levente Kocsis, Csaba Szepesvári. "Bandit based Monte-Carlo Planning"
    return (
        child.sum_rewards / child.count
        + math.sqrt(k * math.log(parent.count / child.count))
    )


def single_player_ucb(child: CounterNode, parent: CounterNode, c=1, d=100) -> float:
    # Compute UCB for single-player context as proposed by
    # Schadda, Winandsan, Taka, Uiterwijka. "Single-Player Monte-Carlo Tree Search for SameGame"
    return (
        child.sum_rewards / child.count
        + math.sqrt(c * math.log(parent.count / child.count))
        + math.sqrt(
            (child.sum_of_square_rewards - child.count * ((child.sum_rewards / child.count) ** 2) + d) / child.count
        )
    )
