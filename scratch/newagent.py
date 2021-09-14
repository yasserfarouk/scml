import numpy as np

from scml import DecentralizingAgent
from scml import TradePredictionStrategy


class MyPredictor(TradePredictionStrategy):
    def trade_prediction_init(self):
        """Assume that you can expect full production and selling everything
        you can produce"""
        self.expected_outputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)
        self.expected_inputs = self.awi.n_lines * np.ones(self.awi.n_steps, dtype=int)


class MyAgent(MyPredictor, DecentralizingAgent):
    pass


def main():
    from scml.scml2020 import anac2020_std

    results = anac2020_std(
        (DecentralizingAgent, MyAgent), n_runs_per_world=1, n_steps=20, verbose=True
    )
    print(str(results))


if __name__ == "__main__":
    main()
