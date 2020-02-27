import time
import sys

import pandas as pd

MY_PATH = "/".join(__file__.split("/")[:-1])

from . import pydata


# Generate all uncertainty models
if __name__ == "__main__":
    print(f"Trying to generate quantity and price uncertainty models...")
    t0 = time.time()
    for num_intermediate_products in [3]:
        print(f"\t num_intermediate_products = {num_intermediate_products}")
        game_logs = pd.concat(
            [
                pd.read_csv(
                    f"{MY_PATH}/mylogs/num_intermediate_products_{num_intermediate_products}_"
                    f"production_cost_{cost}_n_steps_100_log.csv"
                )
                for cost in range(1, 3)
            ]
        )
        game_data = pydata.get_raw_quantity_uncertainty_model(game_logs)
        pydata.save_json_uncertainty_model(
            json_file_name=f"{MY_PATH}/dict_qtty_num_intermediate_products"
            f"_{num_intermediate_products}",
            raw_uncertainty_model=game_data,
            products=game_data["product"].unique(),
        )
        pydata.save_json_price_data(
            json_file_name=f"{MY_PATH}/dict_price_num_intermediate_products"
            f"_{num_intermediate_products}",
            the_game_logs=game_logs,
        )
    print(
        f"Done generating quantity and price uncertainty models, took {time.time() - t0}"
    )
