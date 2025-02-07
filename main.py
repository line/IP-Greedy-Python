"""
MIT License

Copyright 2024 LY Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import random
import time
import csv
from src.data_input import Item, User, load_mf_data_from_file, load_item2vec_data_from_file, load_config, get_param_from_config
from src.greedy_functions import calculate_items_norm_mf, calculate_items_norm_i2v, make_key, calculate_scale, execute_greedy_reverse_search, ResultData

def record_user_time_data(user_id: int, record: list[list[float]], record_tmp: list[float]) -> None:
    """Records time vector data for a user.

    Args:
        user_id (int): The ID of the user.
        record (list[list[float]]): The list to store record data.
        record_tmp (list[float]): The temporary record data.
    """
    record.append([user_id] + record_tmp)

def export_record_to_csv(output_file_name: str, record: list[list[float]]) -> None:
    """Writes the record data to a CSV file.

    Args:
        output_file_name (str): The name of the output file.
        record (list[list[float]]): The record data to write.
    """
    with open(output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(record)

def main():
    """The main function that executes the algorithm."""
    # Parameter setup
    config = load_config("../parameter/config.json")
    if config:
        k = get_param_from_config(config, "k")
        lambda_value = get_param_from_config(config, "lambda")

        if isinstance(k, int):
            print(f"Parameter k: {k}")
        else:
            print("Parameter k not found or is not an integer in configuration.")

        if isinstance(lambda_value, float):
            print(f"Parameter lambda: {lambda_value}")
        else:
            print("Parameter lambda not found or is not a float in configuration.")
    else:
        print("Failed to load configuration.")

    items: list[Item] = []
    users: list[User] = []

    load_mf_data_from_file(users, items, input_id=3)  # Use input_id as 3 for MovieLens. 0:netflix, 1:amazon_M, 2:amazon_K, 3:MovieLens
    load_item2vec_data_from_file(items, input_id=3)

    calculate_items_norm_mf(items)
    calculate_items_norm_i2v(items)

    make_key(items)
    items.sort(key=lambda x: x.key, reverse=True)
    s = calculate_scale(items)

    print(f"item {len(items)} user {len(users)}")
    print(f"k {k} lamda {lambda_value}")
    print(f"scale {s}")

    record: list[list[float]] = []
    record_time: list[list[float]] = []
    rand_query_data = random.Random(1)

    for _ in range(100):
        items.sort(key=lambda x: x.key, reverse=True)
        user_id = rand_query_data.randint(0, len(users) - 1)

        result = ResultData()

        start = time.time()
        execute_greedy_reverse_search(users[user_id].vec, k, lambda_value, 5.0 / s, items, result)
        end = time.time()

        msec = (end - start) * 1000

        # Record results
        record_tmp = [
            user_id,
            result.final_score,
            result.dist_min,
            result.msec1,
            result.msec2,
            msec
        ] + result.answer_id

        record.append(record_tmp)
        record_user_time_data(user_id, record_time, result.time_vec)

    # Output
    output_file_name = f"MovieLens200_k-{k}_lam-{lambda_value}_100.csv"

    with open(output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(record)

    export_record_to_csv(f"time_{output_file_name}", record_time)
    print("finish")

if __name__ == "__main__":
    main()
