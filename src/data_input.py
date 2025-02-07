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
from dataclasses import dataclass, field
import json
import os

@dataclass
class Item:
    id: int = 0
    i2v_vec: list[float] = field(default_factory=list)
    mf_vec: list[float] = field(default_factory=list)
    i2v_norm: float = 0.0
    mf_norm: float = 0.0
    ip: float = 0.0
    flag: int = 0
    min_dist: float = float('inf')
    check_ite: int = 0
    flag_ip: int = 0
    key: float = 0.0

@dataclass
class User:
    vec: list[float] = field(default_factory=list)
    norm: float = 0.0

def split(input_str: str, delimiter: str) -> list[str]:
    """Splits a string by a given delimiter.

    Args:
        input_str (str): The string to split.
        delimiter (str): The delimiter to split the string.

    Returns:
        list[str]: A list of substrings split by the delimiter.
    """
    return input_str.split(delimiter)

def load_mf_data_from_file(vec_users: list[User], vec_items: list[Item], *, input_id: int) -> None:
    """Reads matrix factorization data from a file and stores it in the provided lists.

    Args:
        vec_users (list[User]): The list to store user data.
        vec_items (list[Item]): The list to store item data.
        input_id (int): An identifier to select the correct data file.
    """
    input_files = [
        "../dataset/dataset_MF200/netflix_mf-200.txt",
        "../dataset/dataset_MF200/amazon_Movies_and_TV_mf-200.txt",
        "../dataset/dataset_MF200/amazon_Kindle_Store_mf-200.txt",
        "../dataset/dataset_MF200/MovieLens_mf-200.txt"
    ]
    input_file = input_files[input_id]

    if not os.path.exists(input_file):
        print("Error! File can not be opened")
        return

    with open(input_file, 'r') as file:
        for line in file:
            strvec = split(line.strip(), ' ')
            if strvec[1] == "F":
                continue
            tmp_str = strvec[0]
            if tmp_str.startswith('p'):  # user
                user = User(vec=[float(x) for x in strvec[2:]])
                vec_users.append(user)
            elif tmp_str.startswith('q'):  # item
                item = Item(id=int(tmp_str[1:]), mf_vec=[float(x) for x in strvec[2:]])
                vec_items.append(item)

def load_item2vec_data_from_file(vec_items: list[Item], *, input_id: int) -> None:
    """Reads item2vec data from a file and stores it in the provided list.

    Args:
        vec_items (list[Item]): The list to store item data.
        input_id (int): An identifier to select the correct data file.
    """
    input_files = [
        "../dataset/dataset_item2vec/netflix_item2vec_d-200.txt",
        "../dataset/dataset_item2vec/amazon_Movie_item2vec_d-200.txt",
        "../dataset/dataset_item2vec/amazon_Kindle_item2vec_d-200.txt",
        "../dataset/dataset_item2vec/MovieLens_item2vec_d-200.txt"
    ]
    input_file = input_files[input_id]

    if not os.path.exists(input_file):
        print("Error! File can not be opened")
        return

    print(f"mf items {len(vec_items)}")
    with open(input_file, 'r') as file:
        for item, line in zip(vec_items, file):
            item.i2v_vec = [float(x) for x in split(line.strip(), ' ')[1:]]

def load_config(file_path: str) -> dict[str, int | float] | None:
    """Loads configuration from a JSON file.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        dict[str, int | float] | None: The configuration data, or None if the file cannot be opened.
    """
    if not os.path.exists(file_path):
        print("Error! File cannot be opened")
        return None

    with open(file_path, 'r') as file:
        try:
            config = json.load(file)
            return config
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None

def get_param_from_config(config: dict[str, int | float], key: str) -> int | float | None:
    """Extracts a parameter from the configuration by key.

    Args:
        config (dict[str, int | float]): The configuration data.
        key (str): The key of the parameter to extract.

    Returns:
        int | float | None: The value of the parameter, or None if not found.
    """
    return config.get(key)
