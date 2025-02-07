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
from math import sqrt, pow
import time
from data_input import Item

@dataclass
class ResultData:
    answer_id: list[int] = field(default_factory=list)
    min_norm: float = 0.0
    ip_sum: float = 0.0
    dist_min: float = float('inf')
    final_score: float = 0.0
    msec1: float = 0.0
    msec2: float = 0.0
    time_vec: list[float] = field(default_factory=list)

@dataclass
class ResultTmp:
    score: float = -1000000.0
    id: int = 0
    dist_min: float = float('inf')
    ip: float = 0.0

@dataclass
class SignData:
    min_key: float = float('inf')
    checked_id: int = 0

def calculate_norm(vec: list[float]) -> float:
    """Calculates the Euclidean norm of a vector.

    Args:
        vec (list[float]): The vector for which the norm is calculated.

    Returns:
        float: The Euclidean norm of the vector.
    """
    return sqrt(sum(x * x for x in vec))

def calculate_inner_product(a: list[float], b: list[float]) -> float:
    """Calculates the inner product of two vectors.

    Args:
        a (list[float]): The first vector.
        b (list[float]): The second vector.

    Returns:
        float: The inner product of the vectors.
    """
    return sum(x * y for x, y in zip(a, b))

def calculate_distance(a: list[float], b: list[float]) -> float:
    """Calculates the Euclidean distance between two vectors.

    Args:
        a (list[float]): The first vector.
        b (list[float]): The second vector.

    Returns:
        float: The Euclidean distance between the vectors.
    """
    return sqrt(sum(pow(a_xy - b_xy, 2) for a_xy, b_xy in zip(a, b)))

def calculate_scale(items: list[Item]) -> float:
    """Calculates the scaling factor for items based on their i2v vectors.

    Args:
        items (list[Item]): The list of items.

    Returns:
        float: The scale calculated from the items.
    """
    max_scalar_vec = [max(item.i2v_vec[j] for item in items) for j in range(len(items[0].i2v_vec))]
    min_scalar_vec = [min(item.i2v_vec[j] for item in items) for j in range(len(items[0].i2v_vec))]
    return calculate_norm([max_scalar_vec[i] - min_scalar_vec[i] for i in range(len(max_scalar_vec))])

def calculate_items_norm_mf(items: list[Item]) -> None:
    """Computes the norm of the MF vectors for each item.

    Args:
        items (list[Item]): The list of items.
    """
    for item in items:
        item.mf_norm = calculate_norm(item.mf_vec)

def calculate_items_norm_i2v(items: list[Item]) -> None:
    """Computes the norm of the i2v vectors for each item.

    Args:
        items (list[Item]): The list of items.
    """
    for item in items:
        item.i2v_norm = calculate_norm(item.i2v_vec)

def make_key(items: list[Item]) -> None:
    """Sets the key for each item based on its MF norm.

    Args:
        items (list[Item]): The list of items.
    """
    for item in items:
        item.key = item.mf_norm

def exploit_cauchy_schwarz_inequality(query: list[float], items: list[Item], query_norm: float, sign_data: SignData, ip_count: int) -> int:
    """Finds the item with the maximum inner product with the query.

    Args:
        query (list[float]): The query vector.
        items (list[Item]): The list of items.
        query_norm (float): The norm of the query vector.
        sign_data (SignData): The structure to store sign data.
        ip_count (int): The count of inner products computed.

    Returns:
        int: The index of the item with the maximum inner product.
    """
    threshold = -1000000.0
    id = 0
    min_tmp = float('inf')
    checked_id = 0

    for i, item in enumerate(items):
        """
        original condition -> if query_norm * item.mf_norm < threshold and i > len(items) * 0.3:
        After asking the last author of this paper, I received a response stating that
        the second condition 'i > len(items) * 0.3' should be unnecessary, so I removed it.
        """
        # early termination
        if query_norm * item.mf_norm < threshold:
            checked_id = i - 1
            break

        ip = calculate_inner_product(query, item.mf_vec)
        item.ip = ip
        item.flag_ip = 1
        tmp = ip / query_norm
        item.key = tmp

        ip_count += 1

        if tmp < min_tmp:
            min_tmp = tmp

        if threshold < ip:
            threshold = ip
            id = i

    sign_data.min_key = min_tmp
    sign_data.checked_id = checked_id
    return id

def sort_items_by_key_partially(items: list[Item]) -> None:
    """Performs a partial sort on items based on their keys.

    Args:
        items (list[Item]): The list of items.
    """
    items.sort(key=lambda x: x.key, reverse=True)

def compute_time(start: float, end: float) -> float:
    """Calculates the elapsed time in milliseconds.

    Args:
        start (float): The start time.
        end (float): The end time.

    Returns:
        float: The elapsed time in milliseconds.
    """
    return (end - start) * 1000.0

def execute_greedy_reverse_search(query: list[float], k: int, lamda: float, s: float, items: list[Item], result: ResultData) -> None:
    """Performs the greedy reverse search algorithm.

    Args:
        query (list[float]): The query vector.
        k (int): The number of recommendations to be made.
        lamda (float): The lambda parameter for the algorithm.
        s (float): The scale factor.
        items (list[Item]): The list of items.
        result (ResultData): The structure to store the results.
    """
    result_i2v_vec: list[list[float]] = []

    start1 = time.time()
    query_norm = calculate_norm(query)
    sign_data = SignData()
    ip_count = 0

    start_id = exploit_cauchy_schwarz_inequality(query, items, query_norm, sign_data, ip_count)
    result.answer_id.append(items[start_id].id)
    items[start_id].flag = 1
    result.ip_sum += items[start_id].ip
    result.min_norm = items[start_id].i2v_norm
    result_i2v_vec.append(items[start_id].i2v_vec)

    end1 = time.time()
    result.msec1 = compute_time(start1, end1)
    result.time_vec.append(compute_time(start1, end1))

    start = time.time()
    sort_flag = 1

    for _ in range(1, k):
        start1 = time.time()
        result_tmp = ResultTmp()
        ip_count = 0

        if sort_flag == 1:
            sort_items_by_key_partially(items)
            sort_flag = 0

        sign_data.min_key = float('inf')
        for j, item in enumerate(items):
            if item.flag == 1:
                if item.key < sign_data.min_key:
                    sign_data.min_key = item.key
                sign_data.checked_id = j
                continue

            # item-level skipping
            threshold = (result_tmp.score - lamda * item.key * query_norm) / (s * (1 - lamda))
            if result.dist_min < threshold:
                sign_data.checked_id = j - 1
                break

            ip = 0.0

            # item-level skipping
            if item.i2v_norm + result.min_norm < threshold:
                if item.key < sign_data.min_key:
                    sign_data.min_key = item.key
                sign_data.checked_id = j
                continue

            if item.flag_ip == 0:
                ip = calculate_inner_product(query, item.mf_vec)
                item.ip = ip
                item.flag_ip = 1
                item.key = ip / query_norm
                sort_flag = 1
                threshold = (result_tmp.score - lamda * item.ip) / (s * (1 - lamda))

                # item-level skipping
                if result.dist_min < threshold:
                    if item.key < sign_data.min_key:
                        sign_data.min_key = item.key
                    sign_data.checked_id = j
                    continue

                # item-level skipping
                if item.i2v_norm + result.min_norm < threshold:
                    if item.key < sign_data.min_key:
                        sign_data.min_key = item.key
                    sign_data.checked_id = j
                    continue
            else:
                ip = item.ip

            # item-level skipping
            if item.min_dist < threshold:
                if item.key < sign_data.min_key:
                    sign_data.min_key = item.key
                sign_data.checked_id = j
                continue

            min_dist = min(result.dist_min, item.min_dist)
            start_index = item.check_ite
            flag = 0
            for answer_index in range(start_index, len(result.answer_id)):
                tmp = calculate_distance(item.i2v_vec, result_i2v_vec[answer_index])
                item.check_ite += 1

                if tmp < item.min_dist:
                    item.min_dist = tmp

                if tmp < min_dist:
                    min_dist = tmp
                    if min_dist < threshold:
                        flag = 1
                        continue

            if item.key < sign_data.min_key:
                sign_data.min_key = item.key
            sign_data.checked_id = j

            if flag == 1:
                continue

            score_tmp = lamda * ip + s * (1 - lamda) * min_dist
            if result_tmp.score < score_tmp:
                result_tmp.score = score_tmp
                result_tmp.id = j
                result_tmp.dist_min = min_dist
                result_tmp.ip = ip

        answer_id_true = items[result_tmp.id].id
        result.answer_id.append(answer_id_true)
        result_i2v_vec.append(items[result_tmp.id].i2v_vec)
        items[result_tmp.id].flag = 1
        result.ip_sum += result_tmp.ip
        result.dist_min = result_tmp.dist_min
        answer_norm = items[result_tmp.id].i2v_norm
        result.min_norm = min(result.min_norm, answer_norm)

        end1 = time.time()
        result.time_vec.append(compute_time(start1, end1))

    end = time.time()
    result.msec2 = compute_time(start, end)

    result.final_score = (lamda * result.ip_sum) / k + s * (1 - lamda) * result.dist_min
