# IP-Greedy Python Implementation

This repository contains the Python implementation of IP-Greedy (RecSys'22).

For detailed logic, please refer to [the original paper](https://doi.org/10.1145/3523227.3546779).

> Solving Diversity-Aware Maximum Inner Product Search Efficiently and Effectively (RecSys'22)

The author's implementation in C++ is [here](https://github.com/peitaw22/IP-Greedy).

## Differences from the Author's C++ Implementation

- Rewritten the code in Python
- Improved readability of file, variable and function names
- Optimized redundant processes
- Enhanced directory structure
- Changed the parameter configuration method
- Made the code more "Pythonic" as part of the migration to Python
- Add comments
- Introduced `uv` for package management

 I have tried to ensure that the implementation performs the same operations as the C++ version, but there might be mistakes. If you notice any, please let me know.

 ## Requirements
 - Python 3.11.5
 - `uv` 0.5.9

 ## Getting started
- `uv sync`
- `source .venv/bin/activate`

 ## How to run
 `uv run main.py`

## License

This project is licensed under the Apache License 2.0. See the `license.txt` for details.

## Contact

For any questions or issues, please open an issue on this repository or contact the maintainers directly.
