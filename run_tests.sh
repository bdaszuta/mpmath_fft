#!/bin/bash

# =============================================================================
# fix for script pathing [with source] [From SE#59895]
export OLD_PWD=${PWD}
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
   # if $SOURCE was a relative symlink, we need to resolve it relative to the
   # path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
export DIR_PACKAGE=${DIR}

cd ${DIR_PACKAGE}
# =============================================================================

# =============================================================================

# test documentation
python -m pytest tests/ -o "addopts="
python -m mypy mpmath_fft/ tests/ benchmarks/
python -m pytest --doctest-modules mpmath_fft/ -o "addopts="

python -m ruff check mpmath_fft/
python -m ruff check tests/
python -m ruff check usage/
python -m ruff check benchmarks/

# Smoke-test usage scripts and benchmarks
python usage/demo.py
python usage/spectral_decay.py
python benchmarks/benchmark.py

# =============================================================================

# =============================================================================
cd ${OLD_PWD}
# =============================================================================
