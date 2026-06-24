#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PREFIX="${MEEP_CONDA_PREFIX:-/home/smrm/miniconda3/envs/mp}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
CXX="${CXX:-${CONDA_ENV_PREFIX}/bin/mpic++}"

export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"

EXT_SUFFIX="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PY_INCLUDE="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')"
NP_INCLUDE="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.get_include())')"

"${CXX}" -O3 -std=c++17 -shared -fPIC \
  -I"${PY_INCLUDE}" \
  -I"${NP_INCLUDE}" \
  -I"${CONDA_ENV_PREFIX}/include" \
  "${SCRIPT_DIR}/native_sampler.cpp" \
  -L"${CONDA_ENV_PREFIX}/lib" \
  -Wl,-rpath,"${CONDA_ENV_PREFIX}/lib" \
  -lmeep \
  -o "${SCRIPT_DIR}/native_sampler${EXT_SUFFIX}"

echo "Built ${SCRIPT_DIR}/native_sampler${EXT_SUFFIX}"
