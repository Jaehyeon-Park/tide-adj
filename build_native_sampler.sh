#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${SCRIPT_DIR}/src/tide_adj"
CONDA_ENV_PREFIX="${MEEP_CONDA_PREFIX:-${CONDA_PREFIX:-}}"

if [[ -z "${CONDA_ENV_PREFIX}" ]]; then
  echo "Error: activate a conda environment or set MEEP_CONDA_PREFIX." >&2
  exit 1
fi

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
  "${PACKAGE_DIR}/native_sampler.cpp" \
  -L"${CONDA_ENV_PREFIX}/lib" \
  -Wl,-rpath,"${CONDA_ENV_PREFIX}/lib" \
  -lmeep \
  -o "${PACKAGE_DIR}/native_sampler${EXT_SUFFIX}"

echo "Built ${PACKAGE_DIR}/native_sampler${EXT_SUFFIX}"
