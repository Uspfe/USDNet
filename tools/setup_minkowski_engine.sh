#!/usr/bin/env bash

set -euo pipefail

MINKOWSKI_REPO_URL="https://github.com/NVIDIA/MinkowskiEngine"
FORCE_REBUILD_MINKOWSKI="${FORCE_REBUILD_MINKOWSKI:-0}"
THIRD_PARTY_DIR="${PIXI_PROJECT_ROOT}/third_party"
MINKOWSKI_DIR="${THIRD_PARTY_DIR}/MinkowskiEngine"

##########################################################
# Check pixi environment activation.
##########################################################
if [ -z "${PIXI_PROJECT_ROOT:-}" ]; then
		echo "Error: PIXI_PROJECT_ROOT is not set. Please activate the pixi environment."
		exit 1
fi

##########################################################
# Check for required commands.
##########################################################
for cmd in git sed find python grep xargs; do
	if ! command -v "${cmd}" >/dev/null 2>&1; then
		echo "Error: ${cmd} is required but not found in PATH."
		exit 1
	fi
done


##########################################################
# Check for torch install
##########################################################
TORCH_LIB_DIR="$(python - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    print(f"Error: PyTorch import failed: {exc}", file=sys.stderr)
    sys.exit(2)

print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

if [ ! -d "${TORCH_LIB_DIR}" ]; then
	echo "Error: Could not locate PyTorch runtime lib dir at ${TORCH_LIB_DIR}."
	exit 1
fi

export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

##########################################################
# Check for existing MinkowskiEngine install and skip rebuild if possible
##########################################################
if [ "${FORCE_REBUILD_MINKOWSKI}" != "1" ]; then
	if python - <<'PY'
import sys

try:
    import MinkowskiEngine  # noqa: F401
    import MinkowskiEngineBackend._C  # noqa: F401
except Exception:
    sys.exit(1)

print("MinkowskiEngine is already installed and loadable.")
sys.exit(0)
PY
	then
		echo "Skipping rebuild. Set FORCE_REBUILD_MINKOWSKI=1 to force recompilation."
		exit 0
	fi
fi

##########################################################
# Clone MinkowskiEngine
##########################################################
mkdir -p "${THIRD_PARTY_DIR}"

if [ -d "${MINKOWSKI_DIR}" ]; then
	echo "MinkowskiEngine directory already exists. Skipping clone."
else
	git clone --recursive "${MINKOWSKI_REPO_URL}" "${MINKOWSKI_DIR}"
fi

##########################################################
# Apply patches to MinkowskiEngine to fix CUDA 12 compatibility
##########################################################
prepend_if_missing() {
	local file="$1"
	local include_line="$2"
	if ! grep -Fqx "${include_line}" "${file}"; then
		sed -i "1i ${include_line}" "${file}"
	fi
}

pushd "${MINKOWSKI_DIR}" >/dev/null

git submodule update --init --recursive

# Patch setup.py flags to match the currently working local changes.
python - <<'PY'
from pathlib import Path
import re
import sys

path = Path("setup.py")
text = path.read_text()

replacement = """CC_FLAGS += ARGS + [\"-DNVTX_DISABLE\", \"-D_LIBCPP_HAS_NO_ASAN\"]
NVCC_FLAGS += ARGS + [\"-DNVTX_DISABLE\"] + [\"-DNVTX_DISABLE\", '-D_GLIBCXX_USE_CXX11_ABI=1', '-Xcompiler=-fpermissive',]"""

if replacement in text:
	    sys.exit(0)

new_text, count = re.subn(
	    r"CC_FLAGS\s*\+=\s*ARGS\s*\n\s*NVCC_FLAGS\s*\+=\s*ARGS",
	    replacement,
	    text,
	    count=1,
)

if count == 1:
	    path.write_text(new_text)
	    sys.exit(0)

raise SystemExit("Error: Could not patch setup.py ARGS flags block")
PY

# Patch includes required for newer CUDA/thrust combinations.
prepend_if_missing "src/3rdparty/concurrent_unordered_map.cuh" "#include <thrust/execution_policy.h>"
prepend_if_missing "src/convolution_kernel.cuh" "#include <thrust/execution_policy.h>"
prepend_if_missing "src/coordinate_map_gpu.cu" "#include <thrust/unique.h>"
prepend_if_missing "src/coordinate_map_gpu.cu" "#include <thrust/remove.h>"
prepend_if_missing "src/spmm.cu" "#include <thrust/execution_policy.h>"
prepend_if_missing "src/spmm.cu" "#include <thrust/reduce.h>"
prepend_if_missing "src/spmm.cu" "#include <thrust/sort.h>"

# Patch map creation logic in coordinate_map_gpu.cuh.
python - <<'PY'
from pathlib import Path
import re
import sys

path = Path("src/coordinate_map_gpu.cuh")
text = path.read_text()

new_block = """      auto created_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
      // Avoid shared_ptr assignment from unique_ptr, which can trigger
      // std/cuda::std __to_address ambiguity with newer CUDA toolchains.
      m_map = std::shared_ptr<map_type>(created_map.release(),
                        created_map.get_deleter());"""

if new_block in text:
	    sys.exit(0)

pattern = (
	    r"^\s{6}m_map\s*=\s*map_type::create\(\n"
	    r"\s*compute_hash_table_size\(size,\s*m_hashtable_occupancy\),\n"
	    r"\s*m_unused_element,\s*m_unused_key,\s*m_hasher,\s*m_equal,\s*m_map_allocator\);"
)

new_text, count = re.subn(pattern, new_block, text, count=1, flags=re.MULTILINE)
if count == 1:
	    path.write_text(new_text)
	    sys.exit(0)

raise SystemExit("Error: Could not find coordinate_map_gpu.cuh block to patch")
PY

find src -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" \) -print0 | \
	xargs -0 sed -i 's/thrust::device/thrust::cuda::par/g'


##########################################################
# Build MinkowskiEngine with host gcc (<= 13)
##########################################################
env CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ \
	python setup.py install --blas=openblas --force_cuda

popd >/dev/null

echo "MinkowskiEngine setup complete."
