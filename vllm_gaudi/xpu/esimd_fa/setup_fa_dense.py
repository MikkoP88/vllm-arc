"""Build only the fa_dense_ops extension (p72 dense-FA ESIMD kernel).

Donor: setup_qsa_only.py (W14, md5 from
results/2026-10-31-p72-rungb-dense-fa-esimd/donor-w14/) — flags verbatim
per plan Appendix D (DPC++ 2026.2 JIT spir64 law; TORCH_XPU_ARCH_LIST=bmg
and the icpx shim come from the wrapper script, not here).
Build host: the rig's esimd container (tools/esimd/fa_dense_build.sh).
"""
from pathlib import Path
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension

# The donor's build harness lives at /src/esimd in the esimd container
# (W14 recipe); reuse it read-only from there.
sys.path.insert(0, "/src/esimd")
from esimd_build_extention import BuildExtension  # noqa: E402,N813
import torch  # noqa: E402

root = Path(__file__).parent.resolve()
torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    SyclExtension(
        name="fa_dense_ops",
        sources=[
            "csrc/fa_dense.sycl",
        ],
        include_dirs=[
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    ),
    # P6.21 v2: SLM-tile redesign (one extension name per PYBIND module).
    SyclExtension(
        name="fa_dense_v2_ops",
        sources=[
            "csrc/fa_dense_v2.sycl",
        ],
        include_dirs=[
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    ),
    # P6.21 r92: dpas operand/acc layout probe (N=8 raw + SLM-roundtrip
    # + N=16 control). Debug instrument, not serve code.
    SyclExtension(
        name="dpas_probe_ops",
        sources=[
            "csrc/dpas_probe.sycl",
        ],
        include_dirs=[
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    ),
]

setup(
    name="fa_dense_ops",
    version="0.1.0",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
