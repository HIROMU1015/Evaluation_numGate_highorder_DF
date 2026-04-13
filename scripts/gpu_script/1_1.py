
import os
import sys
from glob import glob

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(_PROJECT_DIR, ".venv", "bin", "python")
_CUDA_BOOTSTRAP_ENV = "MYPROJECT_CUDA_BOOTSTRAPPED"

def _collect_cuda_lib_dirs():
    cuda_dirs = []
    site_packages_list = glob(os.path.join(_PROJECT_DIR, ".venv", "lib", "python*", "site-packages"))
    rel_paths = [
        os.path.join("nvidia", "nvjitlink", "lib"),
        os.path.join("nvidia", "cusparse", "lib"),
        os.path.join("nvidia", "cusolver", "lib"),
        os.path.join("nvidia", "cublas", "lib"),
        os.path.join("nvidia", "cuda_runtime", "lib"),
        os.path.join("cuquantum", "lib"),
        os.path.join("cutensor", "lib"),
    ]
    for site_packages in site_packages_list:
        for rel_path in rel_paths:
            lib_dir = os.path.join(site_packages, rel_path)
            if os.path.isdir(lib_dir):
                cuda_dirs.append(lib_dir)
    return cuda_dirs

def _prepend_library_path(env, lib_dirs):
    current = env.get("LD_LIBRARY_PATH", "")
    current_dirs = [path for path in current.split(":") if path]
    merged = []
    for path in [*lib_dirs, *current_dirs]:
        if path and path not in merged:
            merged.append(path)
    if merged:
        env["LD_LIBRARY_PATH"] = ":".join(merged)

_cuda_lib_dirs = _collect_cuda_lib_dirs()

if os.path.exists(_VENV_PYTHON) and (
    os.path.abspath(sys.executable) != _VENV_PYTHON or os.environ.get(_CUDA_BOOTSTRAP_ENV) != "1"
):
    # Re-exec under the project virtualenv with its CUDA libraries first in the search path.
    env = os.environ.copy()
    _prepend_library_path(env, _cuda_lib_dirs)
    env[_CUDA_BOOTSTRAP_ENV] = "1"
    os.execve(_VENV_PYTHON, [_VENV_PYTHON, __file__, *sys.argv[1:]], env)

import Trotter_sim as tsag
s_time = 0.100
e_time = 0.105

s_time = 0.200
e_time = 0.205

dividing = 0.004
gpu = ['2','1']
gup = [f'{i}' for i in range(1,3)]
tsag.trotter_error_plt_qc_gr(s_time, e_time, dividing, mol_type='H4', num_w='my4',use_gpu=gpu)



# from qiskit_aer import AerSimulator
# simulator= AerSimulator(method='statevector')
# print(simulator.available_devices())

# import time
# for i in range(10000):
#     print(i)
#     time.sleep(1)

# import gzip
# import shutil
# import os

# src = "/home/AbeHiromu/myproject/result/-0.112.pkl"        # もともとの pickle ファイル
# dst = "/home/AbeHiromu/myproject/result/-0.112.pkl.gz"     # gzip で圧縮して保存したいファイル名

# try:
#     # gzip で圧縮しながらコピー
#     with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
#         shutil.copyfileobj(f_in, f_out)

#     # ここまで来たら変換成功とみなし、元ファイルを削除
#     os.remove(src)

# except Exception as e:
#     # 失敗時の処理（ログ出力など）
#     print(f"変換に失敗しました: {e}")
