# Load before running anything, inside WSL from the repo root:  source scripts/env.sh
#
# Puts the Windows driver's own CUDA libraries first on the library path. A native Linux
# driver package inside WSL (libnvidia-compute-580) otherwise shadows
# libnvidia-ptxjitcompiler.so.1, and CUDA crashes on start with a segmentation fault.
# Removing that package (sudo apt remove libnvidia-compute-580) also fixes it.
WSL_DRIVER_DIR="$(ls -dt /usr/lib/wsl/drivers/nv_dispig.inf_amd64_*/ 2>/dev/null | head -1)"
export LD_LIBRARY_PATH="${WSL_DRIVER_DIR:+${WSL_DRIVER_DIR%/}:}/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$HOME/.cache/huggingface"
source "$HOME/venvs/research/bin/activate"
