# Source this before ANY embedding-step command:  source embed_env.sh
#
# Keeps every download, cache and temp file on E:. Without these, pip unpacks
# wheels into %TEMP% (C:), pip and Hugging Face cache under the user profile (C:),
# and CUDA writes its kernel cache to %APPDATA% (C:).
# directory that holds venv_embed/, cache/ and tmp/ -- override with EMBED_ROOT
ROOT="${EMBED_ROOT:-E:/ml hackathon}"
export VENV="$ROOT/venv_embed"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TMP="$ROOT/tmp"
export TEMP="$ROOT/tmp"
export TMPDIR="$ROOT/tmp"
export HF_HOME="$ROOT/cache/hf"
export HF_HUB_CACHE="$ROOT/cache/hf/hub"
export SENTENCE_TRANSFORMERS_HOME="$ROOT/cache/hf/st"
export TRANSFORMERS_CACHE="$ROOT/cache/hf/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$ROOT/cache/torch"
export CUDA_CACHE_PATH="$ROOT/cache/cuda"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export PYTHONHASHSEED=0
mkdir -p "$PIP_CACHE_DIR" "$TMP" "$HF_HOME" "$HF_HUB_CACHE" "$SENTENCE_TRANSFORMERS_HOME" \
         "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
