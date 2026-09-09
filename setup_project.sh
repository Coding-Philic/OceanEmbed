#!/usr/bin/env bash
# =============================================================================
#  OceanEmbed — Project Setup Script
#  Run once after cloning:  bash setup_project.sh
# =============================================================================

set -euo pipefail

# ─── Colours ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERR]${RESET}   $*"; exit 1; }

echo -e "\n${BOLD}╔══════════════════════════════════════════╗"
echo    "║       OceanEmbed  Project  Setup         ║"
echo -e "╚══════════════════════════════════════════╝${RESET}\n"

# ─── 1. Python version check ─────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=$(command -v python3 || command -v python || error "Python not found. Install Python >= 3.10.")
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    error "Python >= 3.10 required (found $PY_VER). Please upgrade."
fi
success "Python $PY_VER found at $PYTHON"

# ─── 2. Create virtual environment ───────────────────────────────────────────
VENV_DIR=".venv"
if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment '$VENV_DIR' already exists — skipping creation."
else
    info "Creating virtual environment in '$VENV_DIR'..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

source "$VENV_DIR/bin/activate"
info "Virtual environment activated."

# ─── 3. Upgrade pip / build tools ────────────────────────────────────────────
info "Upgrading pip, setuptools, wheel..."
pip install --quiet --upgrade pip setuptools wheel
success "Build tools up-to-date."

# ─── 4. Install project (editable) ───────────────────────────────────────────
info "Installing OceanEmbed package in editable mode..."
pip install --quiet -e ".[dev]"
success "OceanEmbed installed."

# ─── 5. Create required runtime directories ──────────────────────────────────
info "Creating runtime directory structure..."

DIRS=(
    "data/raw"
    "data/processed"
    "data/aligned"
    "data/cache"
    "outputs/checkpoints"
    "outputs/predictions"
    "outputs/plots"
    "lightning_logs"
    "docs"
    "notebooks"
)

for dir in "${DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        touch "$dir/.gitkeep" 2>/dev/null || true
        echo "  Created: $dir"
    else
        echo "  Exists:  $dir"
    fi
done
success "Directory structure ready."

# ─── 6. Create .env from template if missing ─────────────────────────────────
if [[ ! -f ".env" ]]; then
    info "Creating .env template..."
    cat > .env <<'ENV'
# ── Copernicus Marine ─────────────────────────────────────────────
COPERNICUS_USERNAME=your_username
COPERNICUS_PASSWORD=your_password

# ── NASA EarthData ─────────────────────────────────────────────────
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password

# ── Weights & Biases ──────────────────────────────────────────────
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=oceanembed
WANDB_ENTITY=your_entity

# ── Paths (override if needed) ─────────────────────────────────────
DATA_DIR=data
OUTPUT_DIR=outputs
ENV
    warn ".env created — fill in your API credentials before running scripts."
else
    info ".env already exists — skipping."
fi

# ─── 7. Run tests ─────────────────────────────────────────────────────────────
info "Running test suite..."
if python -m pytest tests/ -q --tb=short 2>&1; then
    success "All tests passed."
else
    warn "Some tests failed — check output above. Setup still complete."
fi

# ─── 8. Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════╗"
echo    "║         Setup complete!                  ║"
echo -e "╚══════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Activate the venv:   ${CYAN}source .venv/bin/activate${RESET}"
echo -e "  Train:               ${CYAN}python scripts/train.py --config configs/poc_bob.yaml${RESET}"
echo -e "  Run webapp:          ${CYAN}python webapp/app.py${RESET}"
echo -e "  Fill credentials:    ${CYAN}.env${RESET}"
echo ""
