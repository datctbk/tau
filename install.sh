#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  tau — universal installer
#
#  One-liner:
#    curl -fsSL https://raw.githubusercontent.com/datctbk/tau/main/install.sh | bash
#
#  Or locally:
#    bash install.sh
#
#  Flags:
#    --minimal          Install core only (skip ecosystem packages)
#    --prefix <path>    Install location (default: ~/.tau)
#    --no-modify-path   Don't add tau to PATH in shell rc files
#    --no-codegraph     Skip codegraph installation
#    --help             Show this help
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
TAU_PREFIX="${HOME}/.tau"
MODIFY_PATH=true
INSTALL_ECOSYSTEM=true
INSTALL_CODEGRAPH=true
MIN_PYTHON="3.11"
GITHUB_ORG="datctbk"

# ── Ecosystem packages (installed via tau extensions install) ─────────
ECOSYSTEM_PACKAGES=(
    "tau-memory"
    "tau-agents"
    "tau-assistant"
    "tau-gateway"
    "tau-web"
    "tau-aidlc"
)

# ── Colours ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' BOLD='' DIM='' RESET=''
fi

# ── Helpers ───────────────────────────────────────────────────────────
info()    { printf "${CYAN}  ℹ ${RESET}%s\n" "$*"; }
success() { printf "${GREEN}  ✓ ${RESET}%s\n" "$*"; }
warn()    { printf "${YELLOW}  ⚠ ${RESET}%s\n" "$*"; }
error()   { printf "${RED}  ✗ ${RESET}%s\n" "$*" >&2; }
fatal()   { error "$*"; exit 1; }
step()    { printf "\n${BOLD}${MAGENTA}  ▸ %s${RESET}\n" "$*"; }

banner() {
    printf "\n"
    printf "${BOLD}${CYAN}"
    printf "    ████████╗ █████╗ ██╗   ██╗\n"
    printf "    ╚══██╔══╝██╔══██╗██║   ██║\n"
    printf "       ██║   ███████║██║   ██║\n"
    printf "       ██║   ██╔══██║██║   ██║\n"
    printf "       ██║   ██║  ██║╚██████╔╝\n"
    printf "       ╚═╝   ╚═╝  ╚═╝ ╚═════╝ \n"
    printf "${RESET}"
    printf "${DIM}    universal installer${RESET}\n"
    printf "\n"
}

usage() {
    cat <<EOF
Usage: install.sh [OPTIONS]

Options:
  --prefix <path>     Install location (default: ~/.tau)
  --minimal           Install core only, skip ecosystem packages
  --no-modify-path    Don't modify shell rc files
  --no-codegraph      Skip codegraph installation
  --help              Show this help

Examples:
  bash install.sh
  bash install.sh --prefix /opt/tau --minimal
  curl -fsSL https://raw.githubusercontent.com/${GITHUB_ORG}/tau/main/install.sh | bash
EOF
    exit 0
}

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            TAU_PREFIX="$2"; shift 2 ;;
        --minimal)
            INSTALL_ECOSYSTEM=false; shift ;;
        --no-modify-path)
            MODIFY_PATH=false; shift ;;
        --no-codegraph)
            INSTALL_CODEGRAPH=false; shift ;;
        --help|-h)
            usage ;;
        *)
            fatal "Unknown option: $1  (use --help)" ;;
    esac
done

# Resolve prefix to absolute path
TAU_PREFIX="$(cd "$(dirname "$TAU_PREFIX")" 2>/dev/null && pwd)/$(basename "$TAU_PREFIX")" 2>/dev/null || TAU_PREFIX="$(realpath -m "$TAU_PREFIX" 2>/dev/null || echo "$TAU_PREFIX")"

TAU_SRC="${TAU_PREFIX}/src"
TAU_VENV="${TAU_PREFIX}/venv"
TAU_BIN="${HOME}/.local/bin"

# ── Detect OS & architecture ─────────────────────────────────────────
detect_platform() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Darwin) OS="macos" ;;
        Linux)  OS="linux" ;;
        *)      fatal "Unsupported operating system: $os" ;;
    esac

    case "$arch" in
        x86_64|amd64)   ARCH="x86_64" ;;
        arm64|aarch64)  ARCH="arm64" ;;
        *)              fatal "Unsupported architecture: $arch" ;;
    esac
}

# ── Python version check ─────────────────────────────────────────────
version_gte() {
    # Returns 0 if $1 >= $2 (version comparison)
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

find_python() {
    # Try common Python names
    for cmd in python3.13 python3.12 python3.11 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" || continue
            if version_gte "$ver" "$MIN_PYTHON"; then
                PYTHON_CMD="$cmd"
                PYTHON_VER="$ver"
                return 0
            fi
        fi
    done
    return 1
}

install_python() {
    step "Installing Python ${MIN_PYTHON}+"

    if [[ "$OS" == "macos" ]]; then
        if command -v brew &>/dev/null; then
            info "Installing Python via Homebrew..."
            brew install python@3.12
            # Homebrew Python path
            if [[ -x "/opt/homebrew/bin/python3.12" ]]; then
                PYTHON_CMD="/opt/homebrew/bin/python3.12"
            elif [[ -x "/usr/local/bin/python3.12" ]]; then
                PYTHON_CMD="/usr/local/bin/python3.12"
            else
                PYTHON_CMD="python3.12"
            fi
            PYTHON_VER="3.12"
        else
            error "Homebrew not found. Please install Python ${MIN_PYTHON}+ manually:"
            info "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            info "  brew install python@3.12"
            fatal "Then re-run this installer."
        fi
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &>/dev/null; then
            info "Installing Python via apt (may require sudo)..."
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip 2>/dev/null || {
                info "Adding deadsnakes PPA..."
                sudo apt-get install -y -qq software-properties-common
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt-get update -qq
                sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip
            }
            PYTHON_CMD="python3.12"
            PYTHON_VER="3.12"
        elif command -v dnf &>/dev/null; then
            info "Installing Python via dnf (may require sudo)..."
            sudo dnf install -y python3.12 python3.12-pip
            PYTHON_CMD="python3.12"
            PYTHON_VER="3.12"
        elif command -v yum &>/dev/null; then
            info "Installing Python via yum (may require sudo)..."
            sudo yum install -y python3.12 python3.12-pip
            PYTHON_CMD="python3.12"
            PYTHON_VER="3.12"
        elif command -v pacman &>/dev/null; then
            info "Installing Python via pacman (may require sudo)..."
            sudo pacman -S --noconfirm python
            PYTHON_CMD="python3"
            PYTHON_VER="$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        else
            fatal "No supported package manager found. Please install Python ${MIN_PYTHON}+ manually."
        fi
    fi

    success "Python ${PYTHON_VER} installed"
}

# ── Git check ─────────────────────────────────────────────────────────
check_git() {
    if command -v git &>/dev/null; then
        return 0
    fi

    step "Installing git"
    if [[ "$OS" == "macos" ]]; then
        if command -v brew &>/dev/null; then
            brew install git
        else
            info "Installing Xcode Command Line Tools (includes git)..."
            xcode-select --install 2>/dev/null || true
            fatal "Please complete Xcode CLT installation and re-run this script."
        fi
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y -qq git
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y git
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm git
        else
            fatal "Please install git manually and re-run this script."
        fi
    fi
    success "git installed"
}

# ── Clone or update a repo ────────────────────────────────────────────
clone_or_update() {
    local repo_url="$1"
    local dest="$2"
    local name="$3"

    if [[ -d "${dest}/.git" ]]; then
        info "Updating ${name}..."
        git -C "$dest" pull --ff-only --quiet 2>/dev/null || {
            warn "Could not fast-forward ${name}, skipping update"
        }
    else
        info "Cloning ${name}..."
        git clone --depth 1 --quiet "$repo_url" "$dest"
    fi
}

# ── Create virtual environment ────────────────────────────────────────
setup_venv() {
    if [[ -d "${TAU_VENV}/bin" ]]; then
        info "Virtual environment already exists"
        return 0
    fi

    info "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$TAU_VENV"
    success "Virtual environment created at ${TAU_VENV}"
}

# ── Install pip package in editable mode ──────────────────────────────
pip_install_editable() {
    local pkg_path="$1"
    local name="$2"

    info "Installing ${name}..."
    "${TAU_VENV}/bin/pip" install --quiet --upgrade pip 2>/dev/null || true
    "${TAU_VENV}/bin/pip" install --quiet -e "$pkg_path"
    success "${name} installed"
}

# ── Install ecosystem packages via tau extensions install ─────────────
install_ecosystem() {
    local tau_bin="${TAU_VENV}/bin/tau"

    for pkg in "${ECOSYSTEM_PACKAGES[@]}"; do
        local repo_url="https://github.com/${GITHUB_ORG}/${pkg}"
        info "Installing package: ${pkg}..."
        "$tau_bin" extensions install "git:${repo_url}" 2>/dev/null && {
            success "${pkg} installed"
        } || {
            # If already installed, try update instead
            local normalized="${pkg//-/_}"
            "$tau_bin" extensions update "$normalized" 2>/dev/null && {
                success "${pkg} updated"
            } || {
                warn "Could not install/update ${pkg} — skipping"
            }
        }
    done
}

# ── Create wrapper script ─────────────────────────────────────────────
create_wrapper() {
    mkdir -p "$TAU_BIN"

    cat > "${TAU_BIN}/tau" <<WRAPPER
#!/usr/bin/env bash
# tau wrapper — auto-generated by install.sh
# Activates the tau venv and delegates to the real tau CLI.
exec "${TAU_VENV}/bin/tau" "\$@"
WRAPPER
    chmod +x "${TAU_BIN}/tau"
    success "Wrapper script created at ${TAU_BIN}/tau"
}

# ── Modify shell rc files ─────────────────────────────────────────────
modify_path() {
    local rc_files=()
    local shell_name=""
    local marker="# Added by tau installer"

    # Detect shell
    if [[ -n "${SHELL:-}" ]]; then
        case "$(basename "$SHELL")" in
            zsh)  rc_files=("$HOME/.zshrc"); shell_name="zsh" ;;
            bash) rc_files=("$HOME/.bashrc" "$HOME/.bash_profile"); shell_name="bash" ;;
            fish) rc_files=("$HOME/.config/fish/config.fish"); shell_name="fish" ;;
            *)    rc_files=("$HOME/.profile"); shell_name="sh" ;;
        esac
    else
        rc_files=("$HOME/.profile")
        shell_name="sh"
    fi

    for rc in "${rc_files[@]}"; do
        # Skip if already present
        if [[ -f "$rc" ]] && grep -q "$marker" "$rc" 2>/dev/null; then
            info "PATH already configured in $(basename "$rc")"
            return 0
        fi
    done

    # Only modify the first rc file
    local target="${rc_files[0]}"
    [[ -f "$target" ]] || touch "$target"

    if [[ "$shell_name" == "fish" ]]; then
        cat >> "$target" <<EOF

${marker}
if not contains "${TAU_BIN}" \$PATH
    set -gx PATH "${TAU_BIN}" \$PATH
end
EOF
    else
        cat >> "$target" <<EOF

${marker}
export PATH="${TAU_BIN}:\$PATH"
EOF
    fi

    success "Added ${TAU_BIN} to PATH in $(basename "$target")"
    info "Run: ${DIM}source ${target}${RESET}  (or open a new terminal)"
}

# ── Summary ───────────────────────────────────────────────────────────
print_summary() {
    local tau_ver
    tau_ver="$("${TAU_VENV}/bin/tau" run --help 2>/dev/null | head -1)" || tau_ver="installed"

    printf "\n"
    printf "${BOLD}${GREEN}  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
    printf "${BOLD}${GREEN}    ✓  tau installed successfully!${RESET}\n"
    printf "${BOLD}${GREEN}  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
    printf "\n"
    printf "  ${BOLD}Location:${RESET}     ${TAU_PREFIX}\n"
    printf "  ${BOLD}Python:${RESET}       ${PYTHON_CMD} (${PYTHON_VER})\n"
    printf "  ${BOLD}Venv:${RESET}         ${TAU_VENV}\n"
    printf "  ${BOLD}Binary:${RESET}       ${TAU_BIN}/tau\n"
    printf "\n"
    printf "  ${BOLD}Quick start:${RESET}\n"
    printf "    ${CYAN}tau run \"hello world\"${RESET}        ${DIM}# single-shot${RESET}\n"
    printf "    ${CYAN}tau run${RESET}                      ${DIM}# interactive REPL${RESET}\n"
    printf "    ${CYAN}tau setup --list${RESET}             ${DIM}# see installed packages${RESET}\n"
    printf "    ${CYAN}tau extensions list${RESET}          ${DIM}# list extensions${RESET}\n"
    printf "\n"
    printf "  ${BOLD}Configure:${RESET}\n"
    printf "    ${CYAN}export OPENAI_API_KEY=\"sk-...\"${RESET}\n"
    printf "    ${CYAN}tau config set provider openai${RESET}\n"
    printf "    ${CYAN}tau config set model gpt-4o${RESET}\n"
    printf "\n"
    printf "  ${BOLD}Uninstall:${RESET}\n"
    printf "    ${DIM}bash ${TAU_SRC}/tau/uninstall.sh${RESET}\n"
    printf "\n"
}

# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

main() {
    banner

    # ── 1. Detect platform ────────────────────────────────────────────
    step "Detecting platform"
    detect_platform
    success "Platform: ${OS} ${ARCH}"

    # ── 2. Check / install prerequisites ──────────────────────────────
    step "Checking prerequisites"

    check_git
    success "git $(git --version | cut -d' ' -f3)"

    if find_python; then
        success "Python ${PYTHON_VER} (${PYTHON_CMD})"
    else
        warn "Python ${MIN_PYTHON}+ not found"
        install_python
        # Verify it worked
        if ! find_python; then
            fatal "Python installation failed. Please install Python ${MIN_PYTHON}+ manually."
        fi
    fi

    # Check venv module
    "$PYTHON_CMD" -m venv --help &>/dev/null || {
        warn "Python venv module not available, installing..."
        if [[ "$OS" == "linux" ]]; then
            sudo apt-get install -y -qq "python${PYTHON_VER}-venv" 2>/dev/null || true
        fi
        "$PYTHON_CMD" -m venv --help &>/dev/null || fatal "Python venv module not available. Install it manually."
    }
    success "Python venv module available"

    # ── 3. Create install directory ───────────────────────────────────
    step "Setting up directory structure"
    mkdir -p "$TAU_SRC"
    success "Install directory: ${TAU_PREFIX}"

    # ── 4. Clone core repo ────────────────────────────────────────────
    step "Installing tau core"
    clone_or_update "https://github.com/${GITHUB_ORG}/tau" "${TAU_SRC}/tau" "tau"

    # ── 5. Create virtual environment ─────────────────────────────────
    step "Setting up Python environment"
    setup_venv

    # ── 6. Install tau core ───────────────────────────────────────────
    pip_install_editable "${TAU_SRC}/tau" "tau core"

    # ── 7. Install codegraph (if requested) ───────────────────────────
    if [[ "$INSTALL_CODEGRAPH" == true ]]; then
        step "Installing codegraph"
        # codegraph lives in the parent workspace — clone separately if not bundled
        if [[ -d "${TAU_SRC}/tau-codegraph" ]]; then
            pip_install_editable "${TAU_SRC}/tau-codegraph" "codegraph"
        else
            # Try to find codegraph as a sibling in the workspace
            local codegraph_url="https://github.com/${GITHUB_ORG}/tau-codegraph"
            # Only attempt if the repo exists (non-fatal if it doesn't)
            if git ls-remote "$codegraph_url" &>/dev/null; then
                clone_or_update "$codegraph_url" "${TAU_SRC}/tau-codegraph" "codegraph"
                pip_install_editable "${TAU_SRC}/tau-codegraph" "codegraph"
            else
                warn "codegraph repo not found at ${codegraph_url} — skipping"
            fi
        fi
    fi

    # ── 8. Create wrapper script ──────────────────────────────────────
    step "Creating command-line wrapper"
    create_wrapper

    # ── 9. Verify core installation ───────────────────────────────────
    step "Verifying installation"
    if "${TAU_VENV}/bin/tau" run --help &>/dev/null; then
        success "tau core is working"
    else
        fatal "tau core verification failed. Check the logs above for errors."
    fi

    # ── 10. Install ecosystem packages ────────────────────────────────
    if [[ "$INSTALL_ECOSYSTEM" == true ]]; then
        step "Installing ecosystem packages"
        install_ecosystem
    else
        info "Skipping ecosystem packages (--minimal)"
    fi

    # ── 11. Configure PATH ────────────────────────────────────────────
    if [[ "$MODIFY_PATH" == true ]]; then
        step "Configuring PATH"
        modify_path
    else
        info "Skipping PATH modification (--no-modify-path)"
        info "Add this to your shell rc: export PATH=\"${TAU_BIN}:\$PATH\""
    fi

    # ── Done ──────────────────────────────────────────────────────────
    print_summary
}

main "$@"
