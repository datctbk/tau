#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  tau — uninstaller
#
#  Usage:
#    bash uninstall.sh            # interactive (confirmation prompt)
#    bash uninstall.sh -y         # skip confirmation
#    bash uninstall.sh --prefix /custom/path
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
TAU_PREFIX="${HOME}/.tau"
SKIP_CONFIRM=false

# ── Colours ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' DIM='' RESET=''
fi

info()    { printf "${CYAN}  ℹ ${RESET}%s\n" "$*"; }
success() { printf "${GREEN}  ✓ ${RESET}%s\n" "$*"; }
warn()    { printf "${YELLOW}  ⚠ ${RESET}%s\n" "$*"; }
error()   { printf "${RED}  ✗ ${RESET}%s\n" "$*" >&2; }

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)    TAU_PREFIX="$2"; shift 2 ;;
        -y|--yes)    SKIP_CONFIRM=true; shift ;;
        --help|-h)
            echo "Usage: uninstall.sh [--prefix <path>] [-y|--yes]"
            exit 0
            ;;
        *)
            error "Unknown option: $1"; exit 1 ;;
    esac
done

TAU_BIN="${HOME}/.local/bin"
MARKER="# Added by tau installer"

# ── Show what will be removed ─────────────────────────────────────────
printf "\n"
printf "${BOLD}${RED}  tau uninstaller${RESET}\n"
printf "\n"
printf "  The following will be removed:\n"
printf "\n"

items=()
if [[ -d "$TAU_PREFIX" ]]; then
    size=$(du -sh "$TAU_PREFIX" 2>/dev/null | cut -f1)
    printf "    ${BOLD}•${RESET} ${TAU_PREFIX}/    ${DIM}(${size})${RESET}\n"
    items+=("dir:$TAU_PREFIX")
fi

if [[ -f "${TAU_BIN}/tau" ]]; then
    printf "    ${BOLD}•${RESET} ${TAU_BIN}/tau   ${DIM}(wrapper script)${RESET}\n"
    items+=("file:${TAU_BIN}/tau")
fi

# Check for PATH entries in rc files
rc_files=("$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.config/fish/config.fish")
modified_rcs=()
for rc in "${rc_files[@]}"; do
    if [[ -f "$rc" ]] && grep -q "$MARKER" "$rc" 2>/dev/null; then
        printf "    ${BOLD}•${RESET} PATH entry in $(basename "$rc")   ${DIM}(will be removed)${RESET}\n"
        modified_rcs+=("$rc")
    fi
done

if [[ ${#items[@]} -eq 0 && ${#modified_rcs[@]} -eq 0 ]]; then
    printf "    ${DIM}Nothing to remove — tau doesn't appear to be installed.${RESET}\n"
    printf "\n"
    exit 0
fi

printf "\n"

# ── Confirm ───────────────────────────────────────────────────────────
if [[ "$SKIP_CONFIRM" != true ]]; then
    printf "  ${YELLOW}Are you sure? This cannot be undone.${RESET}\n"
    printf "  Type ${BOLD}yes${RESET} to confirm: "
    read -r answer
    if [[ "$answer" != "yes" ]]; then
        info "Aborted."
        exit 0
    fi
    printf "\n"
fi

# ── Remove ────────────────────────────────────────────────────────────

# 1. Remove wrapper script
if [[ -f "${TAU_BIN}/tau" ]]; then
    rm -f "${TAU_BIN}/tau"
    success "Removed ${TAU_BIN}/tau"
fi

# 2. Remove PATH entries from shell rc files
for rc in "${modified_rcs[@]}"; do
    # Remove the marker line and the line after it (the export line)
    # Also remove any blank line before the marker
    if [[ "$(uname -s)" == "Darwin" ]]; then
        # macOS sed requires '' after -i
        sed -i '' "/${MARKER}/,+1d" "$rc" 2>/dev/null || true
        # For fish config (multi-line block)
        sed -i '' '/Added by tau installer/,/^end$/d' "$rc" 2>/dev/null || true
    else
        sed -i "/${MARKER}/,+1d" "$rc" 2>/dev/null || true
        sed -i '/Added by tau installer/,/^end$/d' "$rc" 2>/dev/null || true
    fi
    # Clean up trailing blank lines
    if [[ "$(uname -s)" == "Darwin" ]]; then
        sed -i '' -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$rc" 2>/dev/null || true
    else
        sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$rc" 2>/dev/null || true
    fi
    success "Cleaned PATH entry from $(basename "$rc")"
done

# 3. Remove tau directory (the big one — last)
if [[ -d "$TAU_PREFIX" ]]; then
    # Keep config if it exists outside the prefix (shouldn't, but safety)
    rm -rf "$TAU_PREFIX"
    success "Removed ${TAU_PREFIX}"
fi

printf "\n"
printf "${BOLD}${GREEN}  ✓  tau has been uninstalled.${RESET}\n"
printf "\n"
printf "  ${DIM}Your API keys (OPENAI_API_KEY, etc.) are still in your environment.${RESET}\n"
printf "  ${DIM}Remove them from your shell rc file if no longer needed.${RESET}\n"
printf "\n"
