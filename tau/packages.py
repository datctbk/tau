"""Extension package manager for tau.

Supports installing packages containing extensions, skills, prompts, and themes from:
  - npm packages:  tau install npm:<package>[@version]
  - git repos:     tau install git:<url>[@ref]
  - https URLs:    tau install https://<url>[@ref]
  - ssh URLs:      tau install ssh://<url>[@ref]

Installed packages are tracked in ~/.tau/packages/manifest.json.
Resource paths (extensions, skills, prompts, themes) are auto-discovered from
conventional subdirectories or a ``tau.json`` / ``package.json`` manifest.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PACKAGES_DIR = Path.home() / ".tau" / "packages"
MANIFEST_PATH = PACKAGES_DIR / "manifest.json"
GIT_DIR = PACKAGES_DIR / "git"
NPM_DIR = PACKAGES_DIR / "npm"

# Resource type names recognised in package manifests / conventional dirs
RESOURCE_TYPES = ("extensions", "skills", "prompts", "themes")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------

def _parse_source(source: str) -> tuple[str, str, str | None]:
    """Parse a source string into (scheme, url_or_name, ref).

    Supported forms::

        npm:<package>[@version]
        git:<url>[@ref]
        https://<url>[@ref]
        ssh://<url>[@ref]
        git+https://<url>[@ref]   (alias for git:)
        git+ssh://<url>[@ref]     (alias for git:)

    Returns the *ref* (version / git tag / commit) if present, else None.

    For git/@ref detection: an ``@`` that appears AFTER the last ``/`` in
    the URL is treated as a ref separator (not user@host authentication).
    """
    ref: str | None = None

    # Normalise git+https / git+ssh aliases
    if source.startswith(("git+https://", "git+ssh://")):
        source = source[4:]   # strip "git+"

    if source.startswith("npm:"):
        rest = source[4:]
        # Handle scoped packages: @scope/name@version
        if rest.startswith("@"):
            at = rest.rfind("@", 1)
            if at > 0:
                ref = rest[at + 1:]
                rest = rest[:at]
        elif "@" in rest:
            name_part, ref = rest.rsplit("@", 1)
            rest = name_part
        return ("npm", rest, ref)

    for scheme in ("git:", "https://", "ssh://"):
        if source.startswith(scheme):
            # For "git:" the URL follows the colon; https/ssh: scheme is part of the URL
            url = source[len(scheme):] if scheme == "git:" else source
            # Detect @ref: an @ that appears after the last / is a ref tag
            last_at = url.rfind("@")
            last_slash = url.rfind("/")
            if last_at >= 0 and last_at > last_slash:
                ref = url[last_at + 1:]
                url = url[:last_at]
            return (scheme.rstrip(":/"), url, ref)

    raise PackageError(
        f"Unknown source format: {source!r}. "
        "Use 'npm:<pkg>', 'git:<url>', 'https://<url>', or 'ssh://<url>'."
    )


def _name_from_url(url: str) -> str:
    """Derive a safe package name from a URL or npm package name."""
    # Scoped npm package: @scope/pkg → scope__pkg
    if url.startswith("@"):
        name = url.lstrip("@").replace("/", "__")
    else:
        name = url.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        name = name.lstrip("@").replace("/", "__")
    return name.replace("-", "_").lower()


# ---------------------------------------------------------------------------
# Resource discovery
# ---------------------------------------------------------------------------

def _discover_resources(install_path: Path) -> dict[str, list[str]]:
    """Return ``{resource_type: [abs_path, ...]}`` for a package directory.

    Discovery order (first match wins per type):
    1. ``tau.json`` at the package root with a ``{"extensions": [...], "skills": [...], ...}`` key
    2. ``package.json`` at the package root with a nested ``"tau"`` key
    3. Conventional subdirectories: ``extensions/``, ``skills/``, ``prompts/``, ``themes/``
    """
    resources: dict[str, list[str]] = {t: [] for t in RESOURCE_TYPES}

    # 1. tau.json
    tau_json = install_path / "tau.json"
    if tau_json.is_file():
        try:
            data = json.loads(tau_json.read_text(encoding="utf-8"))
            for t in RESOURCE_TYPES:
                for rel in data.get(t, []):
                    p = (install_path / rel).resolve()
                    if p.is_dir():
                        resources[t].append(str(p))
            return resources
        except Exception:
            pass

    # 2. package.json with "tau" key
    pkg_json = install_path / "package.json"
    if pkg_json.is_file():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            tau_manifest = data.get("tau", {})
            if tau_manifest:
                for t in RESOURCE_TYPES:
                    for rel in tau_manifest.get(t, []):
                        p = (install_path / rel).resolve()
                        if p.is_dir():
                            resources[t].append(str(p))
                return resources
        except Exception:
            pass

    # 3. Conventional subdirectories
    for t in RESOURCE_TYPES:
        subdir = install_path / t
        if subdir.is_dir():
            resources[t].append(str(subdir))
    # If no conventional subdir exists, treat the root itself as extensions
    if not any(resources[t] for t in RESOURCE_TYPES):
        resources["extensions"].append(str(install_path))

    return resources


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class InstalledPackage:
    """Metadata about one installed package."""
    name: str
    source: str
    install_path: str
    installed_at: str
    version: str
    enabled: bool = True
    resources: dict[str, list[str]] = field(default_factory=lambda: {t: [] for t in RESOURCE_TYPES})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstalledPackage":
        # Back-compat: old manifests lack enabled / resources
        d.setdefault("enabled", True)
        d.setdefault("resources", {t: [] for t in RESOURCE_TYPES})
        return cls(**d)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PackageError(Exception):
    pass


class PackageAlreadyInstalledError(PackageError):
    pass


class PackageNotFoundError(PackageError):
    pass


# ---------------------------------------------------------------------------
# PackageManager
# ---------------------------------------------------------------------------

class PackageManager:
    """Manages installation, removal, and updating of tau packages."""

    def __init__(
        self,
        packages_dir: Path = PACKAGES_DIR,
        manifest_path: Path | None = None,
    ) -> None:
        self._packages_dir = packages_dir
        self._git_dir = packages_dir / "git"
        self._npm_dir = packages_dir / "npm"
        self._manifest_path = manifest_path or (packages_dir / "manifest.json")

        self._git_dir.mkdir(parents=True, exist_ok=True)
        self._npm_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self._manifest_path.exists():
            return {}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return data.get("packages", {})
        except Exception:
            logger.warning("Could not parse manifest at %s", self._manifest_path)
            return {}

    def _save_manifest(self, packages: dict[str, dict[str, Any]]) -> None:
        self._manifest_path.write_text(
            json.dumps({"packages": packages}, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def install(self, source: str) -> InstalledPackage:
        """Install a package from *source*.

        Accepted source formats::

            npm:<package>[@version]
            git:<url>[@ref]
            https://<url>[@ref]
            ssh://<url>[@ref]
        """
        scheme, url_or_name, ref = _parse_source(source)
        if scheme == "npm":
            return self._install_npm(url_or_name, ref, source)
        else:
            return self._install_git(url_or_name, ref, source)

    def _install_git(self, url: str, ref: str | None, original_source: str) -> InstalledPackage:
        name = _name_from_url(url)
        manifest = self._load_manifest()
        if name in manifest:
            raise PackageAlreadyInstalledError(
                f"Package {name!r} is already installed. Remove it first."
            )

        dest = self._git_dir / name
        try:
            cmd = ["git", "clone", "--depth", "1"]
            if ref:
                cmd += ["--branch", ref]
            cmd += [url, str(dest)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise PackageError(f"git clone failed: {exc.stderr.strip()}") from exc

        version = self._git_head(dest)
        resources = _discover_resources(dest)
        pkg = InstalledPackage(
            name=name,
            source=original_source,
            install_path=str(dest),
            installed_at=_now_iso(),
            version=version,
            resources=resources,
        )
        manifest[name] = pkg.to_dict()
        self._save_manifest(manifest)
        logger.info("Installed git package %r from %s", name, url)
        return pkg

    def _install_npm(self, package_name: str, version: str | None, original_source: str) -> InstalledPackage:
        versioned = f"{package_name}@{version}" if version else package_name
        name = _name_from_url(package_name)
        manifest = self._load_manifest()
        if name in manifest:
            raise PackageAlreadyInstalledError(
                f"Package {name!r} is already installed. Remove it first."
            )

        dest = self._npm_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["npm", "install", "--prefix", str(dest), versioned],
                check=True, capture_output=True, text=True,
            )
        except FileNotFoundError as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise PackageError("npm is not installed or not on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise PackageError(f"npm install failed: {exc.stderr.strip()}") from exc

        # npm installs into dest/node_modules/<package>
        pkg_dir = dest / "node_modules" / package_name
        if not pkg_dir.is_dir():
            shutil.rmtree(dest, ignore_errors=True)
            raise PackageError(f"npm install succeeded but package dir not found: {pkg_dir}")

        resolved_version = self._npm_version(dest, package_name)
        resources = _discover_resources(pkg_dir)
        pkg = InstalledPackage(
            name=name,
            source=original_source,
            install_path=str(pkg_dir),
            installed_at=_now_iso(),
            version=resolved_version,
            resources=resources,
        )
        manifest[name] = pkg.to_dict()
        self._save_manifest(manifest)
        logger.info("Installed npm package %r (%s)", name, resolved_version)
        return pkg

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(self, name: str) -> None:
        """Remove a package by name."""
        manifest = self._load_manifest()
        if name not in manifest:
            raise PackageNotFoundError(f"Package {name!r} is not installed.")

        pkg = InstalledPackage.from_dict(manifest[name])
        install_path = Path(pkg.install_path)
        # For npm packages the prefix dir is install_path/../../../ (node_modules/name)
        # We remove the root prefix (dest = npm_dir/name) not just the node_modules leaf
        npm_root = self._npm_dir / name
        if npm_root.is_dir() and str(install_path).startswith(str(npm_root)):
            shutil.rmtree(npm_root)
        elif install_path.is_dir():
            shutil.rmtree(install_path)

        del manifest[name]
        self._save_manifest(manifest)
        logger.info("Removed package %r", name)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, name: str | None = None) -> list[str]:
        """Update one package by name, or all packages if name is None.

        Pinned packages (version was explicitly specified in the source
        ``@ref``) are skipped unless a specific name is provided.
        Returns list of updated package names.
        """
        manifest = self._load_manifest()
        updated: list[str] = []
        targets = [name] if name else list(manifest.keys())
        for pkg_name in targets:
            if pkg_name not in manifest:
                raise PackageNotFoundError(f"Package {pkg_name!r} is not installed.")
            pkg = InstalledPackage.from_dict(manifest[pkg_name])
            # Skip pinned packages when updating all
            if name is None and self._is_pinned(pkg):
                logger.info("Skipping pinned package %r", pkg_name)
                continue
            try:
                if pkg.source.startswith("git:") or pkg.source.startswith(("https://", "ssh://")):
                    self._update_git(pkg)
                elif pkg.source.startswith("npm:"):
                    self._update_npm(pkg)
                pkg.version = self._get_version(pkg)
                pkg.resources = _discover_resources(Path(pkg.install_path))
                manifest[pkg_name] = pkg.to_dict()
                updated.append(pkg_name)
            except Exception as exc:
                logger.warning("Failed to update %r: %s", pkg_name, exc)

        self._save_manifest(manifest)
        return updated

    def _is_pinned(self, pkg: InstalledPackage) -> bool:
        """Return True if the source contains an explicit @ref (version pin)."""
        _, _, ref = _parse_source(pkg.source)
        return ref is not None

    def _update_git(self, pkg: InstalledPackage) -> None:
        dest = Path(pkg.install_path)
        if not dest.exists():
            raise PackageError(f"Install path {dest} does not exist.")
        try:
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--ff-only"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise PackageError(f"git pull failed: {exc.stderr.strip()}") from exc

    def _update_npm(self, pkg: InstalledPackage) -> None:
        _, pkg_name, version = _parse_source(pkg.source)
        versioned = f"{pkg_name}@{version}" if version else pkg_name
        npm_root = self._npm_dir / pkg.name
        try:
            subprocess.run(
                ["npm", "install", "--prefix", str(npm_root), versioned],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise PackageError(f"npm update failed: {exc.stderr.strip()}") from exc

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self, name: str) -> None:
        """Mark a package as enabled."""
        manifest = self._load_manifest()
        if name not in manifest:
            raise PackageNotFoundError(f"Package {name!r} is not installed.")
        manifest[name]["enabled"] = True
        self._save_manifest(manifest)

    def disable(self, name: str) -> None:
        """Mark a package as disabled (not loaded at runtime)."""
        manifest = self._load_manifest()
        if name not in manifest:
            raise PackageNotFoundError(f"Package {name!r} is not installed.")
        manifest[name]["enabled"] = False
        self._save_manifest(manifest)

    # ------------------------------------------------------------------
    # List / resource paths
    # ------------------------------------------------------------------

    def list_packages(self) -> list[InstalledPackage]:
        manifest = self._load_manifest()
        return [InstalledPackage.from_dict(d) for d in manifest.values()]

    def get_resource_paths(self, resource_type: str) -> list[str]:
        """Return paths for *resource_type* across all enabled installed packages.

        *resource_type* is one of ``"extensions"``, ``"skills"``,
        ``"prompts"``, ``"themes"``.
        """
        paths: list[str] = []
        for pkg in self.list_packages():
            if not pkg.enabled:
                continue
            for p in pkg.resources.get(resource_type, []):
                if Path(p).is_dir():
                    paths.append(p)
        return paths

    def get_extension_paths(self) -> list[str]:
        """Back-compat alias for get_resource_paths('extensions')."""
        return self.get_resource_paths("extensions")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _git_head(self, path: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()[:12]
        except Exception:
            return "unknown"

    def _npm_version(self, npm_root: Path, package_name: str) -> str:
        """Read version from node_modules/<name>/package.json."""
        pkg_json = npm_root / "node_modules" / package_name / "package.json"
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            return data.get("version", "unknown")
        except Exception:
            return "unknown"

    def _pip_version(self, target_dir: Path, package_name: str) -> str:
        normalized = package_name.replace("-", "_").lower()
        for d in target_dir.iterdir():
            if d.is_dir() and d.name.endswith(".dist-info"):
                dist_name = d.name.rsplit("-", 1)
                if len(dist_name) == 2 and dist_name[0].lower().replace("-", "_") == normalized:
                    return dist_name[1].replace(".dist-info", "")
        return "unknown"

    def _get_version(self, pkg: InstalledPackage) -> str:
        dest = Path(pkg.install_path)
        if pkg.source.startswith("npm:"):
            _, pkg_name, _ = _parse_source(pkg.source)
            npm_root = self._npm_dir / pkg.name
            return self._npm_version(npm_root, pkg_name)
        return self._git_head(dest)
