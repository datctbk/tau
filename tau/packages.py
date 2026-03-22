"""Extension package manager for tau.

Supports installing extensions from:
  - git repos:   tau extensions install git:https://github.com/user/my-ext
  - pip packages: tau extensions install pip:some-extension-package

Installed packages are tracked in ~/.tau/packages/manifest.json and
auto-discovered by ExtensionRegistry via get_extension_paths().
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PACKAGES_DIR = Path.home() / ".tau" / "packages"
MANIFEST_PATH = PACKAGES_DIR / "manifest.json"
GIT_DIR = PACKAGES_DIR / "git"
PIP_DIR = PACKAGES_DIR / "pip"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class InstalledPackage:
    """Metadata about one installed extension package."""
    name: str             # derived from repo/package name
    source: str           # "git:https://..." or "pip:package-name"
    install_path: str     # absolute path to installed location
    installed_at: str     # ISO timestamp
    version: str          # git commit hash or pip version

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstalledPackage":
        return cls(**d)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PackageError(Exception):
    """Raised for any package manager error."""
    pass


class PackageAlreadyInstalledError(PackageError):
    pass


class PackageNotFoundError(PackageError):
    pass


# ---------------------------------------------------------------------------
# PackageManager
# ---------------------------------------------------------------------------

class PackageManager:
    """Manages installation, removal, and updating of extension packages."""

    def __init__(
        self,
        packages_dir: Path = PACKAGES_DIR,
        manifest_path: Path | None = None,
    ) -> None:
        self._packages_dir = packages_dir
        self._git_dir = packages_dir / "git"
        self._pip_dir = packages_dir / "pip"
        self._manifest_path = manifest_path or (packages_dir / "manifest.json")

        # Ensure dirs exist
        self._git_dir.mkdir(parents=True, exist_ok=True)
        self._pip_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Load manifest as {name: package_dict}."""
        if not self._manifest_path.exists():
            return {}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return data.get("packages", {})
        except Exception:
            logger.warning("Could not parse manifest at %s", self._manifest_path)
            return {}

    def _save_manifest(self, packages: dict[str, dict[str, Any]]) -> None:
        """Persist manifest to disk."""
        self._manifest_path.write_text(
            json.dumps({"packages": packages}, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def install(self, source: str) -> InstalledPackage:
        """
        Install a package from the given source.

        Supported prefixes:
          - ``git:<url>``   e.g. ``git:https://github.com/user/repo``
          - ``pip:<name>``  e.g. ``pip:tau-ext-foobar``
        """
        if source.startswith("git:"):
            return self._install_git(source[4:])
        elif source.startswith("pip:"):
            return self._install_pip(source[4:])
        else:
            raise PackageError(
                f"Unknown source format: {source!r}. "
                "Use 'git:<url>' or 'pip:<package>'."
            )

    def _install_git(self, url: str) -> InstalledPackage:
        # Derive name from URL (last path component minus .git)
        name = url.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        name = name.replace("-", "_").lower()

        manifest = self._load_manifest()
        if name in manifest:
            raise PackageAlreadyInstalledError(
                f"Package {name!r} is already installed. "
                "Remove it first with 'tau extensions remove'."
            )

        dest = self._git_dir / name
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise PackageError(f"git clone failed: {exc.stderr.strip()}") from exc

        # Get commit hash
        version = self._git_head(dest)

        pkg = InstalledPackage(
            name=name,
            source=f"git:{url}",
            install_path=str(dest),
            installed_at=_now_iso(),
            version=version,
        )
        manifest[name] = pkg.to_dict()
        self._save_manifest(manifest)
        logger.info("Installed git package %r from %s", name, url)
        return pkg

    def _install_pip(self, package_name: str) -> InstalledPackage:
        name = package_name.replace("-", "_").lower()

        manifest = self._load_manifest()
        if name in manifest:
            raise PackageAlreadyInstalledError(
                f"Package {name!r} is already installed. "
                "Remove it first with 'tau extensions remove'."
            )

        dest = self._pip_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "--target", str(dest), package_name],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            # Cleanup on failure
            if dest.exists():
                shutil.rmtree(dest)
            raise PackageError(f"pip install failed: {exc.stderr.strip()}") from exc

        # Get installed version
        version = self._pip_version(dest, package_name)

        pkg = InstalledPackage(
            name=name,
            source=f"pip:{package_name}",
            install_path=str(dest),
            installed_at=_now_iso(),
            version=version,
        )
        manifest[name] = pkg.to_dict()
        self._save_manifest(manifest)
        logger.info("Installed pip package %r", name)
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
        if install_path.exists():
            shutil.rmtree(install_path)

        del manifest[name]
        self._save_manifest(manifest)
        logger.info("Removed package %r", name)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, name: str | None = None) -> list[str]:
        """
        Update one package (by name) or all installed packages.
        Returns list of names that were updated.
        """
        manifest = self._load_manifest()
        updated: list[str] = []

        targets = [name] if name else list(manifest.keys())
        for pkg_name in targets:
            if pkg_name not in manifest:
                raise PackageNotFoundError(f"Package {pkg_name!r} is not installed.")
            pkg = InstalledPackage.from_dict(manifest[pkg_name])
            try:
                if pkg.source.startswith("git:"):
                    self._update_git(pkg)
                elif pkg.source.startswith("pip:"):
                    self._update_pip(pkg)
                pkg.version = self._get_version(pkg)
                manifest[pkg_name] = pkg.to_dict()
                updated.append(pkg_name)
            except Exception as exc:
                logger.warning("Failed to update %r: %s", pkg_name, exc)

        self._save_manifest(manifest)
        return updated

    def _update_git(self, pkg: InstalledPackage) -> None:
        dest = Path(pkg.install_path)
        if not dest.exists():
            raise PackageError(f"Install path {dest} does not exist.")
        try:
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--ff-only"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise PackageError(f"git pull failed: {exc.stderr.strip()}") from exc

    def _update_pip(self, pkg: InstalledPackage) -> None:
        pip_name = pkg.source[4:]  # strip "pip:" prefix
        dest = Path(pkg.install_path)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "--upgrade", "--target", str(dest), pip_name],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise PackageError(f"pip upgrade failed: {exc.stderr.strip()}") from exc

    # ------------------------------------------------------------------
    # List / query
    # ------------------------------------------------------------------

    def list_packages(self) -> list[InstalledPackage]:
        """Return all installed packages."""
        manifest = self._load_manifest()
        return [InstalledPackage.from_dict(d) for d in manifest.values()]

    def get_extension_paths(self) -> list[str]:
        """
        Return filesystem paths that ExtensionRegistry should scan.
        For git packages this is the install directory itself.
        For pip packages this is the install directory.
        """
        paths: list[str] = []
        for pkg in self.list_packages():
            p = Path(pkg.install_path)
            if p.is_dir():
                paths.append(str(p))
        return paths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _git_head(self, path: Path) -> str:
        """Get the HEAD commit hash for a git repo."""
        try:
            result = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()[:12]
        except Exception:
            return "unknown"

    def _pip_version(self, target_dir: Path, package_name: str) -> str:
        """Try to extract the installed version from dist-info."""
        normalized = package_name.replace("-", "_").lower()
        for d in target_dir.iterdir():
            if d.is_dir() and d.name.endswith(".dist-info"):
                dist_name = d.name.rsplit("-", 1)
                if len(dist_name) == 2 and dist_name[0].lower().replace("-", "_") == normalized:
                    return dist_name[1].replace(".dist-info", "")
        return "unknown"

    def _get_version(self, pkg: InstalledPackage) -> str:
        """Get current version of an installed package."""
        dest = Path(pkg.install_path)
        if pkg.source.startswith("git:"):
            return self._git_head(dest)
        elif pkg.source.startswith("pip:"):
            pip_name = pkg.source[4:]
            return self._pip_version(dest, pip_name)
        return "unknown"
