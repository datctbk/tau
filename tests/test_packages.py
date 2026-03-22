"""Tests for the extension package manager (tau/packages.py)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tau.packages import (
    InstalledPackage,
    PackageAlreadyInstalledError,
    PackageError,
    PackageManager,
    PackageNotFoundError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pm(tmp_path: Path) -> PackageManager:
    """Create a PackageManager with temp dirs."""
    return PackageManager(
        packages_dir=tmp_path / "packages",
        manifest_path=tmp_path / "packages" / "manifest.json",
    )


def _seed_manifest(tmp_path: Path, packages: dict) -> None:
    """Pre-populate a manifest file."""
    manifest_path = tmp_path / "packages" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"packages": packages}, indent=2), encoding="utf-8"
    )


def _fake_installed_pkg(tmp_path: Path, name: str = "my_ext", source: str = "git:https://example.com/repo") -> dict:
    """Return a minimal InstalledPackage dict and create the install dir."""
    install_path = tmp_path / "packages" / "git" / name
    install_path.mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "source": source,
        "install_path": str(install_path),
        "installed_at": "2026-01-01T00:00:00+00:00",
        "version": "abc123",
    }


# ===========================================================================
# InstalledPackage dataclass
# ===========================================================================

class TestInstalledPackage:
    def test_to_dict_roundtrip(self):
        pkg = InstalledPackage(
            name="test",
            source="git:https://example.com/repo",
            install_path="/tmp/test",
            installed_at="2026-01-01T00:00:00+00:00",
            version="abc123",
        )
        d = pkg.to_dict()
        restored = InstalledPackage.from_dict(d)
        assert restored.name == pkg.name
        assert restored.source == pkg.source
        assert restored.version == pkg.version


# ===========================================================================
# Manifest I/O
# ===========================================================================

class TestManifestIO:
    def test_empty_manifest_returns_empty_list(self, tmp_path):
        pm = _pm(tmp_path)
        assert pm.list_packages() == []

    def test_save_and_load_manifest(self, tmp_path):
        pm = _pm(tmp_path)
        pkg_data = _fake_installed_pkg(tmp_path)
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        packages = pm.list_packages()
        assert len(packages) == 1
        assert packages[0].name == "my_ext"

    def test_corrupt_manifest_returns_empty(self, tmp_path):
        pm = _pm(tmp_path)
        manifest_path = tmp_path / "packages" / "manifest.json"
        manifest_path.write_text("not json at all{{{", encoding="utf-8")
        assert pm.list_packages() == []


# ===========================================================================
# Install — git
# ===========================================================================

class TestInstallGit:
    @patch("tau.packages.subprocess.run")
    def test_install_git_calls_clone(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123def456\n", stderr="")
        pm = _pm(tmp_path)
        pkg = pm.install("git:https://github.com/user/my-ext")
        # Verify git clone was called
        clone_call = mock_run.call_args_list[0]
        assert clone_call[0][0][0] == "git"
        assert clone_call[0][0][1] == "clone"
        assert "https://github.com/user/my-ext" in clone_call[0][0]
        assert pkg.name == "my_ext"
        assert pkg.source == "git:https://github.com/user/my-ext"

    @patch("tau.packages.subprocess.run")
    def test_install_git_strips_dotgit(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        pkg = pm.install("git:https://github.com/user/my-ext.git")
        assert pkg.name == "my_ext"

    @patch("tau.packages.subprocess.run")
    def test_install_git_duplicate_raises(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        pm.install("git:https://github.com/user/my-ext")
        with pytest.raises(PackageAlreadyInstalledError):
            pm.install("git:https://github.com/user/my-ext")

    @patch("tau.packages.subprocess.run")
    def test_install_git_clone_failure_raises(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "git", stderr="fatal: repo not found"
        )
        pm = _pm(tmp_path)
        with pytest.raises(PackageError, match="git clone failed"):
            pm.install("git:https://github.com/user/nonexistent")

    @patch("tau.packages.subprocess.run")
    def test_install_git_saved_to_manifest(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123def456\n", stderr="")
        pm = _pm(tmp_path)
        pm.install("git:https://github.com/user/my-ext")
        packages = pm.list_packages()
        assert len(packages) == 1
        assert packages[0].name == "my_ext"


# ===========================================================================
# Install — pip
# ===========================================================================

class TestInstallPip:
    @patch("tau.packages.subprocess.run")
    def test_install_pip_calls_pip(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        pm = _pm(tmp_path)
        pkg = pm.install("pip:tau-ext-foobar")
        # Verify pip install was called
        pip_call = mock_run.call_args_list[0]
        cmd = pip_call[0][0]
        assert "-m" in cmd
        assert "pip" in cmd
        assert "install" in cmd
        assert "tau-ext-foobar" in cmd
        assert pkg.name == "tau_ext_foobar"

    @patch("tau.packages.subprocess.run")
    def test_install_pip_duplicate_raises(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        pm = _pm(tmp_path)
        pm.install("pip:tau-ext-foobar")
        with pytest.raises(PackageAlreadyInstalledError):
            pm.install("pip:tau-ext-foobar")


# ===========================================================================
# Install — bad source
# ===========================================================================

class TestInstallBadSource:
    def test_unknown_source_raises(self, tmp_path):
        pm = _pm(tmp_path)
        with pytest.raises(PackageError, match="Unknown source format"):
            pm.install("npm:some-package")


# ===========================================================================
# Remove
# ===========================================================================

class TestRemove:
    def test_remove_deletes_from_manifest(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path)
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        pm.remove("my_ext")
        assert pm.list_packages() == []

    def test_remove_deletes_directory(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path)
        install_path = Path(pkg_data["install_path"])
        assert install_path.exists()
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        pm.remove("my_ext")
        assert not install_path.exists()

    def test_remove_nonexistent_raises(self, tmp_path):
        pm = _pm(tmp_path)
        with pytest.raises(PackageNotFoundError):
            pm.remove("does_not_exist")


# ===========================================================================
# Update
# ===========================================================================

class TestUpdate:
    @patch("tau.packages.subprocess.run")
    def test_update_git_calls_pull(self, mock_run, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path, "my_ext", "git:https://example.com/repo")
        # Create a .git dir so it looks like a real git repo
        git_dir = Path(pkg_data["install_path"]) / ".git"
        git_dir.mkdir()
        _seed_manifest(tmp_path, {"my_ext": pkg_data})

        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        updated = pm.update("my_ext")
        assert "my_ext" in updated
        # Check git pull was called
        pull_call = mock_run.call_args_list[0]
        cmd = pull_call[0][0]
        assert "git" in cmd
        assert "pull" in cmd

    @patch("tau.packages.subprocess.run")
    def test_update_pip_calls_upgrade(self, mock_run, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path, "my_pip_ext", "pip:my-pip-ext")
        _seed_manifest(tmp_path, {"my_pip_ext": pkg_data})

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        pm = _pm(tmp_path)
        updated = pm.update("my_pip_ext")
        assert "my_pip_ext" in updated
        pip_call = mock_run.call_args_list[0]
        cmd = pip_call[0][0]
        assert "--upgrade" in cmd

    def test_update_nonexistent_raises(self, tmp_path):
        pm = _pm(tmp_path)
        with pytest.raises(PackageNotFoundError):
            pm.update("ghost")

    @patch("tau.packages.subprocess.run")
    def test_update_all_when_no_name(self, mock_run, tmp_path):
        pkg1 = _fake_installed_pkg(tmp_path, "ext1", "git:https://example.com/ext1")
        pkg2 = _fake_installed_pkg(tmp_path, "ext2", "git:https://example.com/ext2")
        _seed_manifest(tmp_path, {"ext1": pkg1, "ext2": pkg2})

        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        updated = pm.update(None)
        assert set(updated) == {"ext1", "ext2"}


# ===========================================================================
# get_extension_paths
# ===========================================================================

class TestGetExtensionPaths:
    def test_returns_paths_for_installed_packages(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path)
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        paths = pm.get_extension_paths()
        assert len(paths) == 1
        assert paths[0] == pkg_data["install_path"]

    def test_skips_missing_directories(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path)
        # Delete the directory to simulate a broken install
        import shutil
        shutil.rmtree(pkg_data["install_path"])
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        paths = pm.get_extension_paths()
        assert len(paths) == 0

    def test_empty_manifest_returns_empty_paths(self, tmp_path):
        pm = _pm(tmp_path)
        assert pm.get_extension_paths() == []


# ===========================================================================
# Integration: ExtensionRegistry picks up packages
# ===========================================================================

class TestExtensionRegistryIntegration:
    def test_registry_discovers_package_extension(self, tmp_path):
        """
        Install a fake extension as a package and verify ExtensionRegistry loads it.
        """
        # Create a fake package dir with an extension file
        pkg_dir = tmp_path / "packages" / "git" / "test_pkg"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "my_ext.py").write_text(
            'from tau.core.extension import Extension\n'
            'from tau.core.types import ExtensionManifest\n'
            '\n'
            'class _E(Extension):\n'
            '    manifest = ExtensionManifest(name="pkg_test_ext", description="from package")\n'
            '\n'
            'EXTENSION = _E()\n',
            encoding="utf-8",
        )

        # Seed manifest
        _seed_manifest(tmp_path, {
            "test_pkg": {
                "name": "test_pkg",
                "source": "git:https://example.com/test_pkg",
                "install_path": str(pkg_dir),
                "installed_at": "2026-01-01T00:00:00+00:00",
                "version": "abc123",
            }
        })

        # Create a PackageManager pointing to our temp dirs
        pm = _pm(tmp_path)

        # Build ExtensionRegistry with the package paths
        from tau.core.extension import ExtensionRegistry
        from tau.core.context import ContextManager
        from tau.core.tool_registry import ToolRegistry
        from tau.core.types import AgentConfig

        reg = ExtensionRegistry(
            extra_paths=pm.get_extension_paths(),
            include_builtins=False,
        )
        r = ToolRegistry()
        c = ContextManager(AgentConfig(compaction_enabled=False, retry_enabled=False))
        loaded = reg.load_all(r, c, steering=None, console_print=lambda _: None)
        assert "pkg_test_ext" in loaded
