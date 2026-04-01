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
    _discover_resources,
    _name_from_url,
    _parse_source,
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


def _fake_installed_pkg(
    tmp_path: Path,
    name: str = "my_ext",
    source: str = "git:https://example.com/repo",
    enabled: bool = True,
) -> dict:
    """Return an InstalledPackage dict and create the install dir.

    The resources dict points extensions at the install_path itself,
    matching what _discover_resources returns for a bare directory.
    """
    install_path = tmp_path / "packages" / "git" / name
    install_path.mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "source": source,
        "install_path": str(install_path),
        "installed_at": "2026-01-01T00:00:00+00:00",
        "version": "abc123",
        "enabled": enabled,
        "resources": {
            "extensions": [str(install_path)],
            "skills": [],
            "prompts": [],
            "themes": [],
        },
    }


# ===========================================================================
# _parse_source
# ===========================================================================

class TestParseSource:
    def test_npm_basic(self):
        scheme, name, ref = _parse_source("npm:tau-ext-foo")
        assert scheme == "npm"
        assert name == "tau-ext-foo"
        assert ref is None

    def test_npm_with_version(self):
        scheme, name, ref = _parse_source("npm:tau-ext-foo@1.2.3")
        assert scheme == "npm"
        assert name == "tau-ext-foo"
        assert ref == "1.2.3"

    def test_npm_scoped_with_version(self):
        scheme, name, ref = _parse_source("npm:@scope/pkg@2.0.0")
        assert scheme == "npm"
        assert name == "@scope/pkg"
        assert ref == "2.0.0"

    def test_git_basic(self):
        scheme, url, ref = _parse_source("git:https://github.com/user/repo")
        assert scheme == "git"
        assert "github.com/user/repo" in url
        assert ref is None

    def test_git_with_ref(self):
        scheme, url, ref = _parse_source("git:https://github.com/user/repo@v1.0.0")
        assert scheme == "git"
        assert ref == "v1.0.0"

    def test_https_basic(self):
        scheme, url, ref = _parse_source("https://github.com/user/repo")
        assert scheme == "https"
        assert ref is None

    def test_https_with_ref(self):
        scheme, url, ref = _parse_source("https://github.com/user/repo@main")
        assert scheme == "https"
        assert ref == "main"

    def test_git_plus_https_alias(self):
        scheme, url, ref = _parse_source("git+https://github.com/user/repo")
        assert scheme == "https"

    def test_unknown_source_raises(self):
        with pytest.raises(PackageError, match="Unknown source format"):
            _parse_source("ftp://example.com/pkg")

    def test_bare_string_raises(self):
        with pytest.raises(PackageError, match="Unknown source format"):
            _parse_source("just-a-name")


# ===========================================================================
# _name_from_url
# ===========================================================================

class TestNameFromUrl:
    def test_github_url(self):
        assert _name_from_url("https://github.com/user/my-ext") == "my_ext"

    def test_url_with_git_suffix(self):
        assert _name_from_url("https://github.com/user/my-ext.git") == "my_ext"

    def test_scoped_npm(self):
        assert _name_from_url("@scope/pkg") == "scope__pkg"


# ===========================================================================
# _discover_resources
# ===========================================================================

class TestDiscoverResources:
    def test_empty_dir_uses_root_as_extensions(self, tmp_path):
        resources = _discover_resources(tmp_path)
        assert str(tmp_path) in resources["extensions"]
        assert resources["skills"] == []
        assert resources["prompts"] == []

    def test_conventional_subdirs(self, tmp_path):
        (tmp_path / "extensions").mkdir()
        (tmp_path / "skills").mkdir()
        resources = _discover_resources(tmp_path)
        assert str(tmp_path / "extensions") in resources["extensions"]
        assert str(tmp_path / "skills") in resources["skills"]
        assert resources["prompts"] == []

    def test_tau_json_wins_over_convention(self, tmp_path):
        (tmp_path / "extensions").mkdir()
        custom = tmp_path / "my_ext"
        custom.mkdir()
        (tmp_path / "tau.json").write_text(
            json.dumps({"extensions": ["my_ext"]}), encoding="utf-8"
        )
        resources = _discover_resources(tmp_path)
        assert str(custom) in resources["extensions"]
        # conventional dir not present because tau.json takes over
        assert str(tmp_path / "extensions") not in resources["extensions"]

    def test_package_json_tau_key(self, tmp_path):
        custom = tmp_path / "my_skills"
        custom.mkdir()
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "test", "tau": {"skills": ["my_skills"]}}),
            encoding="utf-8",
        )
        resources = _discover_resources(tmp_path)
        assert str(custom) in resources["skills"]

    def test_tau_json_invalid_json_falls_back(self, tmp_path):
        (tmp_path / "tau.json").write_text("not json!!!", encoding="utf-8")
        (tmp_path / "extensions").mkdir()
        resources = _discover_resources(tmp_path)
        assert str(tmp_path / "extensions") in resources["extensions"]

    def test_nonexistent_path_in_tau_json_skipped(self, tmp_path):
        (tmp_path / "tau.json").write_text(
            json.dumps({"extensions": ["does_not_exist"]}), encoding="utf-8"
        )
        resources = _discover_resources(tmp_path)
        assert resources["extensions"] == []


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
            enabled=True,
            resources={"extensions": ["/tmp/test"], "skills": [], "prompts": [], "themes": []},
        )
        restored = InstalledPackage.from_dict(pkg.to_dict())
        assert restored.name == pkg.name
        assert restored.source == pkg.source
        assert restored.version == pkg.version
        assert restored.enabled == pkg.enabled
        assert restored.resources == pkg.resources

    def test_back_compat_without_enabled_resources(self):
        """Old manifests without enabled/resources fields should load fine."""
        d = {
            "name": "old_pkg",
            "source": "git:https://example.com/repo",
            "install_path": "/tmp/old",
            "installed_at": "2026-01-01T00:00:00+00:00",
            "version": "abc",
        }
        pkg = InstalledPackage.from_dict(d)
        assert pkg.enabled is True
        assert "extensions" in pkg.resources


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
        clone_call = mock_run.call_args_list[0]
        cmd = clone_call[0][0]
        assert cmd[0] == "git"
        assert cmd[1] == "clone"
        assert "https://github.com/user/my-ext" in cmd
        assert pkg.name == "my_ext"
        assert pkg.source == "git:https://github.com/user/my-ext"

    @patch("tau.packages.subprocess.run")
    def test_install_git_strips_dotgit(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        pkg = pm.install("git:https://github.com/user/my-ext.git")
        assert pkg.name == "my_ext"

    @patch("tau.packages.subprocess.run")
    def test_install_git_with_branch_ref(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        pm.install("git:https://github.com/user/my-ext@main")
        clone_call = mock_run.call_args_list[0]
        cmd = clone_call[0][0]
        assert "--branch" in cmd
        assert "main" in cmd

    @patch("tau.packages.subprocess.run")
    def test_install_https_source(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        pkg = pm.install("https://github.com/user/my-ext")
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
        assert packages[0].enabled is True


# ===========================================================================
# Install — npm
# ===========================================================================

class TestInstallNpm:
    @patch("tau.packages.subprocess.run")
    def test_install_npm_calls_npm_install(self, mock_run, tmp_path):
        pm = _pm(tmp_path)
        pkg_dir = pm._npm_dir / "tau_ext_foo" / "node_modules" / "tau-ext-foo"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "package.json").write_text(
            json.dumps({"name": "tau-ext-foo", "version": "1.0.0"}), encoding="utf-8"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        pkg = pm.install("npm:tau-ext-foo")
        npm_call = mock_run.call_args_list[0]
        cmd = npm_call[0][0]
        assert cmd[0] == "npm"
        assert "install" in cmd
        assert pkg.name == "tau_ext_foo"
        assert pkg.version == "1.0.0"

    @patch("tau.packages.subprocess.run")
    def test_install_npm_with_version(self, mock_run, tmp_path):
        pm = _pm(tmp_path)
        pkg_dir = pm._npm_dir / "tau_ext_foo" / "node_modules" / "tau-ext-foo"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "package.json").write_text(
            json.dumps({"name": "tau-ext-foo", "version": "2.3.4"}), encoding="utf-8"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        pkg = pm.install("npm:tau-ext-foo@2.3.4")
        npm_call = mock_run.call_args_list[0]
        cmd = npm_call[0][0]
        assert "tau-ext-foo@2.3.4" in cmd
        assert pkg.version == "2.3.4"

    @patch("tau.packages.subprocess.run")
    def test_install_npm_duplicate_raises(self, mock_run, tmp_path):
        pm = _pm(tmp_path)
        pkg_dir = pm._npm_dir / "tau_ext_foo" / "node_modules" / "tau-ext-foo"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "package.json").write_text(
            json.dumps({"name": "tau-ext-foo", "version": "1.0.0"}), encoding="utf-8"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        pm.install("npm:tau-ext-foo")
        with pytest.raises(PackageAlreadyInstalledError):
            pm.install("npm:tau-ext-foo")

    @patch("tau.packages.subprocess.run", side_effect=FileNotFoundError("npm not found"))
    def test_install_npm_not_installed_raises(self, mock_run, tmp_path):
        pm = _pm(tmp_path)
        with pytest.raises(PackageError, match="npm is not installed"):
            pm.install("npm:tau-ext-foo")

    @patch("tau.packages.subprocess.run")
    def test_install_npm_failure_raises(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, "npm", stderr="E404 not found")
        pm = _pm(tmp_path)
        with pytest.raises(PackageError, match="npm install failed"):
            pm.install("npm:no-such-package-xyz9999")


# ===========================================================================
# Install — bad source
# ===========================================================================

class TestInstallBadSource:
    def test_unknown_source_raises(self, tmp_path):
        pm = _pm(tmp_path)
        with pytest.raises(PackageError, match="Unknown source format"):
            pm.install("ftp://example.com/pkg")

    def test_bare_name_raises(self, tmp_path):
        pm = _pm(tmp_path)
        with pytest.raises(PackageError, match="Unknown source format"):
            pm.install("some-package-name")


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
        git_dir = Path(pkg_data["install_path"]) / ".git"
        git_dir.mkdir()
        _seed_manifest(tmp_path, {"my_ext": pkg_data})

        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        updated = pm.update("my_ext")
        assert "my_ext" in updated
        pull_call = mock_run.call_args_list[0]
        cmd = pull_call[0][0]
        assert "git" in cmd
        assert "pull" in cmd

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

    @patch("tau.packages.subprocess.run")
    def test_update_pinned_skipped(self, mock_run, tmp_path):
        """Packages with a pinned @ref are skipped when updating all."""
        pkg = _fake_installed_pkg(tmp_path, "pinned", "git:https://example.com/repo@v1.0.0")
        _seed_manifest(tmp_path, {"pinned": pkg})
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        pm = _pm(tmp_path)
        updated = pm.update(None)
        assert "pinned" not in updated


# ===========================================================================
# Enable / disable
# ===========================================================================

class TestEnableDisable:
    def test_disable_sets_enabled_false(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path)
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        pm.disable("my_ext")
        pkgs = pm.list_packages()
        assert pkgs[0].enabled is False

    def test_enable_sets_enabled_true(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path, enabled=False)
        pkg_data["enabled"] = False
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        pm.enable("my_ext")
        pkgs = pm.list_packages()
        assert pkgs[0].enabled is True

    def test_disable_nonexistent_raises(self, tmp_path):
        pm = _pm(tmp_path)
        with pytest.raises(PackageNotFoundError):
            pm.disable("ghost")

    def test_enable_nonexistent_raises(self, tmp_path):
        pm = _pm(tmp_path)
        with pytest.raises(PackageNotFoundError):
            pm.enable("ghost")


# ===========================================================================
# get_resource_paths / get_extension_paths
# ===========================================================================

class TestGetResourcePaths:
    def test_returns_extension_paths_for_enabled_packages(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path)
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        paths = pm.get_resource_paths("extensions")
        assert str(tmp_path / "packages" / "git" / "my_ext") in paths

    def test_disabled_package_excluded(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path, enabled=False)
        pkg_data["enabled"] = False
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        paths = pm.get_resource_paths("extensions")
        assert paths == []

    def test_get_extension_paths_alias(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path)
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        assert pm.get_extension_paths() == pm.get_resource_paths("extensions")

    def test_skills_paths(self, tmp_path):
        install_path = tmp_path / "packages" / "git" / "my_ext"
        skills_dir = install_path / "skills"
        skills_dir.mkdir(parents=True)
        pkg_data = {
            "name": "my_ext",
            "source": "git:https://example.com/repo",
            "install_path": str(install_path),
            "installed_at": "2026-01-01T00:00:00+00:00",
            "version": "abc123",
            "enabled": True,
            "resources": {
                "extensions": [],
                "skills": [str(skills_dir)],
                "prompts": [],
                "themes": [],
            },
        }
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        paths = pm.get_resource_paths("skills")
        assert str(skills_dir) in paths

    def test_skips_missing_directories(self, tmp_path):
        pkg_data = _fake_installed_pkg(tmp_path)
        # Remove the install dir to simulate a broken install
        import shutil
        shutil.rmtree(str(tmp_path / "packages" / "git" / "my_ext"))
        _seed_manifest(tmp_path, {"my_ext": pkg_data})
        pm = _pm(tmp_path)
        assert pm.get_resource_paths("extensions") == []

    def test_empty_manifest_returns_empty(self, tmp_path):
        pm = _pm(tmp_path)
        assert pm.get_extension_paths() == []


# ===========================================================================
# Integration: ExtensionRegistry picks up packages
# ===========================================================================

class TestExtensionRegistryIntegration:
    def test_registry_discovers_package_extension(self, tmp_path):
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

        _seed_manifest(tmp_path, {
            "test_pkg": {
                "name": "test_pkg",
                "source": "git:https://example.com/test_pkg",
                "install_path": str(pkg_dir),
                "installed_at": "2026-01-01T00:00:00+00:00",
                "version": "abc123",
                "enabled": True,
                "resources": {
                    "extensions": [str(pkg_dir)],
                    "skills": [],
                    "prompts": [],
                    "themes": [],
                },
            }
        })

        pm = _pm(tmp_path)

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

