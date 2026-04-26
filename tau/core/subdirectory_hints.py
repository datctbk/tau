import os
from pathlib import Path
from typing import Dict, List, Optional

class SubdirectoryHintEngine:
    """Provides semantic hints about the structure of a given workspace.
    
    This pre-computes a map of top-level directories to prevent the agent from
    aimlessly grepping massive codebases (monorepos) looking for logic.
    """
    
    COMMON_MARKERS = {
        "package.json": "Node/JS/TS project",
        "Cargo.toml": "Rust workspace",
        "requirements.txt": "Python project",
        "pyproject.toml": "Python project",
        "go.mod": "Go module",
        "pom.xml": "Java/Maven project",
        "build.gradle": "Java/Kotlin Gradle project",
        "CMakeLists.txt": "C/C++ project",
        "Makefile": "Make build system",
        "docker-compose.yml": "Docker deployment",
    }
    
    COMMON_DIRS = {
        "src": "Main source code",
        "tests": "Test suite",
        "docs": "Documentation",
        "scripts": "Utility/build scripts",
        "pkg": "Public packages",
        "cmd": "Main executables (Go)",
        "frontend": "Frontend UI",
        "backend": "Backend APIs",
        "app": "Application core",
        ".github": "GitHub Actions / CI",
    }
    
    IGNORE_DIRS = {
        ".git", ".svelte-kit", "node_modules", "target", "build", "dist", 
        ".venv", "venv", "__pycache__", ".pytest_cache", ".next", ".tau"
    }

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()

    def generate_hints(self) -> str:
        """Generates a text summary of the repository layout."""
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            return "Workspace not found."

        project_type = self._detect_project_type()
        dir_hints = self._analyze_directories()
        
        lines = []
        if project_type:
            lines.append(f"Project Type: {project_type}")
        
        if dir_hints:
            lines.append("Top-level directory map:")
            for d, desc in dir_hints.items():
                lines.append(f"  - /{d}/ : {desc}")
                
        if not lines:
            return "No recognizable project structure found."
            
        return "\n".join(lines)

    def _detect_project_type(self) -> str:
        found_markers = []
        for marker, desc in self.COMMON_MARKERS.items():
            if (self.workspace_root / marker).is_file():
                found_markers.append(desc)
                
        if not found_markers:
            return "Unknown"
            
        return " + ".join(sorted(set(found_markers)))

    def _analyze_directories(self) -> Dict[str, str]:
        hints = {}
        try:
            for item in sorted(self.workspace_root.iterdir()):
                if not item.is_dir() or item.name in self.IGNORE_DIRS:
                    continue
                
                # Check for tau specific hints file
                hint_file = item / ".tau-hint"
                if hint_file.is_file():
                    try:
                        content = hint_file.read_text(encoding="utf-8").strip().split('\\n')[0]
                        hints[item.name] = content
                        continue
                    except Exception:
                        pass
                
                # Fall back to heuristic
                if item.name in self.COMMON_DIRS:
                    hints[item.name] = self.COMMON_DIRS[item.name]
                else:
                    # Provide a light summary of children
                    children = [x.name for x in item.iterdir() if x.is_file() and not x.name.startswith(".")]
                    if children:
                        disp = children[:3]
                        sfx = "..." if len(children) > 3 else ""
                        hints[item.name] = f"contains {', '.join(disp)}{sfx}"
                    else:
                        hints[item.name] = "folder"
        except Exception:
            pass
            
        return hints
