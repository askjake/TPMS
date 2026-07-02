"""
Unit tests to ensure no raw filesystem paths are exposed in rendered UI components.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tpms_dashboard import config


# Patterns that must NEVER appear in any UI-rendered string
FORBIDDEN_PATTERNS = [
    "/home/montjac",
    "/home/",
    "montjac",
    str(config.EXPORT_DIR),
    str(config.ARCHIVE_DIR),
]


def test_display_names_are_friendly():
    """Display names must not contain filesystem paths."""
    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in config.EXPORT_DISPLAY_NAME, (
            f"EXPORT_DISPLAY_NAME contains forbidden pattern: {pattern}"
        )
        assert pattern not in config.ARCHIVE_DISPLAY_NAME, (
            f"ARCHIVE_DISPLAY_NAME contains forbidden pattern: {pattern}"
        )


def test_app_subtitle_no_paths():
    """APP_SUBTITLE must not contain filesystem paths or 'Performance-optimized'."""
    assert "Performance-optimized" not in config.APP_SUBTITLE
    assert "/home" not in config.APP_SUBTITLE
    assert "montjac" not in config.APP_SUBTITLE


def test_views_no_paths():
    """View descriptions must not contain filesystem paths."""
    for view_name, meta in config.VIEWS.items():
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in meta["description"], (
                f"View '{view_name}' description contains forbidden pattern: {pattern}"
            )


def test_sidebar_module_no_path_rendering():
    """Verify sidebar module source code does not render raw paths."""
    sidebar_path = Path(__file__).parent.parent / "components" / "sidebar.py"
    source = sidebar_path.read_text()
    # Should use display names, not raw paths
    assert "EXPORT_DISPLAY_NAME" in source or "ARCHIVE_DISPLAY_NAME" in source
    # Should NOT contain hardcoded paths
    assert "/home/montjac" not in source
    assert "EXPORTS_DIR" not in source or "EXPORT_DISPLAY_NAME" in source


def test_config_uses_env_vars():
    """Config must source paths from environment variables."""
    config_path = Path(__file__).parent.parent / "config.py"
    source = config_path.read_text()
    assert "os.getenv" in source, "Config must use os.getenv for path configuration"


if __name__ == "__main__":
    test_display_names_are_friendly()
    test_app_subtitle_no_paths()
    test_views_no_paths()
    test_sidebar_module_no_path_rendering()
    test_config_uses_env_vars()
    print("\u2705 All path-safety tests passed!")
