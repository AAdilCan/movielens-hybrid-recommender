"""Integration tests for the click CLI.

The data loaders are monkeypatched to return the in-memory toy tables so the CLI
exercises the real train/fit/recommend path without any network download.
"""

from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from recsys import cli as cli_module


def _patch_loaders(monkeypatch, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    monkeypatch.setattr(cli_module, "load_ratings", lambda dataset: ratings)
    monkeypatch.setattr(cli_module, "load_movies", lambda dataset: movies)


def test_recommend_command_outputs_titles(monkeypatch, toy_ratings, toy_movies):
    _patch_loaders(monkeypatch, toy_ratings, toy_movies)
    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli, ["recommend", "--user", "1", "--top-n", "3"]
    )
    assert result.exit_code == 0, result.output
    assert "recommendations for user 1" in result.output
    # At least one known title should be printed.
    assert "(" in result.output  # titles carry a "(year)" suffix


def test_recommend_unknown_user_falls_back(monkeypatch, toy_ratings, toy_movies):
    _patch_loaders(monkeypatch, toy_ratings, toy_movies)
    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli, ["recommend", "--user", "424242", "--top-n", "3"]
    )
    # Cold-start user still gets popularity-backed recommendations, not a crash.
    assert result.exit_code == 0, result.output
    assert "recommendations for user 424242" in result.output


def test_cli_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "recommend" in result.output
    assert "evaluate" in result.output
