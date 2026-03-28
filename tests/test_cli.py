"""Tests for CLI commands."""

from click.testing import CliRunner
from chatmind.cli import main


class TestCLI:

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ChatMind" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_index_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["index", "nonexistent.json"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_search_no_index(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["search", "test query"])
            assert result.exit_code == 1
            assert "No index found" in result.output

    def test_stats_no_index(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["stats"])
            assert result.exit_code == 1

    def test_rooms_no_index(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["rooms"])
            assert result.exit_code == 1

    def test_people_no_index(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["people"])
            assert result.exit_code == 1

    def test_search_invalid_date(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["search", "test", "--after", "not-a-date"])
            assert result.exit_code == 1
            assert "Invalid date" in result.output
