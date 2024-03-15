from pytest import mark
from click.testing import CliRunner

from scml.cli import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])

    assert len(result.output) >= 0
    assert result.exit_code == 0


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    output = result.output.split("\n")
    assert len(output) > 0
    assert output[-2].strip().startswith("version"), result.output


def test_versionm():
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    output = result.output.split("\n")
    assert len(output) > 0
    assert output[0].strip().startswith("SCML")
    assert "NegMAS" in output[0].strip()


@mark.parametrize("year", list(range(2019, 2025)))
def test_run_std(year):
    runner = CliRunner()
    result = runner.invoke(main, [f"run{year}", "--steps", "5"])
    assert result.exit_code == 0, result.output
    for x in ("Running Time",):
        assert x in result.output
    for x in ("Winners:", "Welfare:", "Business size"):
        assert year in (2019, 2020) or x in result.output


@mark.parametrize("year", list(range(2021, 2025)))
def test_run_oneshot(year):
    runner = CliRunner()
    result = runner.invoke(main, [f"run{year}", "--oneshot", "--steps", "5"])
    assert result.exit_code == 0, result.output
    for x in ("Winners:", "Welfare:", "Running Time", "Business size"):
        assert x in result.output


@mark.parametrize("year", list(range(2021, 2025)))
def test_run_tournaments_oneshot(year):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            f"tournament{year}",
            "--no-ufun-logs",
            "--no-neg-logs",
            "--ttype",
            "oneshot",
            "--steps",
            "5",
            "--compact",
        ],
    )
    assert result.exit_code == 0, result.output
    for x in ("Tournament will be run",):
        assert x in result.output
