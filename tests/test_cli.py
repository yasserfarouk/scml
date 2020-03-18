from click.testing import CliRunner

from scml.scml import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])

    assert len(result.output) >= 0
    assert result.exit_code == 0
