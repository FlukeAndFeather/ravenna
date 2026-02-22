import ravenna
from cli.main import cli


def test_package_importable():
    assert ravenna is not None


def test_cli_entry_point():
    assert callable(cli)
