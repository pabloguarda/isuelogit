"""Tests for main.py module"""

from examples import main


def test_printHelloWorld(capsys):

    main.printHelloWorld()
    captured = capsys.readouterr()
    assert captured.out == 'Hello World\n'

