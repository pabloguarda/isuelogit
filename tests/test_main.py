"""Tests for main.py module"""

import pytest

from transportAI import main


def test_printHelloWorld(capsys):

    main.printHelloWorld()
    captured = capsys.readouterr()
    assert captured.out == 'Hello World\n'

