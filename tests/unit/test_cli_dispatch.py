"""Tests that every registered CLI subcommand is wired to a handler.

`create_parser` and `COMMANDS` are two separate lists that have to stay in
sync. A subcommand registered in the parser but missing from `COMMANDS` falls
through to `parser.print_help()` and exits 1, which reads like a usage error
rather than the wiring bug it is.
"""

import argparse

from imst_quant.cli import COMMANDS, create_parser


def _registered_subcommands() -> set[str]:
    parser = create_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers) == 1, "expected exactly one subparser group"
    return set(subparsers[0].choices)


def test_every_subcommand_has_a_handler():
    """No registered subcommand silently falls through to the help text."""
    missing = _registered_subcommands() - set(COMMANDS)
    assert not missing, f"subcommands with no handler: {sorted(missing)}"


def test_no_dead_dispatch_keys():
    """COMMANDS has no keys that argparse can never produce."""
    dead = set(COMMANDS) - _registered_subcommands()
    assert not dead, f"dispatch keys that are not registered commands: {sorted(dead)}"


def test_every_handler_is_callable():
    """Each dispatch value is a real function, not a stub or a typo'd name."""
    for name, handler in COMMANDS.items():
        assert callable(handler), f"handler for {name!r} is not callable"


def test_subcommands_build_their_own_help():
    """Every subparser is well-formed enough to render help without raising."""
    parser = create_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ][0]
    for name, subparser in subparsers.choices.items():
        assert subparser.format_help(), f"{name} produced empty help"
