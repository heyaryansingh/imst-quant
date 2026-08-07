"""Every module in the package must be importable.

Modules that are not reached by any other test can rot silently - a stale
dependency, a mojibake byte, a name used before it is imported. None of that
shows up until someone actually imports the module, which is usually in
production. Importing all of them here is cheap and catches the whole class.
"""

import importlib
import pkgutil

import pytest

import imst_quant

MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(imst_quant.__path__, "imst_quant.")
)


def test_module_list_is_not_empty():
    """Guards against the walk silently finding nothing and vacuously passing."""
    assert len(MODULES) > 50, f"only discovered {len(MODULES)} modules"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
