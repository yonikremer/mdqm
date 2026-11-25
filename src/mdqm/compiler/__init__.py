"""Spec to SQL: the compiler lane (tasks C09, C10).

:func:`compile_spec` turns one validated :class:`~mdqm.spec.models.QuerySpec` into one
:class:`~mdqm.models.Fragment` for one source, or into the :class:`~mdqm.models.ErrorResponse` that
says what to change. Every statement is built through the sqlglot AST, has its timestamps frozen to
``exec_ts`` and is inspected by the read-only guard before it leaves this package.
"""

from __future__ import annotations

from mdqm.compiler.core import (
    DEFAULT_MAX_ROWS,
    CompileContext,
    CompiledQuery,
    compile_spec,
)

__all__ = [
    "DEFAULT_MAX_ROWS",
    "CompileContext",
    "CompiledQuery",
    "compile_spec",
]
