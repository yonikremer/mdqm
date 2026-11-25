"""``mdqm.sql``: pure SQL logic (guard, ``exec_ts`` rewriting, fingerprints).

Nothing here does I/O: these modules take SQL text or an AST and return a decision, so they are
unit-testable with tables of cases. Import the modules, not names from this package:
``from mdqm.sql.guard import guard`` (re-exporting ``guard`` here would shadow the submodule).
"""

from __future__ import annotations
