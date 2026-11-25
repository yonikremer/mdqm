"""One module per bootstrap stage (DESIGN.md section 10): D1..D10 are deterministic, A1..A5 agentic.

A stage owns its input model and its report, reads only what its caller injects, and writes nothing
outside the metadata store. A stage that changes its mind about a fact writes a *candidate* or a
*proposal*, never a fact: the human gate is the only thing that approves.
"""
