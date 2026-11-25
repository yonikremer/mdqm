"""Planning: decide what a request may read before anything runs (docs/DESIGN.md sections 6-7).

``time`` turns a spec time intent into predicates on the right timestamp column; ``partitions``
(C08) proves the scan is pruned. Both are pure: no I/O, no clock read, ``exec_ts`` is injected.
"""
