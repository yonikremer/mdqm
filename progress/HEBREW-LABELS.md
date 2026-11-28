status: done
task: the Hebrew labelling question (docs/HANDOFF.md section 6)
agent: pi (main session)   branch: task/hebrew-labels
tests run: `python -m pytest tests/unit/test_context_index.py tests/unit/test_context_pack.py tests/unit/test_context_endpoint.py -q` (117 passed); the whole `tests/unit` suite and `scripts/check.py` are recorded below.

## Summary
`/context` answered an English question with a pack of measures, joins and value maps, and answered
the same question in Hebrew with `""`. The endpoint was fine; the **layer had nowhere to hold a
Hebrew name**. `value_maps` was the only section with synonyms in both languages, so a Hebrew prompt
matched only if it contained a Hebrew *value* ("פעיל"), never a measure ("מסגרת אשראי").

The fix is one optional section and one merge:
* `SemanticLayer.labels` — `tuple[str, ...]` per key, keyed `table:<source.table>`,
  `measure:<name>` or `column:<qualified>`. Nothing Hebrew specific: it is "the deployment's own
  words for a thing", which also covers an internal nickname or an abbreviation.
* `ContextIndex` reads `layer.labels` itself and merges any caller labels on top (caller wins), so
  no caller has to remember to pass them — `ContextService`, `find_data` and the endpoint all get
  the behaviour without a signature change.
* `_checked_labels` already refused a key that names nothing; it now refuses one declared in the
  layer too, at build time, because a silently ignored label is a search that mysteriously misses.
* Labels are part of `version_hash` (they are semantic facts), so changing a word invalidates the
  index instead of lagging it. Pinned by a test.

## Verified by running it
`examples/semantic_example.yaml` now carries Hebrew labels. Loading exactly that file and building
the pack:

| prompt | before | after |
|---|---|---|
| `מסגרת אשראי ממוצעת לפי סטטוס` | 0 chars | 522 chars, 5 items |
| `average credit limit by customer status` | 1106 chars | 1106 chars (unchanged) |
| `leaderboard` (names nothing) | 0 chars | 0 chars (still no noise) |

`tests/unit/test_context_pack.py::test_the_examples_hebrew_question_gets_a_pack` locks all three.

## Decisions
* **Labels live in the semantic file, not in a config file of their own.** They are facts about the
  data, they belong to the same review/approval flow as everything else in the layer, and the
  `generated/` + `curated/` merge already gives precedence (a curated label overrides a generated
  one). A separate file would have been a second place to look.
* **A caller can still override per key.** One deployment sharing a base file needs to say its own
  word for the same measure without editing the shared file.
* **The section is a list of words per key, not a `{lang: word}` map.** Nothing in the code needs to
  know the language of a word, and a `lang` key would have to be right; a flat list also lets one
  key carry a Hebrew name, an English name and the internal abbreviation at once.

## Deviations
none

## Dependencies
none

## Not verified
* The running server: Docker is down (see `docs/HANDOFF.md` §1), so `/context` was exercised
  in-process through `ContextPackSource.build`, not over HTTP. The endpoint calls the same object.
* `tests/integration --require-services`: same blocker.

## Plan issues
* A1+A2 proposes names but has no field to propose a *label* into; with this section it should. That
  is a proposal-flow change (targets, validators, the review queue), not a schema change, and is not
  done here.
* Nothing produces Hebrew labels automatically: they are hand-written until A1 grows that target.
