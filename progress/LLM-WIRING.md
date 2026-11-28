status: partial
task: LLM-WIRING
agent: pi (llm-wiring session)   branch: task/llm-wiring   commit: 037f286 (code) + this note
tests run: `python -m pytest tests/unit -q` -> 4673 passed, 1 warning (pytest-socket warning from
  tests/unit/test_mock_api.py, pre-existing); `python -m ruff check src tests scripts` -> clean;
  `python -m mypy` -> "Success: no issues found in 165 source files";
  `python -m ruff format --check` on the touched files -> 28 files already formatted;
  `python scripts/check_sql_sinks.py src/mdqm` -> 3 findings, all in `src/mdqm/stats/` and byte
  identical on `main` (pre-existing, another lane); `python -m pyright` (temporary pyrightconfig.json
  with venvPath at the main checkout, deleted before the last commit) -> the touched files add 0
  errors; the 8 remaining come from `src/mdqm/stats/postgres.py` and are on `main` too (pyright is
  not part of `scripts/check.py`).

## Summary

`docs/HANDOFF.md` §4c is now wired, and the labels half of §6 is proposed instead of hand-written.

* **Settings -> client -> context.** `mdqm/llm/factory.py` is the one translation from
  `settings.model` to a `ModelClient`. `pipeline/cli.py:context_for` puts it on
  `PipelineContext.llm`; `bootstrap run` prints which model answers (or that none will) *before* the
  stage list, because it decides whether A1+A2 can propose anything.
* **Labels.** A1+A2 already asked the model for names, descriptions and synonyms per table/column/
  value. The accepted answer's words now also become one `labels` proposal per table, column and
  measure -- keyed `table:<source.table>`, `column:<qualified>`, `measure:<name>`, exactly as
  `SemanticLayer.labels` and the context index key them. That is the payload whose approval *loads*,
  which is what §6 lacked.
* **Measures are targets.** A measure has no table, so its labels are the only names a question can
  match them by. `ProposeNamesStage` takes `measures` from the layer, gives a measure its own
  namespace (`measure:<name>`), lists its declared `source.table.column` entries in the fact sheet,
  and lets it reference only those columns plus the tables they live in.
* **Garbage is refused, not indexed.** A new validator (`label_word_problem`) rejects a word that is
  code/markup (JSON, SQL, `{}`, `|`, control characters), carries no letter (a lone number or a
  symbol), or is over 64 characters -- with `RejectionCode.NOT_WORDS` and a sentence naming the
  field, the offending text and the character class. One key is capped at eight words and the number
  dropped into `applied_fixes`, so a truncated set is never silent.
* **The review queue learned one new shape.** `ProposalKind.LABEL`, and the single list-valued merge
  it needs: a labels entry is written as a list of words. Provenance is *not* put inside it (every
  element is a search term); the decision lives in the review row and the audit log. The digest
  gained the kind in Hebrew and English -- without that, `_KIND_LABELS[language][kind]` raised
  `KeyError`, which a test now pins.

## Decisions

1. **Absent endpoint = absent client, and that is a supported deployment.** `model.base_url` and
   `model.model` both empty -> `model_client` returns `None`, no line printed at that level, and
   A1+A2 reports "no model client configured: nothing proposed" exactly as today. Half configured ->
   `None` plus one line naming the missing field, because a base URL with no model id is a typo.
2. **The key is never read while building the client.** `ModelClient` resolves its secret reference
   lazily (`MDQM_SECRET_MODEL_API_KEY` or `secrets.dir`), so a deployment can configure the endpoint
   before the secret exists; the first call then fails with the reference name and a fix, not a
   process that will not start.
3. **No sentinel in `context_for`.** `llm=None` means "build from settings"; a caller who wants no
   client passes settings without an endpoint (or, for tests, passes a `FakeLLM`). One less concept
   and no `object()` sentinel to type around.
4. **Labels are derived from the answer that is already proposed**, not from a new question. That
   keeps the ask at two calls per target (the second one rotated, for agreement) and reuses every
   existing validator; the only new rule is the word-shape one.
5. **All-or-nothing per target, as before.** A refused field drops the whole target's payloads
   (name, synonyms, labels) and is counted and named, rather than queueing a partial answer.
6. **A model failure still fails the stage.** `ModelError` (unreachable, timeout, unparsable after
   retries) propagates to the runner, which records `FAILED` with the sanitised problem+fix and
   exits 1, so the next run resumes at A1+A2. Degrading to "ran with nothing proposed" would hide an
   outage behind a green run.
7. **`other` cross-language label sets are not a collision.** Collisions stay scoped to one namespace
   (one table and its columns, one value map, one measure), so two tables may both answer to
   "customers" -- that is the semantic layer's job, not an ambiguity.

## Deviations

* **A `name`/`synonym` proposal still cannot be approved.** It targets a `names` section that
  `SemanticLayer` does not declare, so `ReviewQueue.approve` writes the file, `load()` rejects the
  unknown key, and the approval is rolled back. That was true before this change (nothing wrote
  `curated/names.yaml` because nothing had a model); now that labels are approvable the asymmetry is
  visible, so two tests pin it by name (`TestTheStatedGap`). Fixing it needs a design decision about
  where a display name and a one-sentence description live in the layer -- a `names` section with
  `he`/`en`/`description` per key, plus its effect on `version_hash` and the context index. Out of
  scope here, and deliberately not papered over by writing words the loader would ignore.
* `docs/DESIGN.md` §10 gained one clause (A1+A2 also propose `labels`). No other design document was
  touched: `docs/HANDOFF.md` and `README.md` belong to the owner's session, so §4c still says "not
  started" -- it is now done for the wiring and the label target, unverified against a real endpoint.

## Dependencies

None added. The client, its cache, its probe and `httpx` were already there; the wiring needed no new
package.

## Not verified

* **The real on-prem GLM endpoint. Unverified, and unverifiable from here.** There is no endpoint in
  this environment (`AGENTS.md` §9): its context window, its `response_format`/`json_schema` support
  and its tool calling remain guesses. Everything is either configured (`model.*`) or probed
  (`mdqm.llm.probe`), and the one behaviour this change relies on is the documented chat-completions
  shape -- nothing about the model's *quality*. Every test drives `FakeLLM`; a deployment must run
  `mdqm bootstrap run` against the real server and read the rejections before trusting a label set.
* **No Docker here** (`docker ps` fails), so `tests/integration --require-services` was not run. The
  live-stack path (`bootstrap run` in the container, `review list`/`approve` against the volume) is
  therefore unverified end to end; everything it exercises is covered by unit tests over the same
  code, and `python -m mdqm.cli bootstrap run` was run by hand (no sources configured) and printed
  `model endpoint not configured ...: A1+A2 will propose nothing` followed by the stage report.
* **A measure target was exercised only against fakes**: no live layer with measures was available,
  so `build_stages` handing `layer.measures` to the stage is covered by construction, not by a run
  against the container's `generated/` + `curated/` volumes.

## Plan issues

* `docs/HANDOFF.md` §2 lists `pyright` as 0 errors/0 warnings and §8 repeats it, but `main` reports 17
  (8 in `src/mdqm/stats/postgres.py`, the rest in tests). The document is stale relative to the tree;
  whoever owns it should re-measure rather than trust the number.
* `scripts/check_sql_sinks.py` reports 3 findings on `main` too (a statement built from a variable in
  `stats/collect.py` and twice in `stats/postgres.py`). That lane is not mine to change; it is worth
  a note in the ledger because it makes the documented "0 findings" gate untrue.
* The `names` section gap (see Deviations) needs a backlog task: as long as `TableSemantics` is an
  `extra="forbid"` skeleton with no place for a display name, A1's first two fields have nowhere to go
  and the pipeline pays for model calls whose result cannot be approved.
