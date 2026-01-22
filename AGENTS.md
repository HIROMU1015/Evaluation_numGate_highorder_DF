# AGENTS.md

## Tests
- Run: `source .venv/bin/activate && pytest -q`
- If tests fail: fix the code and re-run `pytest -q` until it passes.

## Slow tests
- Fast subset: `pytest -q -m "not slow"`
- If you add heavy regression tests, mark them with `@pytest.mark.slow`.

## Notes
- Prefer adding small deterministic unit tests (no randomness unless seeded).
