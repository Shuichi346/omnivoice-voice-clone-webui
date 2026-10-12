# AGENTS.md

Project instructions for coding agents working in `Shuichi346/qwen-voice-clone-webui`.

## Documentation maintenance

- After completed implementation work, update `CHANGELOG.md`, `NOTES.md`, and `AGENTS.md` in English using the current task's evidence, including already committed or pushed work.
- Record what changed in the changelog in past tense under `Fixed`, `Added`, `Changed`, and `Removed`. Omit empty categories and routine cleanup or test-only changes unless their impact matters.
- Use `## YYYY-MM-DD` for new work, newest first, and append to existing same-day categories. Use the completion date in the user's timezone; do not guess dates or release versions. Preserve existing dated/versioned history and classify undated historical entries under `## DATE UNKNOWN` with `### Fixed`, `### Added`, `### Changed`, and `### Removed`. Do not create new `Unreleased` sections.
- Keep causes, solution rationale, behavior to preserve, known useful paths/tests, and unresolved limitations in `NOTES.md`, following its existing conventions. Keep durable, verified repository instructions in `AGENTS.md`; avoid duplicate summaries, session logs, and one-time rules.
- Read only relevant sections needed to avoid duplicates or contradictions. Reuse task evidence; inspect the smallest relevant diff or source excerpt only for missing material facts, without routinely rebuilding history.
- For a full update, consider all three root files and create missing files with minimal headings. Leave existing files unchanged when there is nothing material to add.
- Review the documentation diff and unresolved references, and run required documentation checks. Documentation-only changes do not require rerunning implementation tests solely to record the work.
