"""User-controlled exclusion rules for index_folder.

Rules come from two sources, unioned:

1. A `.semanticignore` file at the indexed root (durable, gitignore syntax).
2. An `extra_patterns` list supplied per call (ad-hoc, same syntax).

Plus one hard-coded protection: the active LanceDB directory is never indexed,
regardless of user rules. The check is implemented separately from pathspec so
a user negation (`!lancedb/`) cannot cancel it.
"""

from pathlib import Path

import pathspec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError


SEMANTICIGNORE_FILENAME = ".semanticignore"


class ExclusionRules:
    """Compiled exclusion rules for one index_folder call."""

    def __init__(
        self,
        patterns: list[str],
        spec: pathspec.PathSpec,
        source_file: Path | None,
        db_dir_resolved: Path | None,
    ) -> None:
        self._patterns = patterns
        self._spec = spec
        self._source_file = source_file
        self._db_dir_resolved = db_dir_resolved

    @classmethod
    def load(
        cls,
        folder: Path,
        extra_patterns: list[str] | None,
        db_dir: str,
    ) -> "ExclusionRules":
        """Compile rules from .semanticignore (if present) + extra_patterns.

        Order matters for gitignore-style negation: file patterns are applied
        first (in their declared order), then per-call extras, so a caller can
        re-include something the file excludes by passing `!pattern` in
        `extra_patterns`.

        Raises ValueError when any pattern fails to compile, with the offending
        line in the message so the MCP layer can surface a useful error.
        """
        file_patterns: list[str] = []
        source_file: Path | None = None
        ignore_file = folder / SEMANTICIGNORE_FILENAME
        if ignore_file.is_file():
            source_file = ignore_file
            for raw in ignore_file.read_text().splitlines():
                line = raw.strip()
                # Skip blank lines and comments, matching gitignore semantics.
                if not line or line.startswith("#"):
                    continue
                file_patterns.append(line)

        all_patterns: list[str] = file_patterns + list(extra_patterns or [])

        # Compile patterns one by one so we can name the offending line in the
        # error. PathSpec.from_lines does not surface which line failed.
        compiled = []
        for line in all_patterns:
            try:
                compiled.append(pathspec.patterns.GitWildMatchPattern(line))
            except GitWildMatchPatternError as e:
                raise ValueError(
                    f"invalid exclusion pattern {line!r}: {e}"
                ) from e
        spec = pathspec.PathSpec(compiled)

        db_dir_resolved: Path | None = None
        try:
            db_dir_resolved = Path(db_dir).resolve()
        except OSError:
            # A non-existent db_dir is fine — nothing to protect from yet.
            db_dir_resolved = Path(db_dir)

        return cls(
            patterns=all_patterns,
            spec=spec,
            source_file=source_file,
            db_dir_resolved=db_dir_resolved,
        )

    def is_excluded(self, abs_path: Path, folder: Path, is_dir: bool) -> bool:
        """True iff the path is matched by any rule.

        `is_dir` MUST be set correctly — pathspec's gitwildmatch needs a
        trailing `/` on directory paths for directory-only patterns
        (e.g. `node_modules/`) to match. Passing `is_dir=False` for a directory
        silently fails to match these patterns.
        """
        # Hard rule first: a path inside the active LanceDB directory is always
        # excluded, regardless of pathspec rules. Implemented separately so a
        # user `!lancedb/` line can't negate it (gitignore: later rule wins).
        if self._db_dir_resolved is not None:
            try:
                resolved = abs_path.resolve()
            except OSError:
                resolved = abs_path
            if resolved.is_relative_to(self._db_dir_resolved):
                return True

        try:
            rel = abs_path.relative_to(folder)
        except ValueError:
            return False
        rel_str = rel.as_posix()
        if is_dir and not rel_str.endswith("/"):
            rel_str += "/"
        return self._spec.match_file(rel_str)

    @property
    def patterns(self) -> list[str]:
        return list(self._patterns)

    @property
    def source_file(self) -> Path | None:
        return self._source_file
