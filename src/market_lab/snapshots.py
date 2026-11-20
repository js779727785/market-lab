from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .provenance import file_sha256


class SnapshotRef(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path
    sha256: str

    @classmethod
    def capture(cls, path: Path) -> SnapshotRef:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return cls(path=resolved, sha256=file_sha256(resolved))

    def valid(self) -> bool:
        return self.path.is_file() and file_sha256(self.path) == self.sha256


class SnapshotBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_files: tuple[SnapshotRef, ...]
    model: SnapshotRef
    config: SnapshotRef
    code: SnapshotRef

    @classmethod
    def capture(
        cls,
        *,
        data_files: tuple[Path, ...],
        model_file: Path,
        config_file: Path,
        code_file: Path,
    ) -> SnapshotBundle:
        return cls(
            data_files=tuple(SnapshotRef.capture(path) for path in data_files),
            model=SnapshotRef.capture(model_file),
            config=SnapshotRef.capture(config_file),
            code=SnapshotRef.capture(code_file),
        )

    def references(self) -> tuple[SnapshotRef, ...]:
        return (*self.data_files, self.model, self.config, self.code)
