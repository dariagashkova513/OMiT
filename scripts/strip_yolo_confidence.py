"""Remove the trailing confidence value from YOLO segmentation label lines."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def stripped_content(path: Path) -> tuple[str, int]:
    output = []
    changed = 0
    for line_number, original in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not original.strip():
            output.append("")
            continue
        fields = original.split()
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: contains a non-numeric value") from error
        coordinates, confidence = values[:-1], values[-1]
        if class_id < 0:
            raise ValueError(f"{path}:{line_number}: class ID must be non-negative")
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(
                f"{path}:{line_number}: removing the final value would not leave a valid polygon"
            )
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: polygon coordinate is outside [0, 1]")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{path}:{line_number}: trailing confidence is outside [0, 1]")
        output.append(" ".join(fields[:-1]))
        changed += 1
    return "\n".join(output) + "\n", changed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--no-backup", action="store_true", help="do not create .bak copies")
    parser.add_argument("--dry-run", action="store_true", help="validate without changing files")
    args = parser.parse_args(argv)

    files = sorted(args.labels_dir.resolve().glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files found in {args.labels_dir}")

    # Validate the whole dataset before modifying the first file.
    prepared = [(path, *stripped_content(path)) for path in files]
    total_lines = sum(changed for _, _, changed in prepared)
    if args.dry_run:
        print(f"Validated {total_lines} lines in {len(files)} files; no files changed.")
        return

    for path, content, _ in prepared:
        if not args.no_backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            stream.write(content)
            temporary_path = Path(stream.name)
        temporary_path.replace(path)
    backup_note = " Backups use the .txt.bak suffix." if not args.no_backup else ""
    print(f"Updated {total_lines} lines in {len(files)} files.{backup_note}")


if __name__ == "__main__":
    main()
