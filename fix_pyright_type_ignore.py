import re
import os


def parse_pyright_file(pyright_path):
    file_issues = {}
    current_file = None
    with open(pyright_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            # Match file path lines
            if line.startswith("/") and line.endswith(".py"):
                current_file = line.strip()
                if current_file not in file_issues:
                    file_issues[current_file] = []
            # Match error lines with line:col
            m = re.match(r"\s*(/.*\.py):(\d+):(\d+)\s+-\s+error:", line)
            if m:
                file_path, lineno, col = m.group(1), int(m.group(2)), int(m.group(3))
                file_issues.setdefault(file_path, []).append(int(lineno))
            # Match error lines without line:col (just message)
            elif current_file and line.strip().startswith('"'):
                # If no line number, mark as line 1
                file_issues[current_file].append(1)
    return file_issues


def append_type_ignore_to_lines(file_path, lines_to_ignore):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    with open(file_path, "r") as f:
        lines = f.readlines()
    # Remove duplicates and sort
    lines_to_ignore = sorted(set(lines_to_ignore))
    for lineno in lines_to_ignore:
        idx = lineno - 1
        if 0 <= idx < len(lines):
            line = lines[idx]
            if "# type: ignore" not in line:
                # Remove trailing newline, add type: ignore, then add newline
                lines[idx] = line.rstrip("\n") + "  # type: ignore\n"
    with open(file_path, "w") as f:
        f.writelines(lines)


def main():
    pyright_path = "pyright"
    file_issues = parse_pyright_file(pyright_path)
    for file_path, lines in file_issues.items():
        append_type_ignore_to_lines(file_path, lines)
        print(f"Patched {file_path} with # type: ignore on lines: {lines}")


if __name__ == "__main__":
    main()
