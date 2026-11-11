import os
from pathlib import Path

def bundle_project(root_dir: str, output_file: str = "project_bundle.py"):
    """
    Собирает все .py файлы проекта в один файл.
    Работает корректно даже если скрипт запущен из подпапки.
    """
    
    # Определяем корень проекта как папку, где лежит этот скрипт
    script_dir = Path(__file__).parent.resolve()
    
    IGNORE_DIRS = {
        ".git", "__pycache__", ".venv", "venv", "env", "envs", ".idea", ".vscode",
        "build", "dist", ".pytest_cache", "htmlcov"
    }
    
    CORE_ORDER = [
        "dtypes.py",
        "metrics.py", 
        "splitter.py",
        "tree.py",
        "session.py",
        "csv_loader.py"
    ]
    
    bundle_content = []
    bundle_content.append(f"# PROJECT BUNDLE: {script_dir.name}")
    bundle_content.append("# Generated for AI context.")
    bundle_content.append("#" * 50 + "\n")

    # 1. Core (строгий порядок)
    core_path = script_dir / "src" / "core"
    if core_path.exists():
        for filename in CORE_ORDER:
            file_path = core_path / filename
            if file_path.exists():
                _append_file_to_bundle(file_path, bundle_content, script_dir, "CORE")

    # 2. Остальной core
    if core_path.exists():
        for file_path in sorted(core_path.glob("*.py")):
            if file_path.name not in CORE_ORDER:
                _append_file_to_bundle(file_path, bundle_content, script_dir, "CORE")

    # 3. UI
    ui_path = script_dir / "src" / "ui"
    if ui_path.exists():
        for file_path in sorted(ui_path.glob("*.py")):
            _append_file_to_bundle(file_path, bundle_content, script_dir, "UI")

    # 4. Main, Workers, Tests
    src_path = script_dir / "src"
    if src_path.exists():
        main_file = src_path / "main.py"
        if main_file.exists():
            _append_file_to_bundle(main_file, bundle_content, script_dir, "ENTRY POINT")
            
        workers_file = src_path / "workers.py"
        if workers_file.exists():
            _append_file_to_bundle(workers_file, bundle_content, script_dir, "WORKERS")

        tests_path = script_dir / "tests"
        if tests_path.exists():
            for file_path in sorted(tests_path.glob("test_*.py")):
                _append_file_to_bundle(file_path, bundle_content, script_dir, "TESTS")

    # Запись
    with open(script_dir / output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(bundle_content))
        
    print(f"[OK] Bundle created at: {script_dir / output_file}")

def _append_file_to_bundle(file_path: Path, bundle_list: list, root_dir: Path, section: str):
    """Читает файл и добавляет его в список с корректным относительным путем."""
    try:
        # Вычисляем путь относительно корня проекта, а не текущей папки терминала
        rel_path = file_path.relative_to(root_dir)
        
        bundle_list.append("\n" + "#" * 50)
        bundle_list.append(f"# FILE: {rel_path} | SECTION: {section}")
        bundle_list.append("#" * 50 + "\n")
        
        with open(file_path, "r", encoding="utf-8-sig") as f: 
            # encoding='utf-8-sig' на случай, если в Windows есть BOM
            content = f.read()
            bundle_list.append(content)
    except Exception as e:
        bundle_list.append(f"# ERROR READING FILE {file_path.name}: {e}")

if __name__ == "__main__":
    bundle_project(root_dir=".")