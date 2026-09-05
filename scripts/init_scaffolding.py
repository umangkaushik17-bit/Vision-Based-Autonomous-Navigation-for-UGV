import os
from pathlib import Path

DIRECTORIES = [
    "src/ugv_core",
    "src/ugv_perception/detection",
    "src/ugv_perception/segmentation",
    "src/ugv_perception/preprocessing",
    "src/ugv_traversability",
    "src/ugv_localization",
    "src/ugv_costmap",
    "src/ugv_navigation",
    "src/ugv_safety",
    "src/ugv_mission",
    "src/ugv_visualization",
    "ai/datasets",
    "ai/models",
    "ai/training",
    "ai/export",
    "ai/evaluation",
    "simulation/worlds",
    "simulation/models",
    "simulation/scenarios",
    "matlab/calibration",
    "matlab/localization",
    "matlab/trajectory",
    "matlab/planning",
    "configs",
    "launch",
    "experiments",
    "benchmarks",
    "scripts",
    "docs",
    "tests",
]

PYTHON_PACKAGES = [
    "src/ugv_core",
    "src/ugv_perception",
    "src/ugv_perception/detection",
    "src/ugv_perception/segmentation",
    "src/ugv_perception/preprocessing",
    "src/ugv_traversability",
    "src/ugv_localization",
    "src/ugv_costmap",
    "src/ugv_navigation",
    "src/ugv_safety",
    "src/ugv_mission",
    "src/ugv_visualization",
    "ai",
    "ai/models",
    "ai/datasets",
    "ai/training",
    "ai/export",
    "ai/evaluation",
    "tests",
]

def main():
    root = Path(__file__).resolve().parent.parent
    print(f"Scaffolding repository at {root}...")
    
    for dir_path in DIRECTORIES:
        full_path = root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        # Keep empty directories tracked
        gitkeep = full_path / ".gitkeep"
        if not gitkeep.exists() and not any(full_path.iterdir()):
            gitkeep.touch()
            
    for pkg in PYTHON_PACKAGES:
        pkg_init = root / pkg / "__init__.py"
        if not pkg_init.exists():
            pkg_name = pkg.replace("/", ".")
            with open(pkg_init, "w", encoding="utf-8") as f:
                f.write(f'"""Package initialization for {pkg_name}."""\n')

    print("Scaffolding created successfully!")

if __name__ == "__main__":
    main()
