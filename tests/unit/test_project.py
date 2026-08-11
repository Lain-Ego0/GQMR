from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from gqmr.project import add_resource, load_project, new_project, pack_project, save_project
from gqmr.project.model import ProjectError


def test_project_save_load_backup_and_portable_pack(tmp_path: Path) -> None:
    resource = tmp_path / "animal.npz"
    resource.write_bytes(b"canonical-motion-fixture")
    project = add_resource(new_project(), resource, make_active="animal")
    destination = tmp_path / "demo.gqmr"

    save_project(destination, project)
    loaded = load_project(destination)
    assert loaded.project_id == project.project_id
    assert loaded.active_animal_motion in loaded.resources
    save_project(destination, loaded)
    assert destination.with_suffix(".gqmr.bak").is_file()

    packed_path = tmp_path / "portable.gqmr"
    pack_project(packed_path, project)
    packed = load_project(packed_path)
    packed_resource = packed.resources[packed.active_animal_motion]
    assert packed_resource.embedded
    assert packed_resource.uri.startswith("embedded/")
    with zipfile.ZipFile(packed_path) as archive:
        assert archive.read(packed_resource.uri) == resource.read_bytes()


def test_project_loader_rejects_path_traversal(tmp_path: Path) -> None:
    destination = tmp_path / "unsafe.gqmr"
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("project.json", "{}")
        archive.writestr("edits.json", "[]")
        archive.writestr("../escape", "bad")

    with pytest.raises(ProjectError, match="unsafe project member"):
        load_project(destination)
