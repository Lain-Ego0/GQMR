from __future__ import annotations

import zipfile
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from gqmr.project import (
    add_resource,
    load_project,
    materialize_resource,
    new_project,
    pack_project,
    save_project,
)
from gqmr.project.model import EditCommand, ProjectError
from gqmr.retarget import LocalRepairConfig, LocalRepairDiagnostics


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
    materialized = materialize_resource(
        packed_path, packed, packed.active_animal_motion, cache_dir=tmp_path / "cache"
    )
    assert materialized.read_bytes() == resource.read_bytes()
    imported_again = add_resource(packed, materialized, make_active="animal")
    assert len(imported_again.resources) == 1
    assert imported_again.active_animal_motion == packed.active_animal_motion

    materialized.write_bytes(b"x" * packed_resource.size)
    assert materialized.stat().st_size == packed_resource.size
    restored = materialize_resource(
        packed_path, packed, packed.active_animal_motion, cache_dir=tmp_path / "cache"
    )
    assert restored.read_bytes() == resource.read_bytes()

    save_project(packed_path, packed)
    saved_portable = load_project(packed_path)
    saved_resource = saved_portable.resources[saved_portable.active_animal_motion]
    assert saved_resource.embedded
    assert materialize_resource(
        packed_path,
        saved_portable,
        saved_portable.active_animal_motion,
        cache_dir=tmp_path / "saved-cache",
    ).read_bytes() == resource.read_bytes()

    repacked_path = tmp_path / "repacked.gqmr"
    pack_project(repacked_path, saved_portable, source_path=packed_path)
    repacked = load_project(repacked_path)
    assert materialize_resource(
        repacked_path,
        repacked,
        repacked.active_animal_motion,
        cache_dir=tmp_path / "repacked-cache",
    ).read_bytes() == resource.read_bytes()


def test_add_resource_deduplicates_an_unchanged_file(tmp_path: Path) -> None:
    resource = tmp_path / "animal.npz"
    resource.write_bytes(b"canonical-motion-fixture")
    first = add_resource(new_project(), resource, make_active="animal")
    second = add_resource(first, resource, make_active="animal")

    assert len(second.resources) == 1
    assert second.active_animal_motion == first.active_animal_motion


def test_project_loader_rejects_path_traversal(tmp_path: Path) -> None:
    destination = tmp_path / "unsafe.gqmr"
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("project.json", "{}")
        archive.writestr("edits.json", "[]")
        archive.writestr("../escape", "bad")

    with pytest.raises(ProjectError, match="unsafe project member"):
        load_project(destination)


def test_project_loader_rejects_corrupted_embedded_resource(tmp_path: Path) -> None:
    resource = tmp_path / "input.bin"
    resource.write_bytes(b"original")
    project = add_resource(new_project(), resource, make_active="animal")
    packed_path = tmp_path / "portable.gqmr"
    pack_project(packed_path, project)
    with zipfile.ZipFile(packed_path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    embedded = next(name for name in members if name.startswith("embedded/"))
    members[embedded] = b"corrupt!"
    with zipfile.ZipFile(packed_path, "w") as destination:
        for name, payload in members.items():
            destination.writestr(name, payload)

    with pytest.raises(ProjectError, match="hash mismatch"):
        load_project(packed_path)


def test_project_model_rejects_malformed_local_repair_command() -> None:
    with pytest.raises(ValidationError, match="frame_range"):
        EditCommand(
            command_id=str(uuid.uuid4()),
            kind="local_repair",
            resource_id=str(uuid.uuid4()),
            parameters={},
            created_at="2026-08-12T00:00:00Z",
        )


def test_project_persists_local_repair_command(tmp_path: Path) -> None:
    resource = tmp_path / "robot.npz"
    resource.write_bytes(b"canonical-robot-motion-fixture")
    project = add_resource(new_project(), resource, make_active="robot")
    resource_id = project.retarget["active_robot_motion"]
    config = LocalRepairConfig(root_height_offset_m=0.01)
    diagnostics = LocalRepairDiagnostics(
        solver="project-test-solver",
        solver_version="1.0",
        frames_processed=2,
        converged=True,
    )
    digest = "0" * 64
    command = EditCommand(
        command_id=str(uuid.uuid4()),
        kind="local_repair",
        resource_id=resource_id,
        parameters={
            "frame_range": [1, 2],
            "requested_config": config.model_dump(mode="json"),
            "applied_config": config.model_dump(mode="json"),
            "diagnostics": diagnostics.model_dump(mode="json"),
            "input_motion_sha256": digest,
            "output_motion_sha256": digest,
        },
        created_at="2026-08-12T00:00:00Z",
    )
    project = project.model_copy(update={"edits": (command,)})
    destination = tmp_path / "local-repair.gqmr"

    save_project(destination, project)
    loaded = load_project(destination)

    assert loaded.edits == (command,)
    assert loaded.edits[0].parameters["requested_config"] == config.model_dump(
        mode="json"
    )
