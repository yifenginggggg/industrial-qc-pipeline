from industrial_qc.config import ProjectPaths


def test_project_paths_are_relative_to_root(tmp_path):
    paths = ProjectPaths.from_root(tmp_path)

    assert paths.data_dir == tmp_path / "data"
    assert paths.outputs_dir == tmp_path / "outputs"
