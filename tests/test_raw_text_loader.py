import pytest
from minigpt.utils import RawTextLoader


@pytest.fixture
def mock_dataset_files(tmp_path):
    mock_dir = tmp_path / "mock_data"
    mock_dir.mkdir()
    files = {
        "file1.txt": "Content of file1",
        "file2.txt": "Content of file2"
    }
    for filename, content in files.items():
        with open(mock_dir / filename, "w", encoding="utf-8") as f:
            f.write(content)
    return mock_dir, files


def test_file_list_exists(mock_dataset_files):
    mock_dir, files = mock_dataset_files
    loader = RawTextLoader(dataset_path=str(mock_dir))
    assert sorted(loader.file_list) == sorted(files.keys())


def test_load_text(mock_dataset_files):
    mock_dir, files = mock_dataset_files
    loader = RawTextLoader(dataset_path=str(mock_dir))
    combined_text = loader.load_text()
    expected_text = f" {loader.special_token} ".join(files.values())
    print(combined_text)
    print(expected_text)
    assert combined_text == expected_text


def test_empty_folder(tmp_path):
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()
    loader = RawTextLoader(dataset_path=str(empty_dir))
    assert loader.file_list == []
    assert loader.load_text() == ""


def test_folder_not_exist():
    with pytest.raises(FileNotFoundError):
        RawTextLoader(dataset_path="/non/existent/path")