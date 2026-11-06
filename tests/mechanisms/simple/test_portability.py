import pytest
from apc_anonymizer.mechanisms.simple import database
from apc_anonymizer.mechanisms.simple import hyperparameter_optimization as hpo


@pytest.mark.parametrize("url", ["postgresql://host/db", "sqlite:///local.db"])
def test_external_database_url(monkeypatch, mocker, url):
    monkeypatch.setenv("APC_ANONYMIZER_DB_URL", url)
    run = mocker.patch.object(database.subprocess, "run")
    database.prepare_database("unused")
    database.close_database()
    run.assert_not_called()
    assert hpo.get_database_url("unused") == url


def test_default_database_url(monkeypatch):
    monkeypatch.delenv("APC_ANONYMIZER_DB_URL", raising=False)
    url = hpo.get_database_url("hyperparameter_opt")
    assert url.drivername == "postgresql+psycopg2"
    assert url.username == "postgres"
    assert url.database == "hyperparameter_opt"


@pytest.mark.parametrize("cpu_count,expected", [(4, 4), (None, 1)])
def test_process_count_without_affinity(monkeypatch, cpu_count, expected):
    monkeypatch.delattr(hpo.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(hpo.os, "cpu_count", lambda: cpu_count)
    config = {"options": {"numberOfProcesses": "one-per-core"}}
    assert hpo.get_parallel_process_count(config) == expected


@pytest.mark.parametrize("override", [True, False])
def test_output_directory_override(monkeypatch, mocker, tmp_path, override):
    output = str(tmp_path / "nested" / "output")
    monkeypatch.setenv("OUTPUT_DIR", output if override else "")
    mocker.patch.object(database, "prepare_database")
    mocker.patch.object(database, "close_database")
    run = mocker.patch.object(hpo, "run_inference_in_parallel")
    config = {
        "outputDirectory": "default",
        "inference": {},
        "vehicleModels": [{"outputFilenames": ["test.csv"]}],
    }
    hpo.run_inference_for_all_vehicle_models(config)
    expected = output if override else "default"
    assert run.call_args.kwargs["output_directory"] == expected
