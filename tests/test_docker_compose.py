from pathlib import Path

COMPOSE = (Path(__file__).parents[1] / "docker-compose.yml").read_text()


def test_db_healthcheck_probes_tcp():
    # On a fresh volume the postgres image first runs a socket-only server while it
    # initializes; probing TCP keeps `db` unhealthy until migrate can connect.
    assert "pg_isready -h 127.0.0.1" in COMPOSE
