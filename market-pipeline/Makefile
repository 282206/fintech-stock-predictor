.PHONY: up down init-db run-pipeline test api logs clean

## Start all services
up:
	docker compose up -d --build
	@echo "Services starting... Dagster UI: http://localhost:3000 | API: http://localhost:8000/docs | MinIO: http://localhost:9001"

## Stop all services
down:
	docker compose down

## Stop and remove volumes (destructive — wipes all data)
clean:
	docker compose down -v

## Initialize schema inside running Postgres (also runs automatically on first start via initdb)
init-db:
	docker compose exec postgres psql -U market_user -d market_data \
		-f /docker-entrypoint-initdb.d/001_create_schema.sql \
		-f /docker-entrypoint-initdb.d/002_create_tables.sql \
		-f /docker-entrypoint-initdb.d/003_create_indexes.sql

## Trigger the full pipeline via Dagster CLI (runs inside the dagster-webserver container)
run-pipeline:
	docker compose exec dagster-webserver \
		dagster job execute -m orchestration.dagster_project.definitions -j market_pipeline_job

## Run unit tests locally (requires local pip install)
test:
	PYTHONPATH=src:. pytest tests/ -v

## Tail all container logs
logs:
	docker compose logs -f

## Open an interactive psql shell
psql:
	docker compose exec postgres psql -U market_user -d market_data

## Open MinIO web console in default browser
minio-ui:
	open http://localhost:9001

## Open Dagster UI
dagster-ui:
	open http://localhost:3000

## Open FastAPI Swagger docs
api-docs:
	open http://localhost:8000/docs
