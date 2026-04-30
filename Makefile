.PHONY: help install chunk chunk-config test clean

help:                  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:               ## Install dependencies via uv
	uv sync

chunk:                 ## Run with --input / --output flags (e.g. make chunk INPUT=data/in OUTPUT=data/out/chunks.jsonl)
	uv run python -m contextual_chunker --input $(INPUT) --output $(OUTPUT)

chunk-config:          ## Run with a YAML config (e.g. make chunk-config CONFIG=config/example.yaml)
	uv run python -m contextual_chunker --config $(CONFIG)

test:                  ## Run the test suite
	uv run pytest tests/ -v

clean:                 ## Remove build artifacts and the venv
	rm -rf .venv build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
