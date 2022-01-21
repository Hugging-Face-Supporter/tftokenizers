# .DEFAULT_GOAL := help
.PHONY: coverage deps testdeps lint test

coverage:  ## Run tests with coverage
	python -m coverage erase
	python -m coverage run -m pytest -ra
	python -m coverage report -m

deps:
	poetry install
	poetry shell
	# pip install --upgrade pip
	# pip install -r requirements.txt

testdeps:
	pip install black coverage flake8 pytest

format:
	python -m black tftokenizers tests

lint:  ## Lint
	python -m flake8 tftokenizers tests

test:  ## Run tests
	python -m pytest -ra

build:
	make deps
	make testdeps
	make format
	make coverage
