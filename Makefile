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
	poetry add --dev black coverage flake8 pytest
	# pip install black coverage flake8 pytest

format:
	poetry run python -m black tftokenizers tests
	# python -m black tftokenizers tests

lint:
	poetry run python -m flake8 tftokenizers tests
	python -m flake8 tftokenizers tests

test:
	poetry run python -m pytest -ra
	# python -m pytest -ra

tox:
	poetry run tox

build:
	make deps
	make testdeps
	make format
	make tox
	# make coverage
