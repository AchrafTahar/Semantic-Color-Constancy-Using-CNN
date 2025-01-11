#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = semantic_color_constancy_using_cnn
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 semantic_color_constancy_using_cnn
	isort --check --diff --profile black semantic_color_constancy_using_cnn
	black --check --config pyproject.toml semantic_color_constancy_using_cnn

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml semantic_color_constancy_using_cnn

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Run the model script
.PHONY: run_model
run_model:
	$(PYTHON_INTERPRETER) semantic_color_constancy_using_cnn/modeling/model.py

## Train the model
.PHONY: train
train:
	$(PYTHON_INTERPRETER) semantic_color_constancy_using_cnn/modeling/train.py

## Run tests
.PHONY: test
test:
	pytest tests/

## Serve documentation locally
.PHONY: docs
docs:
	mkdocs serve

## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) semantic_color_constancy_using_cnn/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
