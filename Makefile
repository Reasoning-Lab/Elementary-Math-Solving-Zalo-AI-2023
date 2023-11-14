install:
	pip install -r requirements.txt
	pre-commit install

format_file:
	autopep8 --aggressive --aggressive --in-place ${file}

run_precommit:
	pre-commit run --all-files
