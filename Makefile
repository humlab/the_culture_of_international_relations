SOURCE_FOLDERS = common notebooks
# 1_quantitative_analysis 2_network_analysis 3_text_analysis

lint: tidy pylint

tidy: black isort

pylint:
	@uv run pylint --extension-pkg-allow-list=community $(SOURCE_FOLDERS)

isort:
	@uv run isort --profile black --float-to-top --line-length 120 --py 311 $(SOURCE_FOLDERS)

black:
	@uv run black --version
	@uv run black $(SOURCE_FOLDERS)

clean-notebooks:
	@find notebooks -name "*.ipynb" -exec uv run nbstripout {} \;


# concise, full, json, json-lines, junit, grouped, github, gitlab, pylint, rdjson, azure, sarif
ruff:
	@uv run ruff --version
	@uv run ruff check --fix --output-format concise  $(SOURCE_FOLDERS)

dead-code:
	@uv run vulture $(SOURCE_FOLDERS)
dead-code2:
	@uv run ruff --version
	@uv run ruff check --select D --output-format grouped $(SOURCE_FOLDERS)
