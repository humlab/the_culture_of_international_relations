import os

import yaml


def get_fox_stopwords() -> set[str]:
    """Reads file fox_stopwords.txt and returns a set of stopwords."""
    filepath: str = os.path.join(os.path.dirname(__file__), "fox_stopwords.txt")
    stopwords: set[str] = set()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            word: str = line.strip()
            if word:
                stopwords.add(word)
    return stopwords


# def store_world_regions_to_yaml(filepath: str) -> None:
#     import yaml

#     with open(filepath, "w", encoding="utf-8") as f:
#         yaml.dump(WORLD_REGIONS, f, default_flow_style=False)


def load_yaml_file(filepath: str) -> dict:
    with open(filepath, encoding="utf-8") as f:
        data: dict = yaml.load(f, Loader=yaml.FullLoader)
    return data


_world_regions: dict[int, list[str]] = {}


def get_world_regions() -> dict[int, list[str]]:
    global _world_regions  # pylint: disable=global-statement
    if _world_regions:
        return _world_regions
    filepath: str = os.path.join(os.path.dirname(__file__), "world_regions.yml")
    _world_regions = load_yaml_file(filepath)
    return _world_regions


def get_region_parties(*region_ids) -> list[str]:

    data: list[str] = []
    world_regions: dict[int, list[str]] = get_world_regions()
    for region_id in region_ids:
        assert region_id in [1, 2, 3]
        data += world_regions[region_id]

    return data
