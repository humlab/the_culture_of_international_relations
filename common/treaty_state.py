import os
import re
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from common import config, utility
from common.configuration import ConfigValue

warnings.simplefilter(action="ignore", category=FutureWarning)


class QueryUtility:

    @staticmethod
    def period_group_years(period_group: dict[str, Any]):
        if period_group["type"] == "range":
            return period_group["periods"]
        period_years: list[list[int]] = [list(range(x[0], x[1] + 1)) for x in period_group["periods"]]
        return utility.flatten(period_years)

    @staticmethod
    def parties_mask(parties):
        return lambda df: (df.party1.isin(parties)) | (df.party2.isin(parties))

    @staticmethod
    def period_group_mask(pg):
        return lambda df: df.signed_year.isin(QueryUtility.period_group_years(pg))

    @staticmethod
    def years_mask(ys):
        return lambda df: (
            ((ys[0] <= df.signed_year) & (df.signed_year <= ys[1]))
            if isinstance(ys, tuple) and len(ys) == 2
            else df.signed_year.isin(ys)
        )

    @staticmethod
    def query_treaties(treaties, filter_masks) -> pd.DataFrame:
        #  period_group=None, treaty_filter='', recode_is_cultural=False, parties=None, year_limit=None):
        if not isinstance(filter_masks, list):
            filter_masks = [filter_masks]
        for mask in filter_masks:
            if mask is True:
                pass
            elif isinstance(mask, str):
                treaties = treaties.query(mask)
            elif callable(mask):
                treaties = treaties.loc[mask(treaties)]
            else:
                treaties = treaties.loc[mask]
        return treaties


def get_treaties_column_names() -> list[str]:
    return ConfigValue("data.treaty_index.columns").resolve()


def get_treaties_skip_column_names() -> list[str]:
    return ConfigValue("data.treaty_index.skip_columns").resolve()


def get_default_extra_parties() -> dict[str, Any]:
    return ConfigValue("data.default_extra_parties").resolve()


def get_party_correction_map() -> dict[str, str]:
    return ConfigValue("data.party_corrections").resolve()


def trim_period_group(period_group, year_limit):
    pg = dict(period_group)
    low, high = year_limit
    if pg["type"] == "range":
        pg["periods"] = [x for x in pg["periods"] if low <= x <= high]
    else:
        pg["periods"] = [(max(low, x), min(high, y)) for x, y in pg["periods"] if not (high < x and low > y)]
    return pg


class TreatyState:

    def __init__(
        self,
        data_folder: str = "./data",
        skip_columns: list[str] | None = None,
        period_groups=None,
        filename: str | None = None,
        is_cultural_yesno_column: str = "is_cultural_yesno_org",
    ) -> None:  # pylint: disable=W0102
        filename = filename or "Treaties_Master_List_Treaties.csv"
        self.data_folder: str = data_folder
        self.period_groups: list[dict[str, Any]] = period_groups or config.DEFAULT_PERIOD_GROUPS
        self.treaties_skip_columns: list[str] = (skip_columns or get_treaties_skip_column_names() or []) + [
            "sequence",
            "is_cultural_yesno",
        ]
        self.treaties_columns: list[str] = get_treaties_column_names()
        self.is_cultural_yesno_column: str = is_cultural_yesno_column
        # self.csv_data_files = ConfigValue("data.treaty_index.source").resolve()
        self.csv_files: list[tuple[str, str, str | None, str | None]] = [
            (filename, "treaties", "Treaties_Master_List.xlsx", "Treaties"),
            ("country_continent.csv", "country_continent", None, None),
            (
                "parties_curated_parties.csv",
                "parties",
                "parties_curated.xlsx",
                "parties",
            ),
            (
                "parties_curated_continent.csv",
                "continent",
                "parties_curated.xlsx",
                "continent",
            ),
            ("parties_curated_group.csv", "group", "parties_curated.xlsx", "group"),
        ]
        self.data: dict[str, pd.DataFrame] = self._read_data()
        self.tagged_headnotes: pd.DataFrame | None = None
        self.groups: pd.DataFrame = self.get_groups()
        self.continents: pd.DataFrame = self.get_continents()
        self.parties: pd.DataFrame = self.get_parties()

        self._treaties: pd.DataFrame | None = None
        self._stacked_treaties: pd.DataFrame | None = None
        self._get_countries_list: list[str] | None = None
        self._party_preset_options: list[tuple[str, list[str]]] | None = None
        self._unique_sources: list[str] = []

    def check_party(self) -> str:
        party1: list[str] = self.treaties.party1.unique().tolist()
        party2: list[str] = self.treaties.party2.unique().tolist()
        df_party: pd.DataFrame = pd.DataFrame(data={"party": list(set(party1 + party2))})
        df: pd.DataFrame = df_party.merge(right=self.parties, left_on="party", right_index=True, how="left")

        unknown_parties: Sequence[str] | str = df[df.group_no.isna()].party.tolist()
        if len(unknown_parties) > 0:
            unknown_parties = ", ".join(['"' + str(x) + '"' for x in unknown_parties])
            logger.warning(f"[{unknown_parties} UNKNOWN PARTIES]")
            return unknown_parties

        return ""

    @property
    def treaties(self) -> pd.DataFrame:
        if self._treaties is None:
            self._treaties = self._process_treaties()
            self._unique_sources = list(sorted(list(self._treaties.source.unique())))
        return self._treaties

    @property
    def unique_sources(self) -> list[str]:
        return self._unique_sources

    @property
    def stacked_treaties(self) -> pd.DataFrame:
        if self._stacked_treaties is None:
            self._stacked_treaties = self._get_stacked_treaties()
        return self._stacked_treaties

    @property
    def cultural_treaties(self) -> pd.DataFrame:
        return self.treaties[self.treaties.is_cultural]

    @property
    def cultural_treaties_of_interest(self) -> pd.DataFrame:
        return self.cultural_treaties[(self.cultural_treaties.signed_period != "OTHER")]

    def _read_data(self) -> dict[str, pd.DataFrame]:  # -> dict[Any, Any]:
        data: dict[str, pd.DataFrame] = {}
        # na_values: list[str] = ["#N/A", "N/A", "NULL", "NaN", "-NaN"]
        for filename, key, xls_filename, xls_sheet in self.csv_files:
            # logger.debug(f"Reading file: {filename}...")
            path: str = os.path.join(self.data_folder, filename)
            if not os.path.exists(path):
                assert xls_filename is not None
                xls_path: str = os.path.join(self.data_folder, xls_filename)
                assert os.path.exists(xls_path)
                df: pd.DataFrame | Any = pd.read_excel(xls_path, sheet_name=xls_sheet)
                assert isinstance(df, pd.DataFrame)
                df.to_csv(path, sep="\t", index=True)
            data[key] = pd.read_csv(path, sep="\t", low_memory=False, keep_default_na=False, na_values=None)
        return data

    def get_treaty_period_group_categories(self, period_group: dict, treaties: pd.DataFrame) -> pd.Series:
        """Returns treaties categorized according to given period group's divisions

        Parameters
        ----------
        treaties : DataFrame
            WTI Treaties. If none then entire self.treaties are categorized

        period_group: dict
            Period group that

        Returns
        -------
        pd.Series
            A series indexed by treaty_id, and values assigned category division label
            e.g. '1939 to 1945' if division is (1939, 1945)

        """
        periods: Sequence[tuple[int, int]] = period_group["periods"]
        column: str = period_group["column"]

        if column in treaties.columns:
            return treaties[column]

        year_map: dict[int, str] = {year: f"{d[0]} to {d[1]}" for d in periods for year in list(range(d[0], d[1] + 1))}

        series: pd.Series = treaties.signed_year.apply(lambda x: year_map.get(x, "OTHER"))

        return series

    def _process_treaties(self) -> pd.DataFrame:

        # def get_period(division, year):
        #    match = [ p for p in division if p[0] <= year <= p[1]]
        #    return '{} to {}'.format(match[0][0], match[0][1]) if len(match) > 0 else 'OTHER'

        treaties: pd.DataFrame = self.data["treaties"]

        if len(treaties.columns) != len(self.treaties_columns):
            logger.error(
                f"WTI Treaties columns length mismatch! Expected {len(self.treaties_columns)} but got {len(treaties.columns)}"
            )

            expected_cols: set[str] = set(self.treaties_columns)
            actual_cols: set[str] = set(t.lower() for t in treaties.columns)
            logger.error(f"Expected columns: {expected_cols}")
            logger.error(f"Actual columns: {actual_cols}")
            raise ValueError("WTI Treaties columns length mismatch!")

        treaties.columns = self.treaties_columns

        treaties["is_cultural_yesno"] = treaties[self.is_cultural_yesno_column]
        treaties["vol"] = treaties.vol.fillna(0).astype("int", errors="ignore")
        treaties["page"] = treaties.page.fillna(0).astype("int", errors="ignore")
        treaties["signed"] = pd.to_datetime(treaties.signed, errors="coerce")
        treaties["is_cultural_yesno"] = treaties.is_cultural_yesno.astype(str)
        treaties["signed_year"] = treaties.signed.apply(lambda x: x.year)

        for period_group in self.period_groups:
            column: str = period_group["column"]
            if column not in treaties.columns:
                treaties[column] = self.get_treaty_period_group_categories(period_group, treaties)
                # treaties[column] = treaties.signed.apply(lambda x: get_period(definition['periods'], x.year))

        treaties["force"] = pd.to_datetime(treaties.force, errors="coerce")
        treaties["sequence"] = treaties.sequence.astype("int", errors="ignore")
        # treaties['group1'] = treaties.group1.fillna(0).astype('int', errors='ignore')
        # treaties['group2'] = treaties.group2.fillna(0).astype('int', errors='ignore')
        treaties["is_cultural"] = treaties.is_cultural_yesno.apply(lambda x: x.lower() == "yes")
        treaties["headnote"] = treaties.headnote.fillna("").astype(str).str.upper()

        treaties["party1"] = treaties.party1.fillna("").astype(str).str.upper()
        treaties["party2"] = treaties.party2.fillna("").astype(str).str.upper()

        party_correction_map: dict[str, str] = get_party_correction_map()
        treaties["party1"] = treaties.party1.apply(lambda x: party_correction_map.get(x, x))
        treaties["party2"] = treaties.party2.apply(lambda x: party_correction_map.get(x, x))

        treaties.loc[(treaties.topic1 == "7CULT") | (treaties.topic2 == "7CULT"), "topic"] = "7CULT"

        # Drop columns not used
        skip_columns: list[str] = list(set(treaties.columns).intersection(set(self.treaties_skip_columns)))
        if skip_columns is not None and len(skip_columns) > 0:
            treaties.drop(skip_columns, axis=1, inplace=True)

        treaties = treaties.set_index("treaty_id")
        return treaties

    def _get_stacked_treaties(self) -> pd.DataFrame:
        """
        Returns a bi-directional (duplicated) and processed version of the treaties master list.
        Each treaty has two records where party1 and party2 are reversed:
            Record #1: party=party1, party_other=party2, reversed=False
            Record #2: party=party2, party_other=party1, reversed=True
        Fields are also added for the party's and party_other's country code (2 chars), continent and WTI group.
        The two rows are identical for all other fields.
        """
        df1: pd.DataFrame = self.treaties.rename(
            columns={
                "party1": "party",
                "party2": "party_other",
                "group1": "party_group_no",
                "group2": "party_other_group_no",
            }
        ).assign(reversed=False)

        df2: pd.DataFrame = self.treaties.rename(
            columns={
                "party2": "party",
                "party1": "party_other",
                "group2": "party_group_no",
                "group1": "party_other_group_no",
            }
        ).assign(reversed=True)

        treaties: pd.DataFrame = pd.concat([df1, df2], axis=0)

        # Add fields for party's name, country, continent and WTI group
        parties: pd.DataFrame = self.parties[
            ["party_name", "country_code", "continent_code", "group_name", "short_name"]
        ]

        parties.columns = [
            "party_name",
            "party_country",
            "party_continent",
            "party_group",
            "party_short_name",
        ]
        treaties = treaties.merge(parties, how="left", left_on="party", right_index=True)

        # Add fields for party_other's country, continent and WTI group
        parties.columns = [
            "party_other_name",
            "party_other_country",
            "party_other_continent",
            "party_other_group",
            "party_other_short_name",
        ]
        treaties = treaties.merge(parties, how="left", left_on="party_other", right_index=True)

        # set 7CULT as topic when it is secondary topic
        treaties.loc[treaties.topic2 == "7CULT", "topic"] = "7CULT"

        return treaties

    # def get_stacked_treaties_subset(self, parties, complement=False):
    #     treaties = self.stacked_treaties[(self.stacked_treaties.signed_period!='OTHER')]
    #     if complement is False:
    #         treaties = treaties.loc[(treaties.party.isin(parties))]
    #     else:
    #         treaties = treaties.loc[(treaties.reversed==False)&(~treaties.party.isin(parties))]
    #     return treaties

    def get_continents(self) -> pd.DataFrame:
        """Returns continent reference data"""
        mask: pd.Series = self.data["continent"]["country_code2"] != ""
        self.data["continent"] = self.data["continent"][mask]
        df: pd.DataFrame = self.data["continent"].set_index("country_code2")

        if "Unnamed: 0" in df.columns:
            df = df.drop(["Unnamed: 0"], axis=1)
        name_map: dict[str, str] = ConfigValue("continents").resolve()
        df["continent"] = df.continent_code.apply(lambda x: name_map.get(x, x))
        return df

    def get_groups(self) -> pd.DataFrame:
        """Returns WTI group codes"""

        df: pd.DataFrame = (
            self.data["group"]
            .drop(["Unnamed: 0"], axis=1)
            .rename(columns={"GroupNo": "group_no", "GroupName": "group_name"})
        )
        df["group_no"] = df.group_no.astype(np.int32)
        df["group_name"] = df.group_name.astype(str)

        df = df.set_index("group_no")

        return df

    def get_parties(self, extra_parties: dict[str, Any] | None = None) -> pd.DataFrame:  # pylint: disable=W0102
        extra_parties = extra_parties or get_default_extra_parties()
        parties: pd.DataFrame = (
            self.data["parties"]
            .drop(["Unnamed: 0"], axis=1)
            .dropna(subset=["PartyID"])
            .rename(
                columns={
                    "PartyID": "party",
                    "PartyName": "party_name",
                    "ShortName": "short_name",
                    "GroupNo": "group_no",
                    "reversename": "reverse_name",
                }
            )
            .dropna(subset=["party"])
            .set_index("party")
        )

        parties["group_no"] = parties.group_no.astype(np.int32)
        parties["party_name"] = parties.party_name.apply(lambda x: re.sub(r"\(.*\)", "", x))
        parties["short_name"] = parties.short_name.apply(lambda x: re.sub(r"\(.*\)", "", x))
        parties[["party_name", "short_name"]] = parties[["party_name", "short_name"]].apply(lambda x: x.str.strip())

        parties.loc[(parties.group_no == 8), ["country", "country_code", "country_code3"]] = ""

        parties = pd.merge(parties, self.groups, how="left", left_on="group_no", right_index=True)

        parties = pd.merge(
            parties,
            self.continents,
            how="left",
            left_on="country_code",
            right_index=True,
        )

        extra_keys: list[str] = list(extra_parties.keys())
        extract_values: list[dict] = list(extra_parties.values())
        df: pd.DataFrame = pd.DataFrame(
            extract_values, columns=extra_parties[extra_keys[0]].keys(), index=list(extra_parties.keys())
        )
        parties = pd.concat([parties, df], axis=0)

        return parties

    def get_countries_list(self, excludes: Sequence[str] | None = None) -> list[Any]:

        if self._get_countries_list is None:
            parties: pd.DataFrame = self.get_parties()
            parties: pd.DataFrame = parties.loc[~parties.group_no.isin([0, 8, 11])]
            self._get_countries_list = parties.index.to_list()

        if len(excludes or []) > 0:
            assert excludes is not None
            return [x for x in self._get_countries_list if x not in excludes]

        return self._get_countries_list

    def get_party(self, party: str) -> dict | None:
        try:
            d: dict = self.parties.loc[party].to_dict()
            d["party"] = party
            return d
        except:  #  pylint: disable=bare-except
            return None

    def get_headnotes(self) -> pd.Series:
        return self.treaties.headnote.fillna("").astype(str)

    def get_tagged_headnotes(self, tags: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.tagged_headnotes is None:
            filename: str = os.path.join(self.data_folder, "tagged_headnotes.csv")
            self.tagged_headnotes = pd.read_csv(filename, sep="\t").drop("Unnamed: 0", axis=1)
        if tags is None:
            return self.tagged_headnotes
        return self.tagged_headnotes.loc[(self.tagged_headnotes.pos.isin(tags))]

    # def get_treaty_subset(self, options: dict[str, Any], language: str) -> pd.DataFrame:
    #     lang_field: str = {
    #         "en": "english",
    #         "fr": "french",
    #         "de": "other",
    #         "it": "other",
    #     }[language]
    #     df: pd.DataFrame = self.treaties
    #     df = df.loc[df[lang_field] == language]
    #     if options.get("source") is not None:
    #         df = df.loc[df.source.isin(options.get("source"))]  # type: ignore
    #     if options.get("from_year") is not None:
    #         df = df.loc[df.signed >= datetime.date(options["from_year"], 1, 1)]
    #     if options.get("to_year") is not None:
    #         df = df.loc[df.signed < datetime.date(options["to_year"] + 1, 1, 1)]
    #     if options.get("parties") is not None:
    #         df = df.loc[df.party1.isin(options["parties"]) | df.party2.isin(options["parties"])]
    #     return df  # .set_index('treaty_id')

    def filter_by_is_cultural(self, df: pd.DataFrame, treaty_filter: str) -> pd.DataFrame:

        if treaty_filter == "is_cultural":
            return df.loc[df.is_cultural.astype(bool)]

        if treaty_filter == "is_7cult":
            return df.loc[(df.topic1 == "7CULT")]

        return df

    def get_topic_category(
        self,
        df: pd.DataFrame,
        topic_category: dict[str, Any] | None,
        topic_column: str = "topic1",
    ) -> pd.Series:
        if topic_column not in df.columns:
            raise ValueError(f"Column {topic_column} not found in DataFrame")
        if topic_category is not None:
            return df.apply(lambda x: topic_category.get(x[topic_column], "OTHER"), axis=1)
        return df[topic_column]

    def get_treaties_within_division(
        self,
        treaties: pd.DataFrame | None = None,
        period_group: dict[str, Any] | None = None,
        treaty_filter: str = "",
        recode_is_cultural: bool = False,
        parties: list[str] | None = None,
        year_limit: Sequence[int] | None = None,
        treaty_sources: list[str] | None = None,
    ) -> pd.DataFrame:
        """Base filter function. Returns treaties filtered by given set of parameters.

         Parameters
         ----------
         treaties : DataFrame
             Optional. Treaty source to use instead of main WTI treaty index

         period_group: [dict]
             Period grouping to use for filtering. Treaties with signed year outside of group are filtered out.

         treaty_filter: str in [ 'is_cultural',  'is_7cult', '' ]
             Optional. 'is_cultural' filters out treaties where is_cultural is False, 'is_7cult' filters out treaties where topic1 is not '7cult'

         recode_is_cultural: bool
             Optional. Sets topic1 to '7CORR' for treaties having 'is_cultural' equal to true

         parties: List[str]
             Optional. Filters out treaties where neither of party1, party2 is in 'parties' list

         year_limit: Union[List[int], Tuple[int]] of ints
             Optional. Filters out treaties where signed_year is outside of given limit

         treaty_sources: Optional[List[str]]
             Optional. Filters out treaties where SOURCE not in treaty_sources

        Returns
         -------
         DataFrame

             Remaining treaties after filters are applied.

        """
        if treaties is None:
            treaties = self.treaties

        if treaties is None:
            raise ValueError("get_treaties_within_division: treaties is None")

        if period_group is None:
            raise ValueError("get_treaties_within_division: period_group is None")

        treaties2: pd.DataFrame = treaties  # We do this only to make pylance happy

        period_column: str = period_group["column"]

        if period_column not in treaties2.columns:
            raise ValueError(f"get_treaties_within_division: got unknown {period_column!r} as column")

        if period_group is not None:
            treaties2 = QueryUtility.query_treaties(treaties2, QueryUtility.period_group_mask(period_group))
        # if period_column != 'signed_year':
        #    df = base[base[period_column] != 'OTHER']
        # else:
        #    df = base[base.signed_year.isin(period_group['periods'])]

        if year_limit is not None:
            # base = QueryUtility.query_treaties(base, QueryUtility.years_mask(year_limit))
            if isinstance(year_limit, tuple) and len(year_limit) == 2:
                treaties2 = treaties2[
                    (year_limit[0] <= treaties2.signed_year) & (treaties2.signed_year <= year_limit[1])
                ]
            else:
                treaties2 = treaties2[treaties2.signed_year.isin(year_limit)]

        if isinstance(parties, list):
            # base = QueryUtility.query_treaties(base, QueryUtility.parties_mask(parties))
            treaties2 = treaties2.loc[(treaties2.party1.isin(parties)) | (treaties2.party2.isin(parties))]

        # if (treaty_filter or '') != '':
        treaties2 = self.filter_by_is_cultural(treaties2, treaty_filter)

        if recode_is_cultural:
            treaties2.loc[treaties2.is_cultural, "topic1"] = "7CORR"

        if treaty_sources is not None:
            treaties2 = treaties2.loc[treaties2.source.isin(treaty_sources)]

        return treaties2

    def get_categorized_treaties(
        self, treaties: pd.DataFrame | None = None, topic_category: dict[str, Any] | None = None, **kwargs
    ) -> pd.DataFrame:

        df: pd.DataFrame = self.get_treaties_within_division(treaties, **kwargs)
        df["topic_category"] = self.get_topic_category(df, topic_category, topic_column="topic1")
        return df

    def get_party_network(self, party_name: str, topic_category, parties: list[str], **kwargs) -> None | pd.DataFrame:

        treaty_ids = self.get_treaties_within_division(parties=parties, **kwargs).index

        treaties: pd.DataFrame = self.stacked_treaties.loc[treaty_ids]

        mask: pd.Series = treaties.party.isin(parties) if isinstance(parties, list) else ~treaties.reversed

        treaties = treaties.loc[mask]

        if treaties.shape[0] == 0:
            return None

        if kwargs.get("recode_is_cultural", False):
            treaties.loc[treaties.is_cultural, "topic"] = "7CORR"

        party_other_name = party_name.replace("party", "party_other")
        treaties = treaties[[party_name, party_other_name, "signed", "topic", "headnote"]]
        treaties.columns = ["party", "party_other", "signed", "topic", "headnote"]

        treaties["weight"] = 1.0
        treaties["category"] = self.get_topic_category(treaties, topic_category, topic_column="topic")

        return treaties.sort_values("signed")

    # def get_treaty_text_languages(self) -> pd.DataFrame:
    #     """Returns avaliable treaty text languages for treaties having a language mark in the wti-index.

    #         The languages of the compiled text are marked in columns 'english', 'french' and 'other'

    #         The only allowed value for each column are:
    #             'english': 'en'
    #             'french':  'fr'
    #             'other': 'it', 'de' or both

    #         The retrieval needs to handle the case when a treaty has two values in 'other' column.
    #         This is solved with the apply(split).apply(pd.Series).stack() chaining.

    #     Parameters:
    #     -----------

    #     Returns:
    #     -------
    #         DataFrame: index = treaty_id, columns = { 'language': 'en|fr|de|it' }

    #     """
    #     treaties: pd.DataFrame = self.treaties
    #     treaty_langs: pd.DataFrame = (
    #         pd.concat([treaties.english, treaties.french, treaties.other], axis=0)
    #         .dropna()
    #         .apply(lambda x: x.lower().replace(" ", "").split(","))
    #         .apply(pd.Series)
    #         .stack()
    #         .reset_index()[["treaty_id", 0]]
    #         .rename(columns={0: "language"})
    #     )
    #     return treaty_langs

    def get_continent_states(self) -> pd.Series:
        df: pd.DataFrame = self.parties
        df = df[~df.continent.isna()]
        cf: pd.Series = df[["continent"]].groupby("continent").apply(lambda x: list(x.index))
        return cf

    def get_wti_group_states(self) -> pd.Series:
        df: pd.DataFrame = self.parties
        df = df[~df.group_no.isin([1, 8])]
        cf: pd.Series = df[["group_name"]].groupby("group_name").apply(lambda x: list(x.index))
        return cf

    def get_party_preset_options(self) -> list[tuple[str, list[str]]]:

        preset_options: dict[str, list[str]] = dict(config.PARTY_PRESET_OPTIONS)

        preset_options.update(
            {
                "Regions: 1st + 2nd World": config.get_region_parties(1, 2),
                "Regions: 1st + 3rd World": config.get_region_parties(1, 3),
                "Regions: 2nd + 3rd World": config.get_region_parties(2, 3),
            }
        )

        countries: set[str] = set(self.get_countries_list()) - set(["ALL", "ALL OTHER"])

        for group_id in [1, 2, 3]:
            preset_options[f"Region: Not in World {group_id}"] = [
                x for x in countries if x not in set(config.get_region_parties(group_id))
            ]

        if self._party_preset_options is None:

            options: list[tuple[str, list[str]]] = []
            options += list(preset_options.items())
            options += [("Continent: " + x.title(), y) for x, y in self.get_continent_states().to_dict().items()]
            options += [("WTI:" + x, y) for x, y in self.get_wti_group_states().to_dict().items()]

            options = sorted(options, key=lambda x: x[0])

            self._party_preset_options = options

        return self._party_preset_options

    def get_language_column(self, language: str) -> str:
        language_column: str = ConfigValue("data.treaty_index.language.columns").resolve()[language]
        return language_column

    def get_treaties(
        self,
        language: str,
        period_group: str = "years_1945-1972",
        treaty_filter: str = "is_cultural",
        parties=None,
        treaty_sources=None,
    ) -> pd.DataFrame:
        treaties: pd.DataFrame = self.get_treaties_within_division(
            period_group=config.PERIOD_GROUPS_ID_MAP[period_group],
            treaty_filter=treaty_filter,
            recode_is_cultural=False,
            parties=parties,
            treaty_sources=treaty_sources,
        )
        treaties = treaties[treaties[self.get_language_column(language)] == language]
        treaties = treaties.sort_values("signed_year", ascending=True)
        return treaties
