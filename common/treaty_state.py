import os
import pandas as pd
import numpy as np
import re
import warnings
import logging
import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

import common.config as config
import common.utility as utility

logger = logging.getLogger(__name__)

default_treaties_skip_columns = [
    'extra_entry', 'dbflag', 'regis', 'regisant', 'vol', 'page', 'force', 'group1', 'group2'
]

treaties_column_names = [
    'sequence', 'treaty_id', 'is_cultural_yesno', 'english', 'french', 'other', 'source', 'vol', 'page', 'signed',
    'force', 'regis', 'regisant', 'party1', 'group1', 'party2', 'group2', 'laterality', 'headnote', 'topic', 'topic1', 'topic2',
    'title', 'extra_entry', 'dbflag', 'ispartyof4'
]

default_extra_parties = {
    'ALL OTHER': {
        'country': 'All other parties',
        'country_code': '88',
        'country_code3': '888',
        'group_name': 'All other parties',
        'group_no': 1,
        'party_name': 'All other',
        'processed': 1,
        'reverse_name': 'All other parties',
        'short_name': 'Rest'
    },
    'ALL': {
        'country': 'All parties',
        'country_code': '99',
        'country_code3': '999',
        'group_name': 'All parties',
        'group_no': 1,
        'party_name': 'All parties',
        'processed': 1,
        'reverse_name': 'All parties',
        'short_name': 'All'
    }
}

party_correction_map = {
    'ABLANI': 'ALBANI',
    'AMBASS': 'IGNORE',
    'AMBASS CONF': 'IGNORE',
    'AMBASS. CONF.': 'IGNORE',
    'ASIAN BANK': 'IGNORE',
    'BARBBAD': 'BARBAD',
    'BEGIU': 'BELGIU',
    'BEL-LUX. EC U': 'BNLXEC',
    'BEL-LUX. EC. U': 'BNLXEC',
    'BEL. LUX. EC. U': 'BNLXEC',
    'BELGIUM': 'BELGIU',
    'BLEGIU': 'BELGIU',
    'BRIT. INDIA': 'BRIT.INDIA',
    'BULGARI': 'BULGAR',
    'CAMBOD': 'KAMPUC',
    'CAMBODIA': 'KAMPUC',
    'CANDA': 'CANADA',
    'CAPEVER': 'CAPVER',
    'CENTRAL-AMERICAN COMMISSION ON ENVIRONMENT AND DEVELOPMENT': 'IGNORE',
    'CEZCHO': 'CZECHO',
    'COLOLMB': 'COLOMB',
    'COMMISSION FOR THE CONSERVATION OF SOUTHERN BLUEFIN TUNA': 'IGNORE',
    'EAST ATLANTIC FISHERIES COMMISSION': 'IGNORE',
    'ESTONIA': 'ESTONI',
    'EUROPEAN COMMUNITIES AND THEIR MEMBER STATES': 'IGNORE',
    'EUROPEAN MONETARY INSTITUTE': 'IGNORE',
    'GERMW*': 'GERMW',
    'HUNGARY': 'HUNGAR',
    'INTER-AMERICAN INSTITUTE FOR GLOBAL CHANGE RESEARCH': 'IGNORE',
    'INTERGOVERNMENTAL AUTHORITY ON DEVELOPMENT': 'IGNORE',
    'INTERGOVERNMENTAL AUTHORITY ON DROUGHT AND DEVELOPMENT': 'IGNORE',
    'INTERPOL': 'IGNORE',
    'IRELAND': 'IRELAN',
    'LATVI': 'LATVIA',
    'MACAU': 'MACAO',
    'MULTINATIONAL FORCE AND OBSERVERS': 'MFO',
    'NEGERI': 'NIGERI',
    'NETHR': 'NETHER',
    'PALISTINE': 'PALESTINE',
    'PARAGUA': 'PARAGU',
    'PORTUGAL': 'PORTUG',
    'PROTUG': 'PORTUG',
    'RELIEF BONDS': 'IGNORE',
    'SECRETARIAT OF THE CONVENTION ON BIOLOGICAL DIVERSITY': 'IGNORE',
    'SOUTH AFRICAN DEVELOPMENT COMMUNITY': 'IGNORE',
    'SOUTHERN AFRICAN DEVELOPMENT COMMUNITY': 'IGNORE',
    'ST. LUCIA': 'STLUC',
    'STRAITS': 'STRAIT',
    'SWEDE': 'SWEDEN',
    'TERKME': 'TURKME',
    'TJIKI': 'TAJIKI',
    'TRANS- JORDAN': 'TJORD',
    'TUKEY': 'TURKEY',
    'UN ECONOMIC COMMISSION FOR AFRICA': 'IGNORE',
    'UN UNIVERSITY': 'IGNORE',
    'UN*': 'IGNORE',
    'UNEP*': 'UNEP',
    'UNTAET': 'IGNORE',
    'UPEACE*': 'IGNORE',
    'US': 'USA',
    'VIETS(N)': 'VIETN',
    'W ALLIES': 'IGNORE'
}

from common.treaty_utility import trim_period_group, period_group_years, QueryUtility

class TreatyState:

    def __init__(self, data_folder='./data', skip_columns=default_treaties_skip_columns, period_groups=None): # pylint: disable=W0102
        self.data_folder = data_folder
        self.period_groups = period_groups or config.DEFAULT_PERIOD_GROUPS
        self.treaties_skip_columns = (skip_columns or []) + ['sequence', 'is_cultural_yesno']
        self.treaties_columns = treaties_column_names
        self.csv_files = [
            ('Treaties_Master_List_Treaties.csv', 'treaties', None),
            ('country_continent.csv', 'country_continent', None),
            ('parties_curated_parties.csv', 'parties', None),
            ('parties_curated_continent.csv', 'continent', None),
            ('parties_curated_group.csv', 'group', None)
        ]
        self.data = self._read_data()

        self.treaty_headnote_corpus = None
        self.tagged_headnotes = None

        self.groups = self.get_groups()
        self.continents = self.get_continents()
        self.parties = self.get_parties()

        self._treaties = None
        self._stacked_treaties = None
        self._get_countries_list = None
        self._party_preset_options = None

    def check_party(self):
        # party1 = self.treaties[~self.treaties.group1.isin([0, 8])].party1.unique().tolist()
        # party2 = self.treaties[~self.treaties.group2.isin([0, 8])].party2.unique().tolist()
        party1 = self.treaties.party1.unique().tolist()
        party2 = self.treaties.party2.unique().tolist()
        df_party = pd.DataFrame({ 'party': list(set(party1 + party2)) })
        df = df_party.merge(self.parties, left_on='party', right_index=True, how='left')
        
        unknown_parties = df[df.group_no.isna()].party.tolist()
        if len(unknown_parties) > 0:
            logger.warning('[{} UNKNOWN PARTIES] '.format(len(unknown_parties)) + (' '.join([ '"' + str(x) + '"' for x in unknown_parties])))

        return unknown_parties

    @property
    def treaties(self):
        if self._treaties is None:
            self._treaties = self._process_treaties()
        return self._treaties

    @property
    def stacked_treaties(self):
        if self._stacked_treaties is None:
            self._stacked_treaties = self._get_stacked_treaties()
        return self._stacked_treaties

    @property
    def cultural_treaties(self):
        return self.treaties[self.treaties.is_cultural]

    @property
    def cultural_treaties_of_interest(self):
        return self.cultural_treaties[(self.cultural_treaties.signed_period != 'OTHER')]

    def _read_data(self):
        data = {}
        na_values = ['#N/A','N/A', 'NULL', 'NaN', '-NaN']
        for (filename, key, _) in self.csv_files:
            path = os.path.join(self.data_folder, filename)
            # data[key] = pd.read_csv(path, sep='\t', low_memory=False, na_filter=False)
            data[key] = pd.read_csv(path, sep='\t', low_memory=False, keep_default_na=False, na_values=None)
        return data

    def get_treaty_period_group_categories(self, period_group, treaties=None):
        '''Returns treaties categorized according to given period group's divisions

        Parameters
        ----------
        treaties : DataFrame
            WTI Treaties. If none then entire self.treaties are categorized

        period_group: dict
            Period group that

        Returns
        -------
        A series indexed by treaty_id, and values assigned category division label
            e.g. '1939 to 1945' if division is (1939, 1945)

        '''
        periods = period_group['periods']
        column = period_group['column']

        if column in treaties.columns:
            return treaties[column]

        year_map = {
            year: '{} to {}'.format(d[0], d[1]) for d in periods for year in list(range(d[0], d[1] + 1))
        }

        series = treaties.signed_year.apply(lambda x: year_map.get(x, 'OTHER'))

        return series

    def _process_treaties(self):

        # def get_period(division, year):
        #    match = [ p for p in division if p[0] <= year <= p[1]]
        #    return '{} to {}'.format(match[0][0], match[0][1]) if len(match) > 0 else 'OTHER'

        treaties = self.data['treaties']
        treaties.columns = self.treaties_columns

        treaties['vol'] = treaties.vol.fillna(0).astype('int', errors='ignore')
        treaties['page'] = treaties.page.fillna(0).astype('int', errors='ignore')
        treaties['signed'] = pd.to_datetime(treaties.signed, errors='coerce')
        treaties['is_cultural_yesno'] = treaties.is_cultural_yesno.astype(str)
        treaties['signed_year'] = treaties.signed.apply(lambda x: x.year)

        for period_group in self.period_groups:
            column = period_group['column']
            if not column in treaties.columns:
                treaties[column] = self.get_treaty_period_group_categories(period_group, treaties)
                # treaties[column] = treaties.signed.apply(lambda x: get_period(definition['periods'], x.year))

        treaties['force'] = pd.to_datetime(treaties.force, errors='coerce')
        treaties['sequence'] = treaties.sequence.astype('int', errors='ignore')
        #treaties['group1'] = treaties.group1.fillna(0).astype('int', errors='ignore')
        #treaties['group2'] = treaties.group2.fillna(0).astype('int', errors='ignore')
        treaties['is_cultural'] = treaties.is_cultural_yesno.apply(lambda x: x.lower() == 'yes')
        treaties['headnote'] = treaties.headnote.fillna('').astype(str).str.upper()

        treaties['party1'] = treaties.party1.fillna('').astype(str).str.upper()
        treaties['party2'] = treaties.party2.fillna('').astype(str).str.upper()

        treaties['party1'] = treaties.party1.apply(lambda x: party_correction_map.get(x, x))
        treaties['party2'] = treaties.party2.apply(lambda x: party_correction_map.get(x, x))

        treaties.loc[(treaties.topic1=='7CULT')|(treaties.topic2=='7CULT'), 'topic'] = '7CULT'

        # Drop columns not used
        skip_columns = list(set(treaties.columns).intersection(set(self.treaties_skip_columns)))
        if skip_columns is not None and len(skip_columns) > 0:
            treaties.drop(skip_columns, axis=1, inplace=True)

        treaties = treaties.set_index('treaty_id')
        return treaties

    def _get_stacked_treaties(self):
        '''
        Returns a bi-directional (duplicated) and processed version of the treaties master list.
        Each treaty has two records where party1 and party2 are reversed:
            Record #1: party=party1, party_other=party2, reversed=False
            Record #2: party=party2, party_other=party1, reversed=True
        Fields are also added for the party's and party_other's country code (2 chars), continent and WTI group.
        The two rows are identical for all other fields.
        '''
        df1 = self.treaties\
                .rename(columns={
                    'party1': 'party',
                    'party2': 'party_other',
                    'group1': 'party_group_no',
                    'group2': 'party_other_group_no'
                })\
                .assign(reversed=False)

        df2 = self.treaties\
                .rename(columns={
                    'party2': 'party',
                    'party1': 'party_other',
                    'group2': 'party_group_no',
                    'group1': 'party_other_group_no'
                })\
                .assign(reversed=True)

        treaties = df1.append(df2) #.set_index(['treaty_id'])

        # Add fields for party's name, country, continent and WTI group
        parties = self.parties[['party_name', 'country_code', 'continent_code', 'group_name', 'short_name']]

        parties.columns = ['party_name', 'party_country', 'party_continent', 'party_group', 'party_short_name']
        treaties = treaties.merge(parties, how='left', left_on='party', right_index=True)

        # Add fields for party_other's country, continent and WTI group
        parties.columns = ['party_other_name', 'party_other_country', 'party_other_continent', 'party_other_group', 'party_other_short_name']
        treaties = treaties.merge(parties, how='left', left_on='party_other', right_index=True)

        # set 7CULT as topic when it is secondary topic
        treaties.loc[treaties.topic2=='7CULT', 'topic'] = '7CULT'

        return treaties

    # def get_stacked_treaties_subset(self, parties, complement=False):
    #     treaties = self.stacked_treaties[(self.stacked_treaties.signed_period!='OTHER')]
    #     if complement is False:
    #         treaties = treaties.loc[(treaties.party.isin(parties))]
    #     else:
    #         treaties = treaties.loc[(treaties.reversed==False)&(~treaties.party.isin(parties))]
    #     return treaties

    def get_continents(self):
        '''Returns continent reference data'''
        mask = self.data['continent']['country_code2'] != ''
        self.data['continent'] = self.data['continent'][mask]
        df = self.data['continent'].set_index('country_code2')
                
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis=1)
        name_map = { 'AS': 'ASIA', 'AF': 'AFRICA', 'EU': 'EUROPA', 'SA': 'SOUTH AMERICA', 'OC': 'OCEANIA', 'NA': 'NORTH AMERICA' }
        df['continent'] = df.continent_code.apply(lambda x: name_map.get(x, x))
        return df

    def get_groups(self):
        '''Returns WTI group codes'''

        df = self.data['group']\
            .drop(['Unnamed: 0'], axis=1)\
            .rename(columns={'GroupNo': 'group_no','GroupName': 'group_name'})\

        df['group_no'] = df.group_no.astype(np.int32)
        df['group_name'] = df.group_name.astype(str)

        df = df.set_index('group_no')

        return df

    def get_parties(self, extra_parties=default_extra_parties): # pylint: disable=W0102

        parties = self.data['parties']\
            .drop(['Unnamed: 0'], axis=1)\
            .dropna(subset=['PartyID'])\
            .rename(columns={
                'PartyID': 'party',
                'PartyName': 'party_name',
                'ShortName': 'short_name',
                'GroupNo': 'group_no',
                'reversename': 'reverse_name'
            })\
            .dropna(subset=['party'])\
            .set_index('party')

        parties['group_no'] = parties.group_no.astype(np.int32)
        parties['party_name'] = parties.party_name.apply(lambda x: re.sub(r'\(.*\)', '', x))
        parties['short_name'] = parties.short_name.apply(lambda x: re.sub(r'\(.*\)', '', x))
        parties[['party_name', 'short_name']] = parties[['party_name', 'short_name']].apply(lambda x: x.str.strip())

        parties.loc[(parties.group_no==8), ['country', 'country_code', 'country_code3']] = ''

        parties = pd.merge(parties, self.groups, how='left', left_on='group_no', right_index=True)

        parties = pd.merge(parties, self.continents, how='left', left_on='country_code', right_index=True)

        extra_keys = list(extra_parties.keys())
        extract_values = list(extra_parties.values())
        df = pd.DataFrame(extract_values, columns=extra_parties[extra_keys[0]].keys(), index=extra_parties.keys())
        parties = pd.concat([parties, df], axis=0)

        return parties

    def get_countries_list(self, excludes=None):

        if self._get_countries_list is None:
            parties = self.get_parties()
            parties = parties.loc[~parties.group_no.isin([0,8,11])]
            self._get_countries_list = list(parties.index)

        if len(excludes or []) > 0:
            return [x for x in self._get_countries_list if x not in excludes ]

        return self._get_countries_list

    def get_party_name(self, party, party_name_column):
        try:
            if party in self.parties.index:
                return self.parties.loc[party, party_name_column]
            return party
        except Exception as _:
            logger.warning('Warning: %s not in curated parties list', party)
            return party

    def get_party(self, party):
        try:
            d = self.parties.loc[party].to_dict()
            d['party'] = party
            return d
        except:  # pytlint: disable=W0702
            return None

    def get_headnotes(self):
        return self.treaties.headnote.fillna('').astype(str)

    def get_tagged_headnotes(self, tags=None):
        if self.tagged_headnotes is None:
            filename = os.path.join(self.data_folder, 'tagged_headnotes.csv')
            self.tagged_headnotes = pd.read_csv(filename, sep='\t').drop('Unnamed: 0', axis=1)
        if tags is None:
            return self.tagged_headnotes
        return self.tagged_headnotes.loc[(self.tagged_headnotes.pos.isin(tags))]

    def get_treaty_subset(self, options, language):
        lang_field = {'en': 'english', 'fr': 'french', 'de': 'other', 'it': 'other' }.get(language, None)
        df = self.treaties
        df = df.loc[df[lang_field]==language]
        if options.get('source', None) is not None:
            df = df.loc[df.source.isin(options.get('source', None))]
        if options.get('from_year', None) is not None:
            df = df.loc[df.signed >= datetime.date(options['from_year'], 1, 1)]
        if options.get('to_year', None) is not None:
            df = df.loc[df.signed < datetime.date(options['to_year']+1, 1, 1)]
        if options.get('parties', None) is not None:
            df = df.loc[df.party1.isin(options['parties'])|df.party2.isin(options['parties'])]
        return df #.set_index('treaty_id')

    def filter_by_is_cultural(self, df, treaty_filter):

        if treaty_filter == 'is_cultural':
            return df.loc[(df.is_cultural==True)]

        if treaty_filter == 'is_7cult':
            return df.loc[(df.topic1=='7CULT')]

        return df

    def get_topic_category(self, df, topic_category, topic_column='topic1'):
        if topic_column not in df.columns:
            raise Exception("Column {} not found i DataFrame".format(topic_column))
        if topic_category is not None:
            return df.apply(lambda x: topic_category.get(x[topic_column], 'OTHER'), axis=1)
        return df[topic_column]

    def get_treaties_within_division(self, treaties=None, period_group=None, treaty_filter='', recode_is_cultural=False, parties=None, year_limit=None):
        """Returns treaties filtered by given set of parameters.

        Parameters
        ----------
        treaties : DataFrame
            Optional. Treaty source to use instead of main WTI treaty index

        period_group: [dict]
            Period grouping to use for filtering. Treaties with signed year outside of group are filtered out.

        treaty_filter: str in [ 'is_cultural',  'is_7cult', '' ]
            Optional. 'is_cultural' filters out treaties where is_cultural is False, 'is_7cult' filters out treaties where topic1 is not '7cult'

        recode_is_cultural: Boolean
            Optional. Sets topic1 to '7CORR' for treaties having 'is_cultural' equal to true 
            
        parties: [str]
            Optional. Filters out treaties where neither of party1, party2 is in 'parties' list

        year_limit: tuple or list of ints
            Optional. Filters out treaties where signed_year is outside of given limit
            
        Returns
        -------
        DataFrame
        
            Remaining treaties after filters are applied.

        """
        if treaties is None:
            treaties = self.treaties

        assert period_group is not None, 'get_treaties_within_division: got None as period_group'

        period_column = period_group['column']
        
        assert period_column in treaties.columns, 'get_treaties_within_division: got unknown %r as column' % period_column

        if period_group is not None:
            treaties = QueryUtility.query_treaties(treaties, QueryUtility.period_group_mask(period_group))
        #if period_column != 'signed_year':
        #    df = treaties[treaties[period_column] != 'OTHER']
        #else:
        #    df = treaties[treaties.signed_year.isin(period_group['periods'])]

        if year_limit is not None:
            #treaties = QueryUtility.query_treaties(treaties, QueryUtility.years_mask(year_limit))
            if isinstance(year_limit, tuple) and len(year_limit) == 2:
                treaties = treaties[(year_limit[0]<=treaties.signed_year)&(treaties.signed_year <= year_limit[1])]
            else:
                treaties = treaties[treaties.signed_year.isin(year_limit)]

        if isinstance(parties, list):
            #treaties = QueryUtility.query_treaties(treaties, QueryUtility.parties_mask(parties))
            treaties = treaties.loc[(treaties.party1.isin(parties))|(treaties.party2.isin(parties))]

        #if (treaty_filter or '') != '':
        treaties = self.filter_by_is_cultural(treaties, treaty_filter)

        if recode_is_cultural:
            treaties.loc[treaties.is_cultural, 'topic1'] = '7CORR'

        return treaties

    def get_categorized_treaties(self, treaties=None, topic_category=None, **kwargs):

        df = self.get_treaties_within_division(treaties, **kwargs)
        df['topic_category'] = self.get_topic_category(df, topic_category, topic_column='topic1')
        return df
   
    def get_party_network(self, party_name, topic_category, parties, **kwargs):

        treaty_ids = self.get_treaties_within_division(parties=parties, **kwargs).index

        treaties = self.stacked_treaties.loc[treaty_ids]

        mask = treaties.party.isin(parties) if isinstance(parties, list) else ~treaties.reversed

        treaties = treaties.loc[mask]

        if treaties.shape[0] == 0:
            return None

        if kwargs.get('recode_is_cultural', False):
            treaties.loc[treaties.is_cultural, 'topic'] = '7CORR'

        party_other_name = party_name.replace('party', 'party_other')
        treaties = treaties[[ party_name, party_other_name, 'signed', 'topic', 'headnote']]
        treaties.columns = [ 'party', 'party_other', 'signed', 'topic', 'headnote']

        treaties['weight'] = 1.0
        treaties['category'] = self.get_topic_category(treaties, topic_category, topic_column='topic')

        return treaties.sort_values('signed')
    
    def get_treaty_text_languages(self):
        '''Returns avaliable treaty text languages for treaties having a language mark in the wti-index.
    
            The languages of the compiled text are marked in columns 'english', 'french' and 'other'
            
            The only allowed value for each column are:
                'english': 'en'
                'french':  'fr'
                'other': 'it', 'de' or both
                
            The retrieval needs to handle the case when a treaty has two values in 'other' column.
            This is solved with the apply(split).apply(pd.Series).stack() chaining.
            
        Parameters:
        -----------
        
        Returns:
        -------
            DataFrame: index = treaty_id, columns = { 'language': 'en|fr|de|it' }
            
        '''
        treaties = self.treaties
        treaty_langs = pd.concat([treaties.english, treaties.french, treaties.other], axis=0)\
            .dropna()\
            .apply(lambda x: x.lower().replace(' ','').split(','))\
            .apply(pd.Series)\
            .stack()\
            .reset_index()[['treaty_id', 0]]\
            .rename(columns={0: 'language'})
        return treaty_langs

    def get_continent_states(self):
        df = self.parties
        df = df[~df.continent.isna()]
        df = df[['continent']].groupby('continent').apply(lambda x: list(x.index))
        return df
    
    def get_wti_group_states(self):
        df = self.parties
        df = df[~df.group_no.isin([1, 8])]
        df = df[['group_name']].groupby('group_name').apply(lambda x: list(x.index))
        return df
    
    def get_party_preset_options(self):
        if self._party_preset_options is None:
            options = [  ]
            options += [ (x, y) for x,y in config.PARTY_PRESET_OPTIONS.items() ]
            options += [ (x, y) for x,y in self.get_continent_states().to_dict().items() ]
            options += [ ('WTI:' + x, y) for x,y in self.get_wti_group_states().to_dict().items() ]
            options = sorted(options, key=lambda x: x[0])
            self._party_preset_options = options
        return self._party_preset_options
    
    def get_treaties(self, language, period_group='years_1945-1972', treaty_filter='is_cultural', parties=None):
        period_group = config.PERIOD_GROUPS_ID_MAP[period_group]
        treaties = self.get_treaties_within_division(
            period_group=period_group,
            treaty_filter=treaty_filter,
            recode_is_cultural=False,
            parties=parties
        )
        treaties = treaties[treaties[config.LANGUAGE_MAP[language]]==language]
        treaties = treaties.sort_values('signed_year', ascending=True)
        return treaties
    
    def get_treaty_time_groupings(self):
        groups = {
            'treaty_id': { 'column': 'treaty_id', 'divisor': None, 'title': 'Treaty', 'fx': None},
            'signed_year': { 'column': 'signed_year', 'divisor': 1, 'title': 'Year', 'fx': None },
            'signed_lustrum': { 'column': 'signed_lustrum', 'divisor': 5, 'title': 'Lustrum', 'fx': lambda df: utility.trunc_year_by(df.signed_year, 5) },
            'signed_decade': { 'column': 'signed_decade', 'divisor': 10, 'title': 'Decade', 'fx': lambda df: utility.trunc_year_by(df.signed_year, 10) }
        }
        return groups
    
def load_wti_index(data_folder, skip_columns=default_treaties_skip_columns, period_groups=None):
    try:
        period_groups = period_groups or config.DEFAULT_PERIOD_GROUPS
        state = TreatyState(data_folder, skip_columns, period_groups)
        logger.info("WTI index loaded!")
        return state
    except Exception as ex:
        logger.error(ex)
        #raise
        logger.info('Load failed! Have you run setup cell above?')
        return None

load_treaty_state = load_wti_index
