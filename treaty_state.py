import os
import pandas as pd
import numpy as np
import re

class TreatyState:
    
    def __init__(self, data_folder='./data', divisions=None, skip_columns=None):
        self.data_folder = data_folder
        self.period_divisions = divisions or [
            [ (1919, 1939), (1940, 1944), (1945, 1955), (1956, 1966), (1967, 1972) ],
            [ (1919, 1944), (1945, 1955), (1956, 1966), (1967, 1972) ]
        ]
        self.treaties_skip_columns = skip_columns or [
            'extra_entry', 'dbflag', 'dummy1', 'english', 'french', 'ispartyof4', 'other',
            'regis', 'regisant', 'vol', 'page', 'force', 'group1', 'group2'
        ]
        self.treaties_columns = [
            'sequence',
            'treaty_id',
            'is_cultural_yesno',
            'english',
            'french',
            'other',
            'source',
            'vol',
            'page',
            'signed',
            'force',
            'regis',
            'regisant',
            'party1',
            'group1',
            'party2',
            'group2',
            'laterality',
            'headnote',
            'topic',
            'topic1',
            'topic2',
            'title',
            'extra_entry',
            'dbflag',
            'ispartyof4',
            'dummy1'
        ]
        self.csv_files = [
            ('Treaties_Master_List_Treaties.csv', 'treaties', None),
            ('country_continent.csv', 'country_continent', None),
            ('parties_curated_parties.csv', 'parties', None),
            ('parties_curated_continent.csv', 'continent', None),
            ('parties_curated_group.csv', 'group', None)
        ]
        self.data = self.read_data(data_folder)
        self.treaty_headnote_corpus = None
        
    def read_data(self, data_folder):
        data = {}
        for (filename, key, dtype) in self.csv_files:
            path = os.path.join(self.data_folder, filename)
            data[key] = pd.read_csv(path, sep='\t', low_memory=False)
            print('Imported: {}'.format(filename))
            
        return data
    
    def process_treaties(self):

        def get_period(division, year):
            match = [ p for p in division if p[0] <= year <= p[1]]
            return '{} to {}'.format(match[0][0], match[0][1]) if len(match) > 0 else 'other'
    
        treaties = self.data['treaties']
        treaties.columns = self.treaties_columns
        
        treaties['vol'] = treaties.vol.fillna(0).astype('int', errors='ignore')
        treaties['page'] = treaties.page.fillna(0).astype('int', errors='ignore')
        treaties['signed'] = pd.to_datetime(treaties.signed, errors='coerce')
        treaties['is_cultural_yesno'] = treaties.is_cultural_yesno.astype(str)
        treaties['signed_year'] = treaties.signed.apply(lambda x: x.year)
        treaties['signed_period'] = treaties.signed.apply(lambda x: get_period(self.period_divisions[0], x.year))
        treaties['signed_period_alt'] = treaties.signed.apply(lambda x: get_period(self.period_divisions[1], x.year))
        treaties['force'] = pd.to_datetime(treaties.force, errors='coerce')
        treaties['sequence'] = treaties.sequence.astype('int', errors='ignore')
        treaties['group1'] = treaties.group1.fillna(0).astype('int', errors='ignore')
        treaties['group2'] = treaties.group2.fillna(0).astype('int', errors='ignore')
        treaties['is_cultural'] = treaties.is_cultural_yesno.apply(lambda x: x.lower() == 'yes')
        treaties['headnote'] = treaties.headnote.fillna('').astype(str).str.upper()

        # Drop columns not used
        treaties.drop(self.treaties_skip_columns, axis=1, inplace=True)
        treaties = treaties.set_index(['treaty_id'])
        return treaties

    def get_stacked_treaties(self):
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
        
        # Add fields for party's country, continent and WTI group
        parties = self.parties[['country_code', 'continent_code', 'group_name']]
        
        parties.columns = ['party_country', 'party_continent', 'party_group']
        treaties = treaties.merge(parties, how='left', left_on='party', right_index=True)
        
        # Add fields for party_other's country, continent and WTI group
        parties.columns = ['party_other_country', 'party_other_continent', 'party_other_group']
        treaties = treaties.merge(parties,how='left', left_on='party_other', right_index=True)
        
        # Drop columns
        treaties = treaties.drop(['sequence', 'is_cultural_yesno'], axis=1)
        
        # set 7CULT as topic when it is secondary topic
        treaties.loc[treaties.topic2=='7CULT', 'topic'] = '7CULT'
        
        return treaties
    
    def get_continents(self):
        
        df = self.data['continent'].drop(['Unnamed: 0'], axis=1).set_index('country_code2')
            
        return df
    
    def get_groups(self):
        
        df = self.data['group']\
            .drop(['Unnamed: 0'], axis=1)\
            .rename(columns={'GroupNo': 'group_no','GroupName': 'group_name'})\

        df['group_no'] = df.group_no.astype(np.int32)
        df['group_name'] = df.group_name.astype(str)
        
        df = df.set_index('group_no')

        return df
        
    def get_parties(self):
        
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

        parties.loc[(parties.group_no==8), ['country', 'country_code', 'country_code3']] = ''

        parties = pd.merge(parties, self.groups, how='left', left_on='group_no', right_index=True)
        parties = pd.merge(parties, self.continents, how='left', left_on='country_code', right_index=True)
        
        return parties
    
    def get_party_name(self, party, party_name_column):
        try:
            if party in self.parties.index:
                return self.parties.loc[party, party_name_column]
            return party
        except:
            print('Warning: {} not in curated parties list'.format(party))
            return party
        
    def process(self):
        
        self.groups = self.get_groups()
        self.continents = self.get_continents()
        self.parties = self.get_parties()
        
        self.treaties = self.process_treaties()
        self.stacked_treaties = self.get_stacked_treaties()
        
        self.cultural_treaties = self.treaties[self.treaties.is_cultural]
        self.cultural_treaties_of_interest = self.cultural_treaties[(self.cultural_treaties.signed_period != 'other')]
        
        print('Number of treaties loaded: {}'.format(len(self.treaties)))
        print('Number of cultural treaties: {} (total), {} within periods'.format(
            len(self.cultural_treaties),
            len(self.cultural_treaties_of_interest)
        ))
        
        self.tagged_headnotes = None
        return self

    def get_headnotes(self):
        return self.treaties.headnote.fillna('').astype(str)
    
    def get_tagged_headnotes(self, tags=None):
        if self.tagged_headnotes is None:
            filename = os.path.join(self.data_folder, 'tagged_headnotes.csv')
            self.tagged_headnotes = pd.read_csv(filename, sep='\t').drop('Unnamed: 0', axis=1)
        if tags is None:
            return self.tagged_headnotes
        return self.tagged_headnotes.loc[(self.tagged_headnotes.pos.isin(tags))]
