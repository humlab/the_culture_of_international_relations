from itertools import product
import pandas as pd

def complete_missing_data_points(data, period_group, category_column='Party', value_column='Count'):

    categories = data[category_column].unique()

    if len(categories) == 0:
        return data

    periods = period_group['periods'] \
        if period_group['type'] == 'range' else [ '{} to {}'.format(x[0], x[1])
            for x in period_group['periods']]

    period_categories = pd.DataFrame(
        list(product(periods, categories)),
        columns=['Period', category_column])\
    .set_index(['Period', category_column])

    dx = pd.merge(period_categories, data, right_on=['Period', category_column], left_index=True, how='outer')
    df = pd.concat([data, dx[dx[value_column].isna()]], axis=0, ignore_index=True, copy=False)

    return df.fillna(0)

class QuantityByParty():

    @staticmethod
    def get_top_parties(stacked_treaties, period_group, party_name, n_top=3):
        period_column = period_group['column']
        # data = stacked_treaties.merge(wti_index.parties, how='inner', left_on='party', right_index=True)
        xd = stacked_treaties.groupby([period_column, party_name]).size().rename('TopCount').reset_index()
        top_list = xd.groupby([period_column]).apply(lambda x: x.nlargest(n_top, 'TopCount'))\
            .reset_index(level=0, drop=True)\
            .set_index([period_column, party_name])
        return top_list

    @staticmethod
    def get_treaties_statistics(
        wti_index=None,
        period_group=None,
        party_name='party_name',
        parties=None,
        treaty_filter='',
        extra_category='',
        n_top=0,
        year_limit=None,
        treaty_sources=None
    ):

        period_column = period_group['column']

        treaty_subset = wti_index.get_treaties_within_division(
            wti_index.treaties,
            period_group,
            treaty_filter,
            year_limit=year_limit,
            treaty_sources=treaty_sources
        )

        # Skapa urvalet från stacked_treaties så att vi kan gruppera på valda parties via column "party"
        # Regel: Filterera ut på treaty_ids, och forcera att "party" måste finnas i valda parter (vi vill inte gruppera på motpart såvida den inte finns i parties)
        # Ger överträffar för valda parties som har fördrag mellan varandra

        df = wti_index.stacked_treaties.loc[treaty_subset.index] #  wti_index.stacked_treaties[wti_index.stacked_treaties.index.isin(treaty_subset.index)]

        if 'ALL' in parties:
            print('ALL parties')
            treaties_of_parties = df[(df.reversed==False)]
        elif isinstance(parties, list) and len(parties) > 0:
            treaty_ids = treaty_subset[(treaty_subset.party1.isin(parties))|((treaty_subset.party2.isin(parties)))].index
            treaties_of_parties = df[df.index.isin(treaty_ids)&df.party.isin(parties)] # de avtal som vars länder valts
        else:
            df_top = QuantityByParty.get_top_parties(df, period_group=period_group, party_name=party_name, n_top=n_top)\
                        .drop(['TopCount'], axis=1)
            treaties_of_parties = df.merge(df_top, how='inner', left_on=[period_column, party_name], right_index=True)

        df_extra = None
        if extra_category == 'other_category':
            extra_ids = treaty_subset[~treaty_subset.index.isin(treaties_of_parties.index)].index
            df_extra = df[df.index.isin(extra_ids)&(df.reversed==False)]
            extra_party = wti_index.get_party('ALL OTHER')

        elif extra_category == 'all_category':
            df_extra = df[df.index.isin(treaty_subset.index)&(df.reversed==False)]
            extra_party = wti_index.get_party('ALL')

        if df_extra is not None:
            df_extra = df_extra.assign(party=extra_party['party'],
                                       party_name=extra_party['party_name'],
                                       party_short_name=extra_party['short_name'],
                                       party_country=extra_party['country'])

            treaties_of_parties = pd.concat([treaties_of_parties, df_extra])

        data = treaties_of_parties.groupby([period_column, party_name]).size().reset_index()\
                .rename(columns={ period_column: 'Period', party_name: 'Party', 0: 'Count' })

        data = complete_missing_data_points(data, period_group, category_column='Party', value_column='Count')

        return data

class QuantityByTopic():

    # FIXME: DEPRECATE THIS FUNCTION!
    @staticmethod
    def get_quantity_of_categorized_treaties(treaties, period_group, topic_category, recode_is_cultural):

        if period_group['column'] != 'signed_year':
            df = treaties[treaties[period_group['column']]!='OTHER']
        else:
            df = treaties[treaties.signed_year.isin(period_group['periods'])]

        if recode_is_cultural:
            df.loc[df.is_cultural, 'topic1'] = '7CORR'

        df['topic_category'] = df.apply(lambda x: topic_category.get(x['topic1'], 'OTHER'), axis=1)

        return df

    @staticmethod
    def get_treaty_topic_quantity_stat(
        wti_index,
        period_group,
        topic_category,
        party_group,
        recode_is_cultural,
        extra_other_category,
        target_quantity='topic',
        treaty_sources=None
    ):

        target_column = {
            'party': 'party',
            'topic': 'topic_category',
            'source': 'source',
            'continent': 'continent_code_other',
            'group': 'group_name_other'
        }

        treaties = wti_index.treaties.copy()
        parties = party_group['parties']

        categorized_treaties = wti_index.get_categorized_treaties(
            treaties,
            period_group=period_group,
            treaty_filter='',
            recode_is_cultural=recode_is_cultural,
            topic_category=topic_category,
            treaty_sources=treaty_sources
        )

        if not extra_other_category:
            categorized_treaties = categorized_treaties[categorized_treaties.topic_category!='OTHER']

        if party_group['label'] == 'ALL' or parties is None or 'ALL' in parties:
            parties_treaties = categorized_treaties
        else:
            mask = (categorized_treaties.party1.isin(parties)|(categorized_treaties.party2.isin(parties)))
            if party_group['label'] == 'ALL OTHER':
                parties_treaties = categorized_treaties.loc[~mask]
            else:
                parties_treaties = categorized_treaties.loc[mask]

        if parties_treaties.shape[0] == 0:
            print('No data for: ' + ','.join(parties))
            return None

        if target_quantity == 'topic':

            # Primary case - Compute sum of treaty count over all parties for each topic in selected topic group

            data = parties_treaties\
                    .groupby([period_group['column'], target_column[target_quantity]])\
                    .size()\
                    .reset_index()\
                    .rename(columns={ period_group['column']: 'Period', target_column[target_quantity]: 'Category', 0: 'Count' })

        else:

            # Special case - Compute treaty count per party for treaties with topic in selected treaty group
            stacked_treaties = pd.merge(wti_index.stacked_treaties[['party', 'party_other']], categorized_treaties, left_on='treaty_id', right_index=True, how='inner')

            party_continents = wti_index.parties[['continent_code', 'group_name']]
            stacked_treaties = pd.merge(stacked_treaties, party_continents, left_on=['party_other'], right_index=True, how='inner')\
                .rename(columns={'continent_code': 'continent_code_other', 'group_name': 'group_name_other'})

            stacked_treaties['continent_code_other'] = stacked_treaties['continent_code_other'].fillna(value='IO')

            if not 'ALL' in parties:
                 stacked_treaties = stacked_treaties.loc[stacked_treaties.party.isin(parties)]

            data = stacked_treaties\
                    .groupby([period_group['column'], target_column[target_quantity]])\
                    .size()\
                    .reset_index()\
                    .rename(columns={ period_group['column']: 'Period', target_column[target_quantity]: 'Category', 0: 'Count' })

        data = complete_missing_data_points(data, period_group, category_column='Category', value_column='Count')

        return data

