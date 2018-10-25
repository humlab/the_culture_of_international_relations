import common.utility as utility

def trim_period_group(period_group, year_limit):
    pg = dict(period_group)
    low, high = year_limit
    if pg['type'] == 'range':
        pg['periods'] = [ x for x in pg['periods'] if low <= x <= high ]
    else:
        pg['periods'] = [ (max(low,x), min(high, y)) for x, y in pg['periods'] if not (high < x and low > y) ]
    return pg


def period_group_years(period_group):
    if period_group['type'] == 'range':
        return period_group['periods']
    period_years = [ list(range(x[0], x[1] + 1)) for x in period_group['periods']]
    return utility.flatten(period_years)

class QueryUtility():
    
    @staticmethod
    def parties_mask(parties):
        return lambda df: (df.party1.isin(parties))|(df.party2.isin(parties))

    @staticmethod
    def period_group_mask(pg):
        return lambda df: df.signed_year.isin(period_group_years(pg))

    @staticmethod
    def is_cultural_mask():
        def fx(df):
            df.loc[df.is_cultural, 'topic1'] = '7CORR'
            return True
        return fx

    @staticmethod
    def years_mask(ys):
        return lambda df: ((ys[0] <= df.signed_year) & (df.signed_year <= ys[1])) \
            if isinstance(ys, tuple) and len(ys) == 2 else df.signed_year.isin(ys)

    @staticmethod
    def is_cultural_mask():
        return lambda df: (df.is_cultural)

    @staticmethod
    def topic_7cult_mask():
        return lambda df: (df.topic1=='7CULT')
        
    @staticmethod
    def query_treaties(treaties, filter_masks):
        """ NOT USED but could perhaps replace functions above """
        #  period_group=None, treaty_filter='', recode_is_cultural=False, parties=None, year_limit=None):
        if not isinstance(filter_masks, list):
            filter_masks = [ filter_masks ]
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
