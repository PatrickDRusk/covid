import datetime
import io
import math
import os
import urllib
import zipfile

import numpy
import pandas


DOD_META = [
    ('AL', 7, 18, True),   ('AR', 5, 35, False),  ('AZ', 6, 18, True),   ('CA', 4, 28, False),
    ('CO', 10, 28, False), ('CT', 5, 20, True),   # ('DC', 7, 28, False), 
    ('DE', 4, 28, True),
    ('FL', 7, 24, False),   ('GA', 8, 30, True),   ('IA', 8, 21, True),   ('ID', 4, 28, False), # ('IL', 11, 28, False),
    ('IN', 5, 28, True),   ('KS', 5, 21, True),  # ('KY', 8, 28, False),
    ('LA', 8, 28, False),  ('MA', 11, 5, True),   # ('MD', 10, 32, False),
    ('ME', 4, 28, False),
    ('MI', 10, 21, True),  ('MN', 11, 28, False), ('MO', 5, 28, True),   ('MS', 8, 21, True),
    # ('MT', 10, 28, False),
    ('NC', 6, 28, True),   ('ND', 6, 25, True),   # ('NE', 7, 28, False),
    ('NH', 10, 28, False), ('NJ', 10, 50, True),  ('NM', 5, 28, False),  ('NV', 10, 24, True), # ('NY', 7, 28, False),
    ('OH', 9, 21, True),   # ('OK', 4, 35, False),
    ('OR', 5, 28, False),
    ('PA', 8, 28, True),   ('RI', 0, 20, True),   ('SC', 9, 27, True),   ('SD', 12, 21, True),
    ('TN', 9, 25, True),   ('TX', 4, 28, True),   ('UT', 2, 28, False),  ('VA', 5, 28, True),
    ('WA', 4, 28, False),  ('WI', 4, 28, False),  ('WV', 4, 28, False),  # ('WY', 12, 28, False),
]

DOD_STATES = {v[0] for v in DOD_META if v[3]}

PICKLE_FILES = False
HOSP_SHIFT = 5
IGNORE_DAYS = 21
RATIO_DAYS = 14


def download_path(fname):
    userroot = os.environ['HOME'] if 'HOME' in os.environ else f"{os.environ['HOMEDRIVE']}{os.environ['HOMEPATH']}"
    userroot = "/mnt/c/Users/Patri"
    return os.path.join(userroot, 'Downloads', fname)


def load_data(earliest_date, latest_date):
    earliest_date = pandas.Period(str(earliest_date), freq='D')
    if not latest_date:
        latest_date = (datetime.datetime.now() - datetime.timedelta(hours=19)).date()
    latest_date = pandas.Period(str(latest_date), freq='D')

    all_dates = pandas.period_range(start=earliest_date, end=latest_date, freq='D')

    # Get the state metadata
    uri = 'nyt_states_meta.csv'
    meta = pandas.read_csv(uri)
    meta['Country'] = 'USA'

    def handle_stats(fname, uri, func):
        if PICKLE_FILES:
            try:
                df = pandas.read_pickle(download_path(f'{fname}.pickle'))
            except FileNotFoundError:
                df = func(uri, meta, all_dates)
                df.to_pickle(download_path(f'{fname}.pickle'))
        else:
            df = func(uri, meta, all_dates)
        return df

    # Pull in state date-of-death data from CDC and reduce it to interesting columns
    # uri = "https://data.cdc.gov/api/views/r8kw-7aab/rows.csv?accessType=DOWNLOAD"
    uri = download_path("cdc_dod_data.csv")
    cdc_stats = handle_stats('cdc_stats', uri, load_cdc_dod_data)

    # Pull in the hospital information from the CDC
    uri = download_path("cdc_hospitalization_data.csv")
    hosp_stats = handle_stats('hosp_stats', uri, load_hospital_stats)

    all_stats_pre_hosp = cdc_stats.sort_index()
    all_stats = pandas.concat([all_stats_pre_hosp, hosp_stats], axis=1)
    all_stats = all_stats.join(meta.set_index('ST'))

    # Do the final date cut
    all_stats = all_stats.reset_index()
    all_stats = all_stats.set_index(['ST', 'Date']).sort_index()

    # Project the recent deaths for DoD states
    final_stats = project_dod_deaths(all_stats)

    final_stats = final_stats.reset_index()
    final_stats = final_stats[(final_stats.Date >= earliest_date) & (final_stats.Date <= latest_date)]
    final_stats = final_stats.set_index(['ST', 'Date'])

    return latest_date, meta, final_stats, cdc_stats, hosp_stats


def load_cdc_dod_data(uri, meta, all_dates):
    df = pandas.read_csv(uri, parse_dates=['End Date'])
    df = df[~df.State.isin(['United States', 'Puerto Rico'])]
    df = df[~df.Group.isin(['By Total'])]
    df = df.iloc[:, [2, 3, 4, 5, 6, 8, 9, 15, ]]
    df.columns = ['Date', 'Group', 'Year', 'Month', 'Week', 'State', 'C', 'CPF']
    df.Date = [pandas.Period(d, freq='D') for d in df.Date]
    df.loc[(df.Year == '2019/2020') & (df.Week == 1.0), 'Year'] = '2020'
    df.loc[(df.Year == '2020/2021') & (df.Week == 53.0), 'Year'] = '2020'
    df.loc[(df.Group == 'By Month') & pandas.isnull(df.CPF), 'CPF'] = 8.0
    df.loc[(df.Group == 'By Week') & pandas.isnull(df.CPF) & pandas.isnull(df.C), 'CPF'] = 3.0

    st_dfs = list()
    for st in df.State.unique():
        # print(st)
        st_df = df[df.State == st].sort_index().copy()

        # Calc yearly totals by year
        ytotals = {k: v['C'] for k, v in st_df[st_df.Group == 'By Year'][['Year', 'C']].set_index('Year').to_dict('index').items()}

        # Calc yearly totals by month
        m = st_df[st_df.Group == 'By Month'].copy()
        ybymtotals = {k: v['C'] for k, v in m.groupby('Year').sum()[['C']].to_dict('index').items()}
        for year in ybymtotals:
            my = m.loc[(m.Year == year) & pandas.isnull(m.C), :]
            tcpf = my.CPF.sum()
            vals = (my.CPF / tcpf) * (ytotals[year] - ybymtotals[year])
            m.loc[vals.index, ['C']] = vals

        # Calc totals for each year-month label
        ymtotals = {f"{k[0]}-{int(k[1])}": v['C'] for k, v in m.set_index(['Year', 'Month'])[['C']].to_dict('index').items()}

        # Prepare daily dataframe, including year-month label
        w = st_df[st_df.Group == 'By Week'][['Date', 'C', 'CPF']].set_index('Date')
        w.C = w.C / 7.0
        w.CPF = w.CPF / 7.0
        start = w.index[0] - 6
        end = w.index[-1]
        all_dates = pandas.period_range(start=start, end=end, freq='D')
        d = w.reindex(all_dates, method='bfill')
        d['YM'] = [f"{dt.year}-{dt.month}" for dt in d.index]
        d = d.loc['2020-01-01':, :].copy()

        # Use the year-month totals to fill in the daily values
        for ym in ymtotals:
            ym_df = d.loc[d.YM == ym, :]
            tc = ym_df.C.sum()
            ym_df = ym_df.loc[pandas.isnull(ym_df.C), :].copy()
            if len(ym_df):
                tcpf = ym_df.CPF.sum()
                vals = (ym_df.CPF / tcpf) * (ymtotals[ym] - tc)
                d.loc[vals.index, ['C']] = vals

        # Add the dataframe to the list
        if st == 'New York City':
            d['ST'] = 'NYC'
        else:
            d['ST'] = meta[meta.State == st].iloc[0, 0]
        d.index.name = 'Date'
        d = d.reset_index().set_index(['ST', 'Date'])[['C']]
        d.columns = ['Daily']
        if st == 'New York':
            ny_df = d.copy()
            continue
        elif st == 'New York City':
            both_df = pandas.concat([ny_df, d])
            both_df = both_df.reset_index().groupby('Date').sum()[['Daily']].reset_index()
            both_df.columns = ['Date', 'Daily']
            both_df['ST'] = 'NY'
            d = both_df.set_index(['ST', 'Date']).copy()

        d.Daily = d.Daily.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        d['Deaths'] = d.Daily.cumsum()
        st_dfs.append(d)
    dod_df = pandas.concat(st_dfs).sort_index()
    return dod_df


# noinspection PyUnusedLocal
def load_hospital_stats(uri, meta, all_dates):
    raw = pandas.read_csv(uri, parse_dates=['date'], low_memory=False)
    raw.date = [pandas.Period(d, freq='D') for d in raw.date]

    df = raw[['state', 'date',
              'previous_day_admission_adult_covid_confirmed',
              'previous_day_admission_adult_covid_suspected',
              'previous_day_admission_pediatric_covid_confirmed',
              'previous_day_admission_pediatric_covid_suspected',
              'total_adult_patients_hospitalized_confirmed_and_suspected_covid',
              'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid',
              ]].copy()
    df.columns = ['ST', 'Date',
                  'NewAConf', 'NewASusp', 'NewPConf', 'NewPSusp',
                  'TotA', 'TotP']

    df = df[df.Date >= pandas.Period('2020-08-01', freq='D')]
    df = df[~df.ST.isin(['PR', 'VI'])].copy()
    df.NewAConf = [float(str(v).replace(',', '')) for v in df.NewAConf]
    df.NewASusp = [float(str(v).replace(',', '')) for v in df.NewASusp]
    df.NewPConf = [float(str(v).replace(',', '')) for v in df.NewPConf]
    df.NewPSusp = [float(str(v).replace(',', '')) for v in df.NewPSusp]
    df.TotA = [float(str(v).replace(',', '')) for v in df.TotA]
    df.TotP = [float(str(v).replace(',', '')) for v in df.TotP]
    
    df = df.set_index(['ST', 'Date']).sort_index().copy()

    # Correcting two egregious errors
    df.loc[('NY', '2020-12-17'), 'TotP'] = 100
    df.loc[('TX', '2021-04-11'), 'TotP'] = 100
       
    df['NewConf'] = df.NewAConf + df.NewPConf
    df['NewSusp'] = df.NewASusp + df.NewPSusp
    df['New'] = df.NewConf + df.NewSusp
    df['NewHosp'] = df.NewConf + (df.NewSusp * 0.2)
    df['CurrHosp'] = df.TotA + df.TotP

    st_dfs = list()
    for st, st_df in df.reset_index().groupby('ST'):
        st_df = st_df.set_index('Date').sort_index().copy()
        st_df.NewConf = st_df.NewConf.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df.NewSusp = st_df.NewSusp.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df.New = st_df.New.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df.NewHosp = st_df.NewHosp.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        # st_df.NewHosp = st_df.NewHosp.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df.CurrHosp = st_df.CurrHosp.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_dfs.append(st_df.reset_index().set_index(['ST', 'Date'])[['NewHosp', 'CurrHosp']])

    return pandas.concat(st_dfs).sort_index()


def project_dod_deaths(stats):
    states = list()
    st_dfs = dict()
    for st, st_df in stats.reset_index().groupby('ST'):
        states.append(st)
        st_dfs[st] = st_df.set_index(['ST', 'Date'])

    for st in states:
        hosp_shift, ignore_days = HOSP_SHIFT, IGNORE_DAYS
        st_df = st_dfs[st].reset_index().set_index('Date')

        # Find the max date to keep raw data
        last_date = st_df.index.max()
        max_date = last_date - ignore_days

        # Calculate the ratio of hospitalizations to deaths in the RATIO_DAYS before that
        h = st_df.NewHosp.shift(hosp_shift).loc[max_date - RATIO_DAYS:max_date].sum()
        d = st_df.Daily.loc[max_date - RATIO_DAYS:max_date].sum()
        hd_ratio = h / d

        old_vals = st_df.Daily.loc[max_date:last_date]
        new_vals = st_df.NewHosp.shift(hosp_shift).loc[max_date:last_date] / hd_ratio
        st_df.loc[max_date:last_date, 'Daily'] = new_vals.combine(old_vals, max)
        st_df.Deaths = st_df.Daily.cumsum()

        st_dfs[st] = st_df.reset_index().set_index(['ST', 'Date'])

    return pandas.concat(list(st_dfs.values())).sort_index()


# noinspection DuplicatedCode
def get_infections_df(states, meta, death_lag, ifr_start, ifr_end, ifr_breaks, incubation, infectious,
                      max_confirmed_ratio=0.7):
    meta = meta.set_index('ST')
    avg_nursing = (meta.Nursing.sum() / meta.Pop.sum())

    ifr_breaks = [] if ifr_breaks is None else ifr_breaks
    new_states = []
    for st, state in states.reset_index().groupby('ST'):
        if st in ['AS']:
            continue

        state = state.set_index(['ST', 'Date']).copy()

        st_meta = meta.loc[st]
        st_nursing = st_meta.Nursing / st_meta.Pop
        nursing_factor = math.sqrt(st_nursing / avg_nursing)
        median_factor = (st_meta.Median / 38.2) ** 2
        # nursing has 51% correlation to PFR; median has -3%, but I still think it counts for something for IFR
        ifr_factor = ((2 * nursing_factor) + median_factor) / 3
        print(f'{st} {nursing_factor=:.2f} {median_factor=:.2f} {ifr_factor=:.2f}')

        # Calculate the IFR to apply for each day
        ifr = _calc_ifr(state, ifr_start, ifr_end, ifr_breaks) * ifr_factor
        # ifr = pandas.Series(numpy.linspace(ifr_high, ifr_low, len(state)), index=state.index)
        # Calculate the infections in the past
        infections = state.shift(-death_lag).Daily / ifr

        # Calculate the min infections based on max_confirmed_ratio
        # min_infections = state.Confirms7 / max_confirmed_ratio
        # infections = infections.combine(min_infections, max, 0)

        # Find out the ratio of hospitalizations that were detected on the last date in the past
        last_date = infections.index[-(death_lag + 1)]
        last_ratio = infections.loc[last_date] / (state.loc[last_date, 'CurrHosp'] + 1)
        # last_tests = state.loc[last_date, 'DTests7']
        # print(st, last_tests, state.DTests7.iloc[-death_lag:])

        # Apply that ratio to the dates since that date,
        infections.iloc[-death_lag:] = (state.CurrHosp.iloc[-death_lag:] * last_ratio)

        state['DPerM'] = state.Daily / state.Pop
        state['NewInf'] = infections
        state['NIPerM'] = state.NewInf / state.Pop
        state['TotInf'] = infections.cumsum()
        state['ActInf'] = infections.rolling(infectious).sum().shift(incubation)
        state['AIPer1000'] = state.ActInf / state.Pop / 1000.
        new_states.append(state)

    return pandas.concat(new_states)


def _calc_ifr(state, ifr_start, ifr_end, ifr_breaks):
    st, start = state.index[0]
    spans = []
    start_amt = ifr_start
    for end, end_amt in ifr_breaks:
        end = pandas.Period(end, 'D')
        idx = pandas.period_range(start=start, end=end, freq='D')
        spans.append(pandas.Series(numpy.linspace(start_amt, end_amt, len(idx)), index=idx).iloc[0:-1])
        start, start_amt = end, end_amt

    st, end = state.index[-1]
    idx = pandas.period_range(start=start, end=end, freq='D')
    spans.append(pandas.Series(numpy.linspace(start_amt, ifr_end, len(idx)), index=idx))
    span = pandas.concat(spans)
    span = pandas.Series(span.values, index=state.index)
    return span
