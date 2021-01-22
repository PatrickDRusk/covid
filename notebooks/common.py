import datetime
import io
import math
import os
import urllib
import zipfile

import numpy
import pandas
import scipy.stats


# Set up strategies for smoothing data around weekends and holidays.
# This differs considerably state by state, though many states fall into groupings
SMOOTH_CONFIGS = dict(
    SatSun=
        dict(
            # Ignore the values reported on these days
            DaysOfWeek = ('W-SAT', 'W-SUN', ),
            # Also ignore these days around holidays
            Holidays = (
                '05-23-2020', '05-26-2020', '05-27-2020',  # Memorial Day
                '07-03-2020', '07-04-2020', # Independence Day
                '09-05-2020', '09-08-2020', '09-09-2020',  # Labor Day
                '2020-11-26', '2020-11-27', '2020-11-30', '2020-12-01', # Thanksgiving
                '2020-12-24', '2020-12-25', '2020-12-28', '2020-12-29', '2020-12-30', # Christmas
                '2021-01-01', # New Year's
                '2021-01-18', # MLK
            )
        ),
    SatSunMon=
        dict(
            DaysOfWeek = ('W-SAT', 'W-SUN', 'W-MON', ),
            Holidays = (
                '05-23-2020', '05-26-2020', '05-27-2020',  # Memorial Day
                '07-03-2020', '07-04-2020', # Independence Day
                '09-05-2020', '09-08-2020', '09-09-2020',  # Labor Day
                '2020-11-26', '2020-11-27', '2020-12-01', # Thanksgiving
                '2020-12-24', '2020-12-25', '2020-12-29', '2020-12-30', # Christmas
                '2021-01-01', # New Year's
                '2021-01-19', # MLK
            )
        ),
    SunMon=
        dict(
            DaysOfWeek = ('W-SUN', 'W-MON'),
            Holidays = (
                '05-23-2020', '05-26-2020', '05-27-2020',  # Memorial Day
                '07-03-2020', '07-04-2020', # Independence Day
                '09-05-2020', '09-06-2020', '09-08-2020', '09-09-2020',  # Labor Day
                '2020-11-26', '2020-11-27', '2020-11-28', '2020-12-01', # Thanksgiving
                '2020-12-24', '2020-12-25', '2020-12-26', '2020-12-29', '2020-12-30', # Christmas
                '2021-01-01', '2021-01-02', # New Year's
                '2021-01-19', # MLK
            )
        ),
    Alabama=
        dict(
            DaysOfWeek = ('W-SUN', 'W-MON', 'W-TUE'),
            Holidays = (
                '05-23-2020', '05-26-2020', '05-27-2020',  # Memorial Day
                '07-03-2020', '07-04-2020', # Independence Day
                '09-05-2020', '09-06-2020', '09-08-2020', '09-09-2020',  # Labor Day
                '2020-11-26', '2020-11-27', '2020-11-28', # Thanksgiving
                # Alabama undertook a massive re-check of death numbers that plopped a bunch
                # of deaths in January, particularly 1/12, but other days, too
                '2020-12-17', '2020-12-18', '2020-12-23', '2020-12-24', '2020-12-25', '2020-12-26', # Christmas
                '2020-12-30', '2020-12-31', '2021-01-01', '2021-01-02',
                '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-13', # New Year's
            )
        ),
    Kansas=
        dict(
            # Ignore the values reported on these days
            DaysOfWeek = ('W-SAT', 'W-SUN', ),
            # Also ignore these days around holidays
            Holidays = (
                '05-23-2020', '05-26-2020', '05-27-2020',  # Memorial Day
                '07-03-2020', '07-04-2020', # Independence Day
                '09-05-2020', '09-08-2020', '09-09-2020',  # Labor Day
                '2020-11-26', '2020-11-27', '2020-11-30', '2020-12-01', # Thanksgiving
                '2020-12-24', '2020-12-25', '2020-12-28', '2020-12-29', # Christmas
                '2021-01-01', '2021-01-04', '2021-01-05', # New Year's
                '2021-01-18', # MLK
            )
        ),
    NewYork=
        dict(
            DaysOfWeek = (),
            Holidays = (
                '04-30-2020', '05-01-2020', '05-02-2020',
                '05-03-2020', '05-04-2020', '05-05-2020', 
                '05-23-2020', '05-24-2020', '05-25-2020',  # Memorial Day
                '2020-11-26', '2020-11-27', '2020-11-28', '2020-11-29', '2020-11-30', '2020-12-01', # Thanksgiving
            )
        ),
    RhodeIsland=
        dict(
            DaysOfWeek = ('W-SUN', 'W-MON'),
            Holidays = (
                '05-23-2020', '05-26-2020', '05-27-2020',  # Memorial Day
                '07-03-2020', '07-04-2020', # Independence Day
                '09-05-2020', '09-06-2020', '09-08-2020', '09-09-2020',  # Labor Day
                '2020-11-26', '2020-11-27', '2020-11-28', '2020-12-01', # Thanksgiving
                '2020-12-23', '2020-12-24', '2020-12-25', '2020-12-26', '2020-12-29', # Christmas
                '2021-01-01', '2021-01-02', # New Year's
                '2021-01-19', # MLK
            )
        ),
    Wyoming=
        dict(
            DaysOfWeek = (),
            Holidays = (
                '2020-11-19', '2020-11-23', '2020-11-24', '2020-11-25', '2020-11-26',
                '2020-12-01', '2020-12-02', '2020-12-03', # Thanksgiving
                '2020-12-09', '2020-12-11', '2020-12-12', '2020-12-15', '2020-12-16', '2020-12-17', # random anomalies
                '2020-12-31', # New Year's
                '2021-01-18', # MLK
            )
        ),
)

# Assign states to the various smoothing strategies
SMOOTH_MAPS = dict(
    SatSun=('ID', 'UT', ),
    SatSunMon=('CA', 'CO', 'DE', 'IA', 'IL', 'LA', 'MT', 'NM', 'OH', 'SC', 'WV', ),
    SunMon=('AR', 'AZ', 'HI', 'KY', 'MD', 'MN',
       'MS', 'NE', 'NH', 'NJ', 'OK', 'OR', 'SD', 'WA', 'WI', ),
    Alabama=('AL', ),
    Kansas=('KS', ),
    NewYork=('NY', ),
    RhodeIsland=('RI', ),
    Wyoming=('WY', ),
)

# This will hold the series of dates per state that need smoothing
SMOOTH_DATES = dict()


def download_path(fname):
    return os.path.join(os.environ['HOME'], 'Downloads', fname)


def load_data(earliest_date, latest_date):
    if not latest_date:
        latest_date = pandas.Period((datetime.datetime.now() - datetime.timedelta(hours=19)).date(),
                                    freq='D')

    create_smooth_dates(earliest_date, latest_date)

    # Get the state metadata
    meta = pandas.read_csv('nyt_states_meta.csv')
    meta['Country'] = 'USA'

    # Pull in state data from NY Times and reduce it to interesting columns,
    # joined with the metadata above
    uri = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
    nyt_stats = read_nyt_csv(uri, meta, earliest_date, latest_date)
    uri = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-states.csv'
    nyt_stats_live = read_nyt_csv(uri, meta, earliest_date, latest_date)

    # Attach the live stats if the daily file has not yet rolled
    if len(nyt_stats_live) and (nyt_stats.Date.max() < nyt_stats_live.Date.max()):
        print("Pulling in live stats")
        nyt_stats = pandas.concat([nyt_stats, nyt_stats_live], sort=False)
        nyt_stats = nyt_stats.sort_values(['State', 'Date'])
        nyt_stats.index = list(range(len(nyt_stats)))

    ct = fix_state_data(load_ct_data(), earliest_date, latest_date, latest_days=6)
    replace_state_data(nyt_stats, ct, 'Connecticut')

    fl = fix_state_data(load_fl_data(), earliest_date, latest_date, latest_days=14, decay=1.5)
    replace_state_data(nyt_stats, fl, 'Florida')

    ga = fix_state_data(load_ga_data(), earliest_date, latest_date, latest_days=9)
    replace_state_data(nyt_stats, ga, 'Georgia')

    in_ = fix_state_data(load_in_data(), earliest_date, latest_date, latest_days=7)
    replace_state_data(nyt_stats, in_, 'Indiana')

    ma = fix_state_data(load_ma_data(), earliest_date, latest_date, latest_days=6)
    replace_state_data(nyt_stats, ma, 'Massachusetts')

    mi = fix_state_data(load_mi_data(), earliest_date, latest_date, latest_days=11)
    replace_state_data(nyt_stats, mi, 'Michigan')

    mo = fix_state_data(load_mo_data(), earliest_date, latest_date, latest_days=14, decay=1.5)
    replace_state_data(nyt_stats, mo, 'Missouri')

    nc = fix_state_data(load_nc_data(), earliest_date, latest_date, latest_days=6)
    replace_state_data(nyt_stats, nc, 'North Carolina')

    nv = fix_state_data(load_nv_data(), earliest_date, latest_date, latest_days=8)
    replace_state_data(nyt_stats, nv, 'Nevada')

    pa = fix_state_data(load_pa_data(), earliest_date, latest_date, latest_days=9)
    replace_state_data(nyt_stats, pa, 'Pennsylvania')

    tn = fix_state_data(load_tn_data(), earliest_date, latest_date, latest_days=8)
    replace_state_data(nyt_stats, tn, 'Tennessee')

    tx = fix_state_data(load_tx_data(), earliest_date, latest_date, latest_days=9)
    replace_state_data(nyt_stats, tx, 'Texas')

    va = fix_state_data(load_va_data(), earliest_date, latest_date, latest_days=11)
    replace_state_data(nyt_stats, va, 'Virginia')

    # Pull in the testing information from the COVID Tracking Project
    ct_stats = pandas.read_csv('https://covidtracking.com/api/v1/states/daily.csv', low_memory=False)

    # Remove territories
    ct_stats = ct_stats[~ct_stats.state.isin(['AS', 'GU', 'MP', 'PR', 'VI'])].copy()
    ct_stats.date = [pandas.Period(str(v)) for v in ct_stats.date]

    # Choose and rename a subset of columns
    ct_stats = ct_stats[['date', 'state', 'positive', 'negative']]
    ct_stats.columns = ['Date', 'ST', 'Pos', 'Neg']

    # Set the index to state and date
    ct_stats = ct_stats[ct_stats.Date >= earliest_date]
    ct_stats = ct_stats[ct_stats.Date <= latest_date]
    ct_stats = ct_stats.set_index(['ST', 'Date'])

    # Pull in the statistics for states
    ct_stats = ct_stats.join(meta.set_index('ST')).reset_index().sort_values(['ST', 'Date'])

    return latest_date, meta, nyt_stats, ct_stats


def load_ct_data():
    uri = ("https://data.ct.gov/api/views/abag-bjkj/rows.csv?accessType=DOWNLOAD")
    ct = pandas.read_csv(uri, parse_dates=['Date of death'])
    ct = ct[['Date of death', 'Total deaths']].sort_values('Date of death')
    ct.columns = ['Date', 'Deaths']
    ct.Date = [pandas.Period(d.date(), freq='D') for d in ct.Date]
    ct.Deaths = ct.Deaths.cumsum()
    all_dates = pandas.period_range(start='2020-01-01', end=ct.Date.iloc[-1], freq='D')
    ct = ct.set_index('Date').reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    ct.columns = ['Date', 'Deaths']
    return ct


def load_fl_data():
    uri = download_path('Florida_COVID19_Case_Line_Data.csv')
    fl = pandas.read_csv(uri, parse_dates=['EventDate'])
    df = fl[['Died', 'EventDate']].copy()
    df = df[df.Died.isin(('Yes', 'Recent'))][['EventDate']].copy()
    df['Date'] = [pandas.Period(d.date(), freq='D') for d in df.EventDate]
    df = df.groupby('Date').count().sort_index().cumsum()
    all_dates = pandas.period_range(start='2020-01-01', end=df.index[-1], freq='D')
    df = df.reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    df.columns = ['Date', 'Deaths']
    return df


def load_ga_data():
    uri = "https://ga-covid19.ondemand.sas.com/docs/ga_covid_data.zip"
    df = download_zipped_df(uri, 'epicurve_symptom_date.csv', parse_dates=['symptom date'])
    df = df[df.measure == 'state_total'][['symptom date', 'death_cum']].copy()
    df.columns = ['Date', 'Deaths']
    df.Date = [pandas.Period(str(v), freq='D') for v in df.Date]
    return df


def load_in_data():
    uri = ("https://hub.mph.in.gov/dataset/6bcfb11c-6b9e-44b2-be7f-a2910d28949a/resource/"
           "7661f008-81b5-4ff2-8e46-f59ad5aad456/download/covid_report_death_date_agegrp.xlsx")
    in_ = pandas.read_excel(uri, parse_dates=['date'])[['date', 'covid_deaths']].copy()
    in_ = in_.groupby('date').sum().cumsum().reset_index()
    in_.columns = ['Date', 'Deaths']
    in_.Date = [pandas.Period(str(v), freq='D') for v in in_.Date]
    all_dates = pandas.period_range(start='2020-03-01', end=in_.Date.max(), freq='D')
    in_ = in_.set_index('Date').reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    in_.columns = ['Date', 'Deaths']
    return in_


def load_ma_data():
    df = pandas.read_excel(download_path('covid-19-dashboard.xlsx'),
                           sheet_name='DateofDeath').iloc[:, [0, 2, 4]]
    df.columns = ['Date', 'Confirmed', 'Probable']
    df['Deaths'] = df.Confirmed + df.Probable
    df.Date = [pandas.Period(str(v), freq='D') for v in df.Date]
    return df[['Date', 'Deaths']].copy()


def load_mi_data():
    uri = (download_path('Cases_and_Deaths_by_County_and_by_Date_of_Symptom_Onset_or_by_Date_of_Death.xlsx'))
    mi = pandas.read_excel(uri, parse_dates=['Date'])
    mi = mi.groupby('Date').sum()[['Deaths.Cumulative']].reset_index()
    mi.columns = ['Date', 'Deaths']
    mi.Date = [pandas.Period(str(v), freq='D') for v in mi.Date]
    return mi


def load_mo_data():
    uri = ("https://results.mo.gov/t/COVID19/views/COVID-19DataforDownload/MetricsbyDateofDeath.csv")
    mo = pandas.read_csv(uri).iloc[1:-1, :][['Dod', 'Measure Values']].copy()
    mo.columns = ['Date', 'Deaths']
    mo.Date = [pandas.Period(str(v), freq='D') for v in mo.Date]
    mo = mo[mo.Date >= pandas.Period('2020-01-01', freq='D')].set_index('Date')
    mo.Deaths = [int(x) for x in mo.Deaths]
    mo.Deaths = mo.Deaths.cumsum()
    all_dates = pandas.period_range(start='2020-01-01', end=mo.index[-1], freq='D')
    mo = mo.reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    mo.columns = ['Date', 'Deaths']
    return mo


def load_nc_data():
    uri = download_path('TABLE_DAILY_CASE&DEATHS_METRICS_data.csv')
    nc = pandas.read_csv(uri, encoding='utf_16', sep='\t', parse_dates=['Date'])[['Date', 'Measure Values']]
    nc.Date = [pandas.Period(str(v), freq='D') for v in nc.Date]
    nc = nc.set_index('Date').sort_index()
    nc['Deaths'] = nc['Measure Values'].fillna(0.0).cumsum()
    return nc.reset_index()[['Date', 'Deaths']].copy()


def load_nv_data():
    uri = download_path('Nevada Dashboard Extract.xlsx')
    nv = pandas.read_excel(uri, sheet_name='Deaths', skiprows=2).iloc[:, [0, 1, 2]].copy()
    nv.columns = ['Date', 'Daily', 'Deaths']
    unknown_deaths = nv[nv.Date == 'Unknown'].Daily.iloc[0]
    nv = nv[nv.Date != 'Unknown'].copy()
    nv.Date = [pandas.Period(v, freq='D') for v in nv.Date]
    nv = nv.sort_values('Date')
    total_deaths = nv.iloc[-1, -1]
    nv.Deaths = nv.Deaths * ((total_deaths + unknown_deaths) / total_deaths)
    nv = nv[['Date', 'Deaths']].copy()
    all_dates = pandas.period_range(start='2020-03-01', end=nv.Date.max(), freq='D')
    nv = nv.set_index('Date').reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    nv.columns = ['Date', 'Deaths']
    return nv


def load_pa_data():
    uri = "https://data.pa.gov/api/views/fbgu-sqgp/rows.csv?accessType=DOWNLOAD&bom=true&format=true"
    df = pandas.read_csv(uri, parse_dates=['Date of Death'])
    df = df[df['County Name'] == 'Pennsylvania']
    df = df[['Date of Death', 'Total Deaths']].copy()
    df.columns = ['Date', 'Deaths']
    df = df.sort_values('Date')
    df.Date = [pandas.Period(str(v), freq='D') for v in df.Date]
    # Deal with fact that data starts at 2020-03-18, which is later than we want
    pre_dates = pandas.period_range('2020-03-01', periods=17, freq='D')
    df2 = pandas.DataFrame(0.0, index=pre_dates, columns=['Deaths']).reset_index()
    df2.columns = ['Date', 'Deaths']
    final = pandas.concat([df2, df]).sort_values('Date')
    final.Deaths = [float(str(x).replace(',', '')) for x in final.Deaths]
    return final


def load_tn_data():
    uri = ("https://www.tn.gov/content/dam/tn/health/documents/cedep/novel-coronavirus"
           "/datasets/Public-Dataset-Daily-Case-Info.XLSX")
    tn = pandas.read_excel(uri, parse_dates=['DATE'])[['DATE', 'TOTAL_DEATHS_BY_DOD']].copy()
    tn.columns = ['Date', 'Deaths']
    tn.Date = [pandas.Period(str(v), freq='D') for v in tn.Date]
    return tn


def load_tx_data():
    uri = "https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyFatalityCountData.xlsx"
    df = pandas.read_excel(uri, skiprows=2)
    df = df[df['County Name'] == 'Total'].copy()
    num_cols = len(df.columns)
    df = df.iloc[0, 1:]
    index = pandas.period_range('2020-03-07', periods=len(df), freq='D')
    df = pandas.DataFrame([float(x) for x in df.values], index=index).reset_index()
    df.columns = ['Date', 'Deaths']
    return df.copy()


def load_va_data():
    uri = ("https://data.virginia.gov/api/views/9d6i-p8gz/rows.csv?accessType=DOWNLOAD")
    va = pandas.read_csv(uri, parse_dates=['Event Date'])[['Event Date', 'Number of Deaths']].copy()
    va.columns = ['Date', 'Daily']
    va = va.groupby('Date').sum().reset_index().sort_values('Date')
    va['Deaths'] = va.Daily.cumsum()
    va = va[['Date', 'Deaths']].copy()
    va.Date = [pandas.Period(v, freq='D') for v in va.Date]
    all_dates = pandas.period_range(start='2020-03-01', end=va.Date.max(), freq='D')
    va = va.set_index('Date').reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    va.columns = ['Date', 'Deaths']
    return va.copy()


def replace_state_data(nyt_stats, st, state_name):
    indices = nyt_stats[nyt_stats.State == state_name].index.copy()
    spork = st.copy()
    spork.index = indices
    nyt_stats.loc[indices, 'Deaths'] = spork.Deaths


def fix_state_data(st, earliest_date, latest_date,  latest_days, avg_days=10, decay=1.1):
    max_date = st.Date.max()
    cutoff_date = max_date - latest_days
    if max_date < latest_date:
        latest_days += (latest_date - max_date).n
        max_date = latest_date
    st = st[(st.Date >= earliest_date) & (st.Date <= cutoff_date)]
    st = st.set_index('Date').sort_index().copy()
    extra_dates = pandas.period_range(end=max_date, periods=latest_days, freq='D')
    dailies = (st.Deaths - st.Deaths.shift())[-avg_days:]
    slope, intercept, r, p, std = scipy.stats.linregress(list(range(avg_days)), dailies.values)
    slope_vals = numpy.linspace(slope, slope/decay, latest_days)
    new_dailies = [(intercept + ((avg_days+i)*slope_vals[i])) for i in range(latest_days)]
    new_deaths = [(sum(new_dailies[:i+1]) + st.Deaths[-1])  for i in range(latest_days)]
    st = pandas.concat([st, pandas.DataFrame(new_deaths, index=extra_dates, columns=['Deaths'])])
    st = st.loc[:latest_date].copy()
    return st

def create_smooth_dates(earliest_date, latest_date):
    sd = str(earliest_date)
    ed = str(latest_date)
    all_dates = pandas.date_range(start=sd, end=ed, freq='D')

    for k, cfg in SMOOTH_CONFIGS.items():
        # Start empty
        dates = pandas.DatetimeIndex([], freq='D')
        
        # Compile date ranges excluding certain days of the week
        for dow in cfg['DaysOfWeek']:
            dates = dates.union(pandas.date_range(start=sd, end=ed, freq=dow))
        
        # Add the holidays (and some surrounding days sometimes)
        holidays = cfg.get('Holidays', [])
        if len(holidays):
            dates = dates.union(pandas.DatetimeIndex(holidays))
        
        # Make sure that there is at least one non-excluded day at the end
        for i in range(1, len(dates)):
            if dates[-i] != all_dates[-i]:
                break
        if i > 1:
            i -= 1
            # print(f"Keeping date(s) {list(dates[-i:])}")
            dates = dates[:-i].copy()

        SMOOTH_DATES[k] = pandas.PeriodIndex([pandas.Period(str(v), freq='D') for v in dates])


def find_smooth_dates(st):
    for k, states in SMOOTH_MAPS.items():
        if st in states:
            return SMOOTH_DATES[k]
    return None


def read_nyt_csv(uri, meta, earliest_date, latest_date):
    stats = pandas.read_csv(uri)
    stats = stats[stats.state.isin(meta.State)][['date', 'state', 'deaths']]
    stats.columns = ['Date', 'State', 'Deaths']
    stats.Date = [pandas.Period(str(v)) for v in stats.Date]
    stats = stats[stats.Date >= earliest_date]
    if latest_date:
        stats = stats[stats.Date <= latest_date]

    stats = stats.set_index(['State', 'Date']).sort_index()
    # Pull in the statistics for states
    stats = stats.join(meta.set_index('State'))

    # Remove territories
    stats = stats[~stats.ST.isin(['AS', 'GU', 'MP', 'PR', 'VI'])]

    return stats.reset_index()


def spread_deaths(stats, state, num_deaths, deaths_date, col='Deaths'):
    st = stats[(stats.Date <= deaths_date)][['Date', col]].copy()
    indices = st.index.copy()
    st = st.set_index('Date')[[col]].copy()
    orig_total = st.loc[deaths_date, col]
    st.loc[deaths_date, col] -= num_deaths
    new_total = st.loc[deaths_date, col]
    st['StatAdj'] = st[col] * (orig_total / new_total)
    st = st.reset_index()
    st.index = indices
    stats.loc[indices, col] = st.StatAdj


def smooth_series(s):
    """
    Iterate through a series, determining if a value remains stale for a number of days,
    then jumps more than one once it changes. When this happens, spread that jump over
    the number of stale days.
    """
    i = 0
    last_i = None
    last_val = None
    run_length = 0
    foo = s.copy()
    while i < len(foo):
        val = foo.iloc[i]
        if pandas.isna(val):
            last_i, last_val = i, None
            run_length = 0
        elif last_val is None:
            last_i, last_val = i, val
            run_length = 1
        elif val == last_val:
            run_length += 1
        # elif (val == (last_val + 1)) or (run_length == 1):
        elif run_length == 1:
            last_i, last_val = i, val
            run_length = 1
        elif val < last_val:
            # This almost certainly means we have moved onto a new state, so reset
            last_i, last_val = i, val
            run_length = 1
        else:
            # print(last_val, val, run_length)
            run_length += 1
            new_vals = numpy.linspace(last_val, val, run_length)
            foo.iloc[last_i:i + 1] = new_vals
            last_i, last_val = i, val
            run_length = 1
        i += 1

    return foo


def calc_mid_weekly_average(s):
    # Calculate daily trailing 7-day averages
    daily = (s - s.shift())
    trailing7 = (s - s.shift(7)) / 7

    # Convert into central 7-day mean, with special handling for last three days
    specs = (
        (7.4, numpy.array([0.9, 1.0, 1.0, 1.0, 1.4, 1.1, 1.0])),
        (7.4, numpy.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.4, 1.2])),
        (7.4, numpy.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6])),
    )
    mid7 = trailing7.shift(-3).copy()
    dailies = daily.iloc[-7:].values
    vals = [((dailies * factors).sum() / divisor) for divisor, factors in specs]
    mid7.iloc[-3:] = vals

    return daily, mid7


def calc_state_stats(state, state_stats, meta, latest_date):
    st = state_stats.groupby('Date').sum().sort_index().copy()

    st['ST'] = state
    st['RawDeaths'] = st.Deaths
    st['RawInc'] = st.Deaths - st.Deaths.shift()

    st = st.reset_index().copy()

    # Correct for various jumps/dips in the reporting of death data
    STATE_DEATH_ADJUSTMENTS = (
        ('AL', -20, '2020-04-23'),
        ('AZ', 45, '2020-05-08'),
        ('AR', 143, '2020-09-15'),
        ('CO', 65, '2020-04-24'),
        ('CO', -29, '2020-04-25'),
        ('DE', 67, '2020-06-23'),
        ('DE', 47, '2020-07-24'),
        ('IA', 140, '2020-12-08'),
        ('IL', 123, '2020-06-08'),
        # ('IN', 11, '2020-07-03'),
        ('LA', 40, '2020-04-14'),
        ('LA', 40, '2020-04-22'),
        ('MD', 68, '2020-04-15'),
        # ('MI', 220, '2020-06-05'),
        # ('MI', 60, '2020-09-09'),
        ('NJ', 1854, '2020-06-25'),
        ('NJ', 75, '2020-07-08'),
        ('NJ', -54, '2020-07-22'),
        ('NJ', -38, '2020-07-29'),
        ('NJ', -25, '2020-08-05'),
        ('NJ', -10, '2020-08-12'),
        ('NJ', -44, '2020-08-26'),
        ('NY', 608, '2020-06-30'),  # most apparently happened at least three weeks earlier
        ('NY', -113, '2020-08-06'),
        ('NY', -11, '2020-09-09'),
        ('NY', -11, '2020-09-19'),
        ('NY', -7, '2020-09-22'),
        ('OH', 80, '2020-04-29'),
        ('SC', 25, '2020-04-29'),
        ('SC', 37, '2020-07-16'),
        # ('TN', 16, '2020-06-12'),
        # ('TX', 636, '2020-07-27'),
        # ('VA', 60, '2020-09-15'),
        ('WA', -12, '2020-06-17'),
        ('WA', 7, '2020-06-18'),
        ('WA', 30, '2020-07-24'),
        ('WA', -11, '2020-08-05'),
        ('WA', -90, '2020-12-10'),
        ('WA', -70, '2020-12-12'),
        ('WA', 15, '2020-12-17'),
        ('WA', -10, '2020-12-18'),
        ('WA', 50, '2020-12-29'),
        ('WI', 8, '2020-06-10'),
    )

    for state_, deaths, deaths_date in STATE_DEATH_ADJUSTMENTS:
        if state_ != state:
            continue
        if pandas.Period(deaths_date) <= latest_date:
            spread_deaths(st, state_, deaths, deaths_date)

    # Correct for various jumps/dips in the positive tests data
    # NOTE: There are *tons* of these. I only put them in when they really mess up the estimation algorithm
    STATE_POS_ADJUSTMENTS = (
        ('MA', -8057, '2020-09-02'),
    )

    for state_, cases, cases_date in STATE_POS_ADJUSTMENTS:
        if state_ != state:
            continue
        if pandas.Period(cases_date) <= latest_date:
            spread_deaths(st, state_, cases, cases_date, col='Pos')

    # Correct for various jumps/dips in the negative tests data
    # NOTE: ibid
    STATE_NEG_ADJUSTMENTS = (
        ('KY', -145000, '2020-11-07'),
        ('OR', 920000, '2020-12-01'),
    )

    for state_, cases, cases_date in STATE_NEG_ADJUSTMENTS:
        if state_ != state:
            continue
        if pandas.Period(cases_date) <= latest_date:
            spread_deaths(st, state_, cases, cases_date, col='Neg')

    # Blank out and forward fill entries for days with wimpy reporting
    dates = find_smooth_dates(state)
    if dates is not None:
        st = st.set_index('Date')
        indices = st.index.isin(dates)
        st.loc[indices, 'Deaths'] = numpy.nan
        st.Deaths = st.Deaths.fillna(method='ffill')
        st = st.reset_index().copy()

    # Smooth series that might not be reported daily in some states
    st['PosSm'] = smooth_series(st.Pos)
    st['NegSm'] = smooth_series(st.Neg)
    st['Tests'] = st.PosSm + st.NegSm
    st.Deaths = smooth_series(st.Deaths)

    for col in list(meta.columns):
        st[col] = meta.loc[state][col]

    # Prep for 7-day smoothing calculations
    st['Confirms'], st['Confirms7'] = calc_mid_weekly_average(st.Pos)
    __, st['Confirms7'] = calc_mid_weekly_average(st.Confirms7.cumsum())
    st['Daily'], st['Deaths7'] = calc_mid_weekly_average(st.Deaths)
    __, st['Deaths7'] = calc_mid_weekly_average(st.Deaths7.cumsum())
    st['DTests'], st['DTests7'] = calc_mid_weekly_average(st.Tests)
    __, st['DTests7'] = calc_mid_weekly_average(st.DTests7.cumsum())

    return st.reset_index().set_index(['ST', 'Date']).copy()


def get_infections_df(states, meta, death_lag, ifr_start, ifr_end, ifr_breaks, incubation, infectious,
                      max_confirmed_ratio=0.7):
    meta = meta.set_index('ST')
    avg_nursing = (meta.Nursing.sum() / meta.Pop.sum())

    ifr_breaks = [] if ifr_breaks is None else ifr_breaks
    new_states = []
    for state in states:
        state = state.copy()

        st = state.reset_index().iloc[0, 0]
        st_meta = meta.loc[st]
        st_nursing = st_meta.Nursing / st_meta.Pop
        nursing_factor = math.sqrt(st_nursing / avg_nursing)
        median_factor = (st_meta.Median / 38.2) ** 2
        # nursing has 51% correlation to PFR; median has -3%, but I still think it counts for something for IFR
        ifr_factor = ((2*nursing_factor) + median_factor) / 3
        # print(f'{st} {nursing_factor=:.2f} {median_factor=:.2f} {ifr_factor=:.2f}')

        # Calculate the IFR to apply for each day
        ifr = _calc_ifr(state, ifr_start, ifr_end, ifr_breaks) * ifr_factor
        # ifr = pandas.Series(numpy.linspace(ifr_high, ifr_low, len(state)), index=state.index)
        # Calculate the infections in the past
        infections = state.shift(-death_lag).Deaths7 / ifr
        
        # Calculate the min infections based on max_confirmed_ratio
        min_infections = state.Confirms7 / max_confirmed_ratio
        infections = infections.combine(min_infections, max, 0)

        # Find out the ratio of infections that were detected on the last date in the past
        last_date = infections.index[-(death_lag+1)]
        last_ratio = infections.loc[last_date] / (state.loc[last_date, 'Confirms7'] + 1)
        last_tests = state.loc[last_date, 'DTests7']
        # print(st, last_tests, state.DTests7.iloc[-death_lag:])

        # Apply that ratio to the dates since that date,
        # also adjusting for number of tests performed
        ntests_factor = 1.0 if st in ['MS', 'ND', 'WA'] else (last_tests / state.DTests7.iloc[-death_lag:])
        infections.iloc[-death_lag:] = (state.Confirms7.iloc[-death_lag:] * last_ratio * ntests_factor)

        state['DPerM'] = state.Deaths7 / state.Pop
        state['NewInf'] = infections
        state['NIPerM'] = state.NewInf / state.Pop
        state['TotInf'] = infections.cumsum()
        state['ActInf'] = infections.rolling(infectious).sum().shift(incubation)
        state['ActKnown'] = state.Confirms7.rolling(infectious).sum()
        state['ActUnk'] = state.ActInf - state.ActKnown
        state['AIPer1000'] = state.ActInf / state.Pop / 1000.
        state['AUPer1000'] = state.ActUnk / state.Pop / 1000.
        state['PctFound'] = state.Confirms7 / (state.NewInf + 1)
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


def download_zipped_df(uri, fname, parse_dates=None):
    response = urllib.request.urlopen(uri)
    zippedData = response.read()
    myzipfile = zipfile.ZipFile(io.BytesIO(zippedData))
    foofile = myzipfile.open(fname)
    return pandas.read_csv(foofile, parse_dates=parse_dates)
