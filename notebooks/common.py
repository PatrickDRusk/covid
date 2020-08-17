import datetime

import numpy
import pandas



def load_data(earliest_date, latest_date):
    if not latest_date:
        latest_date = pandas.Period((datetime.datetime.now() - datetime.timedelta(hours=19)).date(),
                                    freq='D')

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

    # Improve the Massachusetts data by using the DateOfDeath.csv from MA site
    # Since the latest MA data is very incomplete, replace the most recent three days with
    # the average from the prior five days
    days = 3
    cur_date = pandas.Period(nyt_stats.Date.max(), freq='D')
    cutoff_date = cur_date - days
    ma = pandas.read_csv('DateOfDeath.csv').iloc[:, [0, 2, 4]]
    ma.columns = ['Date', 'Confirmed', 'Probable']
    ma['Deaths'] = ma.Confirmed + ma.Probable
    ma.Date = [pandas.Period(str(v)) for v in ma.Date]
    ma = ma[(ma.Date >= earliest_date) & (ma.Date <= cutoff_date)]
    ma = ma.set_index('Date').sort_index()[['Deaths']]
    extra_dates = pandas.period_range(end=cur_date, periods=days, freq='D')
    avg_deaths = (ma.loc[cutoff_date].Deaths - ma.loc[cutoff_date-5].Deaths) / 5
    new_deaths = [ma.Deaths[-1] + (avg_deaths * (i+1)) for i in range(days)]
    ma = pandas.concat([ma, pandas.DataFrame(new_deaths, index=extra_dates, columns=['Deaths'])])

    indices = nyt_stats[nyt_stats.State == 'Massachusetts'].index.copy()
    spork = ma.copy()
    spork.index = indices
    nyt_stats.loc[indices, 'Deaths'] = spork.Deaths
    nyt_stats[nyt_stats.State == 'Massachusetts'].tail()

    # Pull in the testing information from the COVID Tracking Project
    ct_stats = pandas.read_csv('https://covidtracking.com/api/v1/states/daily.csv')

    # Remove territories
    ct_stats = ct_stats[~ct_stats.state.isin(['AS', 'GU', 'MP', 'PR', 'VI'])]

    ct_stats.date = [pandas.Period(str(v)) for v in ct_stats.date]

    # Remember Texas deaths
    ct_tx_stats = ct_stats[ct_stats.state == 'TX'][['date', 'death']]

    # Choose and rename a subset of columns
    ct_stats = ct_stats[['date', 'state', 'positive', 'negative']]
    ct_stats.columns = ['Date', 'ST', 'Pos', 'Neg']

    # Set the index to state and date
    ct_stats = ct_stats[ct_stats.Date >= earliest_date]
    ct_stats = ct_stats[ct_stats.Date <= latest_date]
    ct_stats = ct_stats.set_index(['ST', 'Date'])

    # Pull in the statistics for states
    ct_stats = ct_stats.join(meta.set_index('ST')).reset_index().sort_values(['ST', 'Date'])

    # The NY Times treats TX data inappropriately. Use COVID Tracking Project for TX.
    nyt_tx = nyt_stats[nyt_stats.ST == 'TX']
    nyt_range = pandas.period_range(nyt_tx.Date.min(), nyt_tx.Date.max(), freq='D')
    ct_tx = ct_tx_stats.copy()
    ct_tx.columns = ['Date', 'Deaths']
    ct_tx = ct_tx.set_index('Date').sort_index().Deaths.dropna()
    ct_tx = ct_tx.asof(nyt_range).fillna(method='ffill').fillna(0.)
    nyt_stats.loc[nyt_tx.index, 'Deaths'] = ct_tx.values

    # Smooth series that might not be reported daily in some states
    ct_stats.Pos = smooth_series(ct_stats.Pos)
    ct_stats.Neg = smooth_series(ct_stats.Neg)
    ct_stats['Tests'] = ct_stats.Pos + ct_stats.Neg
    nyt_stats.Deaths = smooth_series(nyt_stats.Deaths)

    # Correct for various jumps in the data
    STATE_ADJUSTMENTS = (
        ('Colorado', -29, '2020-04-25'),
        ('New Jersey', 1854, '2020-06-25'),
        ('New Jersey', -54, '2020-07-22'),
        ('New Jersey', -38, '2020-07-29'),
        ('New Jersey', -25, '2020-08-05'),
        ('New York', 125, '2020-04-05'),
        ('New York', 608, '2020-06-30'),  # most apparently happened at least three weeks earlier
        ('New York', -146, '2020-08-06'),
        ('Illinois', 123, '2020-06-08'),
        ('Michigan', 220, '2020-06-05'),
        ('Delaware', 47, '2020-07-24'),
        ('Texas', 636, '2020-07-27'),
        ('Washington', -7, '2020-06-17'),
        ('Washington', 30, '2020-07-24'),
    )

    for state, deaths, deaths_date in STATE_ADJUSTMENTS:
        if pandas.Period(deaths_date) <= latest_date:
            spread_deaths(nyt_stats, state, deaths, deaths_date)

    return latest_date, meta, nyt_stats, ct_stats

    
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


def spread_deaths(stats, state, num_deaths, deaths_date, realloc_end_date=None):
    realloc_end_date = realloc_end_date or deaths_date
    st = stats[(stats.State == state) & (stats.Date <= deaths_date)]
    indices = st.index.copy()
    st = st.set_index('Date')[['Deaths']].copy()
    orig_total = st.loc[deaths_date, 'Deaths']
    st.loc[deaths_date, 'Deaths'] -= num_deaths
    new_total = st.loc[deaths_date, 'Deaths']
    st['Daily'] = st.Deaths - st.shift(1).Deaths
    st['DailyAdj'] = (st.Daily * (orig_total / new_total)) - st.Daily
    st['CumAdj'] = st.DailyAdj.sort_index().cumsum().sort_index()
    st.loc[deaths_date, 'CumAdj'] = 0.
    st = st.reset_index()
    st.index = indices
    stats.loc[indices, 'Deaths'] += st.CumAdj


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
        val = foo[i]
        if pandas.isna(val):
            last_i, last_val = i, None
            run_length = 0
        elif last_val is None:
            last_i, last_val = i, val
            run_length = 1
        elif val == last_val:
            run_length += 1
        #elif (val == (last_val + 1)) or (run_length == 1):
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
            foo[last_i:i+1] = new_vals
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
        (8., numpy.array([1., 1., 1.1, 1.2, 1.4, 1.2, 1.1])),
        (9., numpy.array([1., 1.1, 1.1, 1.2, 1.4, 1.7, 1.5])),
        (10., numpy.array([1., 1.1, 1.2, 1.3, 1.4, 1.8, 2.2])),
    )
    mid7 = trailing7.shift(-3).copy()
    dailies = daily.iloc[-7:].values
    vals = [((dailies * factors).sum() / divisor) for divisor, factors in specs]
    mid7.iloc[-3:] = vals

    return daily, mid7
