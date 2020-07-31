import numpy
import pandas


def load_data(earliest_date, latest_date):
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
        nyt_stats = pandas.concat([nyt_stats, nyt_stats_live], sort=True)
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
    if latest_date:
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

    # Correct for various jumps in the data
    STATE_ADJUSTMENTS = (
        ('Colorado', -29, '2020-04-25'),
        ('New Jersey', 1854, '2020-06-25'),
        ('New Jersey', -54, '2020-07-22'),
        ('New York', 608, '2020-06-30'),  # most apparently happened at least three weeks earlier
        ('Illinois', 123, '2020-06-08'),
        ('Michigan', 220, '2020-06-05'),
        ('Delaware', 47, '2020-07-24'),
        ('Texas', 636, '2020-07-27'),
        ('Washington', -7, '2020-06-17'),
        ('Washington', 30, '2020-07-24'),
    )

    for state, deaths, deaths_date in STATE_ADJUSTMENTS:
        spread_deaths(nyt_stats, state, deaths, deaths_date)

    return meta, nyt_stats, ct_stats

    
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


