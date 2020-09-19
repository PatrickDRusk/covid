import datetime

import numpy
import pandas


SMOOTH_CONFIGS = dict(
    Standard=
        dict(
            DaysOfWeek = ('W-SUN', 'W-MON'),
            Holidays = (
                '05-23-2020', '05-26-2020', '05-27-2020',  # Memorial Day
                '07-03-2020', '07-04-2020', # Independence Day
                '09-05-2020', '09-08-2020', '09-09-2020',  # Labor Day
            )
        )
)

SMOOTH_DATES = dict()

SMOOTH_MAPS = dict(
    Standard=('AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
       'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MD', 'ME', 'MI', 'MN',
       'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'OH',
       'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA',
       'WI', 'WV', 'WY')
)


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

    # Improve the Massachusetts data by using the DateOfDeath.csv from MA site
    # Since the latest MA data is very incomplete, replace the most recent three days with
    # the average from the prior five days
    days = 4
    cur_date = pandas.Period(nyt_stats.Date.max(), freq='D')
    cutoff_date = cur_date - days
    ma = pandas.read_csv('DateOfDeath.csv').iloc[:, [0, 2, 4]]
    ma.columns = ['Date', 'Confirmed', 'Probable']
    ma['Deaths'] = ma.Confirmed + ma.Probable
    ma.Date = [pandas.Period(str(v)) for v in ma.Date]
    ma = ma[(ma.Date >= earliest_date) & (ma.Date <= cutoff_date)]
    ma = ma.set_index('Date').sort_index()[['Deaths']]
    extra_dates = pandas.period_range(end=cur_date, periods=days, freq='D')
    avg_deaths = (ma.loc[cutoff_date].Deaths - ma.loc[cutoff_date - 5].Deaths) / 5
    new_deaths = [ma.Deaths[-1] + (avg_deaths * (i + 1)) for i in range(days)]
    ma = pandas.concat([ma, pandas.DataFrame(new_deaths, index=extra_dates, columns=['Deaths'])])

    indices = nyt_stats[nyt_stats.State == 'Massachusetts'].index.copy()
    spork = ma.copy()
    spork.index = indices
    nyt_stats.loc[indices, 'Deaths'] = spork.Deaths
    nyt_stats[nyt_stats.State == 'Massachusetts'].tail()

    # Pull in the testing information from the COVID Tracking Project
    ct_stats = pandas.read_csv('https://covidtracking.com/api/v1/states/daily.csv')

    # Remove territories
    ct_stats = ct_stats[~ct_stats.state.isin(['AS', 'GU', 'MP', 'PR', 'VI'])].copy()
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

    # Blank out and forward fill entries for days with wimpy reporting
    # dates = find_smooth_dates(
    
    
    # Smooth series that might not be reported daily in some states
    ct_stats.Pos = smooth_series(ct_stats.Pos)
    ct_stats.Neg = smooth_series(ct_stats.Neg)
    ct_stats['Tests'] = ct_stats.Pos + ct_stats.Neg
    nyt_stats.Deaths = smooth_series(nyt_stats.Deaths)

    # Correct for various jumps in the data
    STATE_DEATH_ADJUSTMENTS = (
        ('Alabama', -20, '2020-04-23'),
        ('Arizona', 45, '2020-05-08'),
        ('Arkansas', 143, '2020-09-15'),
        ('Colorado', -29, '2020-04-25'),
        ('Delaware', 34, '2020-06-22'),
        ('Delaware', 33, '2020-06-23'),
        ('Delaware', 47, '2020-07-24'),
        ('Illinois', 123, '2020-06-08'),
        ('Indiana', 11, '2020-07-03'),
        ('Maryland', 68, '2020-04-15'),
        ('Michigan', 220, '2020-06-05'),
        ('New Jersey', 1854, '2020-06-25'),
        ('New Jersey', -54, '2020-07-22'),
        ('New Jersey', -38, '2020-07-29'),
        ('New Jersey', -25, '2020-08-05'),
        ('New Jersey', -39, '2020-08-27'),
        ('New York', 608, '2020-06-30'),  # most apparently happened at least three weeks earlier
        ('New York', -113, '2020-08-06'),
        ('New York', -113, '2020-08-06'),
        ('New York', -11, '2020-09-09'),
        ('South Carolina', 37, '2020-07-16'),
        ('Tennessee', 16, '2020-06-12'),
        ('Texas', 636, '2020-07-27'),
        ('Washington', -7, '2020-06-17'),
        ('Washington', 17, '2020-06-18'),
        ('Washington', 30, '2020-07-24'),
        ('Wisconsin', 8, '2020-06-10'),
    )

    for state, deaths, deaths_date in STATE_DEATH_ADJUSTMENTS:
        if pandas.Period(deaths_date) <= latest_date:
            spread_deaths(nyt_stats, state, deaths, deaths_date)

    # Correct for various jumps in the data
    STATE_CASE_ADJUSTMENTS = (
        ('Massachusetts', -8057, '2020-09-02'),
    )

    for state, cases, cases_date in STATE_CASE_ADJUSTMENTS:
        if pandas.Period(cases_date) <= latest_date:
            spread_deaths(ct_stats, state, cases, cases_date, col='Pos')

    return latest_date, meta, nyt_stats, ct_stats


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
            print(f"Keeping date(s) {list(dates[-i:])}")
            dates = dates[:-i].copy()

        SMOOTH_DATES[k] = dates


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
    st = stats[(stats.State == state) & (stats.Date <= deaths_date)][['Date', 'State', col]].copy()
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
        (7.4, numpy.array([0.9, 1.0, 1.0, 1.0, 1.3, 1.1, 1.1])),
        (7.3, numpy.array([0.9, 0.9, 1.0, 1.0, 1.1, 1.3, 1.1])),
        (7.2, numpy.array([0.9, 0.9, 0.9, 1.0, 1.1, 1.1, 1.3])),
    )
    mid7 = trailing7.shift(-3).copy()
    dailies = daily.iloc[-7:].values
    vals = [((dailies * factors).sum() / divisor) for divisor, factors in specs]
    mid7.iloc[-3:] = vals

    return daily, mid7


def calc_state_stats(state, state_stats, meta):
    st = state_stats.groupby('Date').sum().sort_index().copy()

    st['ST'] = state
    for col in list(meta.columns):
        st[col] = meta.loc[state][col]

    # Smooth series that might not be reported daily in some states
    st.Pos = smooth_series(st.Pos)
    st.Deaths = smooth_series(st.Deaths)

    # Prep for 7-day smoothing calculations
    st['Confirms'], st['Confirms7'] = calc_mid_weekly_average(st.Pos)
    st['Daily'], st['Deaths7'] = calc_mid_weekly_average(st.Deaths)
    st['DTests'], st['DTests7'] = calc_mid_weekly_average(st.Tests)

    return st.reset_index().set_index(['ST', 'Date']).copy()


def get_infections_df(states, death_lag, ifr_high, ifr_low, incubation, infectious,
                      max_confirmed_ratio=0.7):
    new_states = []
    for state in states:
        state = state.copy()

        # Calculate the IFR to apply for each day
        ifr = pandas.Series(numpy.linspace(ifr_high, ifr_low, len(state)), index=state.index)
        # Calculate the infections in the past
        infections = state.shift(-death_lag).Deaths7 / ifr
        
        # Calculate the min infections based on max_confirmed_ratio
        min_infections = state.Confirms7 / max_confirmed_ratio
        infections = infections.combine(min_infections, max, 0)

        # Find out the ratio of infections that were detected on the last date in the past
        last_date = infections.index[-(death_lag+1)]
        last_ratio = infections.loc[last_date] / (state.loc[last_date, 'Confirms7'] + 1)

        # Apply that ratio to the dates since that date
        infections.iloc[-death_lag:] = state.Confirms7.iloc[-death_lag:] * last_ratio

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
