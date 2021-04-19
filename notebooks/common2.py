import datetime
import io
import math
import os
import urllib
import zipfile

import numpy
import pandas


NYT_STATES = {'AK', 'HI', 'IL', 'KY', 'MD', 'NE', 'NY', 'OK', 'VT', 'WY'}

DOD_META = [
    ('AL', 7, 28, True),   ('AR', 5, 42, False),  ('AZ', 6, 28, True),   ('CA', 4, 28, False),
    ('CO', 10, 28, False), ('CT', 5, 20, True),   ('DC', 7, 32, False),  ('DE', 4, 28, True),
    ('FL', 7, 24, True),   ('GA', 8, 30, True),   ('IA', 8, 30, True),   ('ID', 4, 28, False), # ('IL', 11, 28, False),
    ('IN', 5, 30, True),   ('KS', 5, 35, True),  # ('KY', 8, 28, False),
    ('LA', 8, 32, False),  ('MA', 11, 7, True),   # ('MD', 10, 32, False),
    ('ME', 4, 32, False),
    ('MI', 10, 21, True),  ('MN', 11, 28, False), ('MO', 5, 35, False),   ('MS', 8, 25, True),  # TBD Put MO back to True
    ('MT', 10, 28, False), ('NC', 6, 28, True),   ('ND', 6, 25, True),   # ('NE', 7, 28, False),
    ('NH', 10, 35, False), ('NJ', 10, 28, True),  ('NM', 5, 28, False),  ('NV', 10, 24, True), # ('NY', 7, 28, False),
    ('OH', 9, 21, True),   # ('OK', 4, 35, False),
    ('OR', 5, 35, False),
    ('PA', 8, 30, True),   ('RI', 0, 20, True),   ('SC', 9, 35, True),   ('SD', 12, 38, True),
    ('TN', 9, 25, True),   ('TX', 4, 28, True),   ('UT', 2, 32, False),  ('VA', 5, 28, True),
    ('WA', 4, 28, False),  ('WI', 4, 28, False),  ('WV', 4, 28, False),  # ('WY', 12, 28, False),
]

DOD_STATES = {v[0] for v in DOD_META if v[3]}

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
        try:
            df = pandas.read_pickle(download_path(f'{fname}.pickle'))
        except FileNotFoundError:
            df = func(uri, meta, all_dates)
            df.to_pickle(download_path(f'{fname}.pickle'))
        return df

    # Pull in all the state-supplied date-of-death data
    dod_stats = handle_stats('dod_stats', None, load_date_of_death_states)

    # Pull in state date-of-death data from CDC and reduce it to interesting columns
    # uri = "https://data.cdc.gov/api/views/r8kw-7aab/rows.csv?accessType=DOWNLOAD"
    uri = download_path("cdc_dod_data.csv")
    cdc_stats = handle_stats('cdc_stats', uri, load_cdc_dod_data)

    # Pull in the date-of-report death data from the NYT
    nyt_stats = handle_stats('nyt_stats', None, load_nyt_data)

    # Pull in the hospital information from the CDC
    uri = download_path("cdc_hospitalization_data.csv")
    hosp_stats = handle_stats('hosp_stats', uri, load_hospital_stats)

    all_stats_pre_hosp = pandas.concat([cdc_stats, dod_stats, nyt_stats], axis=0).sort_index()
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

    return latest_date, meta, final_stats


def project_dod_deaths(stats):
    st_dfs = dict()
    for st, st_df in stats.reset_index().groupby('ST'):
        st_dfs[st] = st_df.set_index(['ST', 'Date'])

    for st, hosp_shift, ignore_days, state_provided in DOD_META:
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


def load_cdc_dod_data(uri, meta, all_dates):
    df = pandas.read_csv(uri, parse_dates=['Week Ending Date'])
    df = df[(df.Group == 'By Week') & ~df.State.isin(['United States', 'Puerto Rico'])]
    df = df[['Week Ending Date', 'State', 'COVID-19 Deaths']].copy()
    df.columns = ['Date', 'State', 'Weekly']

    df.Date = [pandas.Period(str(v), freq='D') for v in df.Date]

    # Change from 'State' to 'ST'
    df = df.set_index('State')
    df = df.join(meta.set_index('State')[['ST']])
    df = df[(~df.ST.isin(DOD_STATES | NYT_STATES)) & (df.Date <= all_dates[-1])]
    df = df.reset_index()
    df = df[['ST', 'Date', 'Weekly']].sort_values(['ST', 'Date'])

    st_dfs = list()
    for st, st_df in df.groupby('ST'):
        st_df = st_df.set_index('Date').sort_index()
        st_df['Daily'] = st_df.Weekly / 7.
        st_df.Daily = st_df.Daily.fillna(0.)
        st_df = st_df.reindex(all_dates, method='bfill')
        st_df.ST = st
        st_df.index.name = 'Date'
        st_df.Daily = st_df.Daily.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df['Deaths'] = st_df.Daily.cumsum()
        st_df = st_df.reset_index().set_index(['ST', 'Date'])[['Daily', 'Deaths']].copy()

        st_dfs.append(st_df)

    return pandas.concat(st_dfs).sort_index()


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
    df['NewConf'] = df.NewAConf + df.NewPConf
    df['NewSusp'] = df.NewASusp + df.NewPSusp
    df['New'] = df.NewConf + df.NewSusp
    df['NewHosp'] = df.NewConf + (df.NewSusp * 0.2)
    df['CurrHosp'] = df.TotA + df.TotP

    st_dfs = list()
    for st, st_df in df.groupby('ST'):
        st_df = st_df.set_index('Date').sort_index().copy()
        st_df.NewConf = st_df.NewConf.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df.NewSusp = st_df.NewSusp.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df.New = st_df.New.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df.NewHosp = st_df.NewHosp.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        # st_df.NewHosp = st_df.NewHosp.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_df.CurrHosp = st_df.CurrHosp.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        st_dfs.append(st_df.reset_index().set_index(['ST', 'Date'])[['NewHosp', 'CurrHosp']])

    return pandas.concat(st_dfs).sort_index()


# noinspection PyUnusedLocal
def load_date_of_death_states(uri, meta, all_dates):
    st_dfs = list()

    for st, hosp_shift, ignore_days, state_provided in DOD_META:
        if not state_provided:
            continue
        print(st)
        deaths = eval(f'load_{st.lower()}_data')().set_index('Date')
        deaths.columns = ['RawDeaths']
        deaths = deaths.reindex(all_dates, method='ffill').fillna(0.)
        deaths.index.name = 'Date'
        deaths['ST'] = st
        deaths['RawInc'] = (deaths.RawDeaths - deaths.RawDeaths.shift())
        deaths['Daily'] = deaths.RawInc.rolling(window=13, center=True, win_type='triang', min_periods=1).mean()
        deaths['Deaths'] = deaths.Daily.fillna(0).cumsum()

        st_dfs.append(deaths.reset_index().set_index(['ST', 'Date'])[['Daily', 'Deaths']])

    return pandas.concat(st_dfs).sort_index()


def load_al_data():
    uri = './DateOfDeath.xlsx'
    al = pandas.read_excel(uri, sheet_name='Alabama', parse_dates=['Date'])
    al = al[['Date', 'Deaths']].copy()
    al.Date = [pandas.Period(d.date(), freq='D') for d in al.Date]
    return al


def load_az_data():
    uri = './DateOfDeath.xlsx'
    az = pandas.read_excel(uri, sheet_name='Arizona', parse_dates=['Date'])
    az = az[['Date', 'Deaths']].copy()
    az.Date = [pandas.Period(d.date(), freq='D') for d in az.Date]
    return az


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


def load_de_data():
    def make_date(r):
        dt = datetime.date(int(r['Year']), int(r['Month']), int(r['Day']))
        return pandas.Period(dt, freq='D')

    uri = ("https://myhealthycommunity.dhss.delaware.gov/locations/state/download_covid_19_data/overview")
    raw = pandas.read_csv(uri)

    df = raw[(raw['Date used'] == 'Date of death') & (raw.Unit == 'people') & (raw.Statistic == 'Deaths')]
    df = df[['Year', 'Month', 'Day', 'Value']].copy()
    df['Date'] = df.apply(make_date, axis=1)
    df = df[['Date', 'Value']]
    df.columns = ['Date', 'Deaths']
    return df


def load_fl_data():
    """
    Algorithm from https://github.com/mbevand/florida-covid19-deaths-by-day
    """
    import requests

    url = ("https://services1.arcgis.com/CY1LXxl9zlJeBuRZ/ArcGIS/rest/services/"
           "Florida_COVID_19_Deaths_by_Day/FeatureServer/0/query?"
           "where=ObjectId>0&objectIds=&time=&resultType=standard&outFields=*&returnIdsOnly=false"
           "&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false"
           "&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset="
           "&resultRecordCount=&sqlFormat=none&f=pjson&token=")
    # User-Agent spoofing is required, if we use the default ("python-requests/x.x.x") the server returns
    # an empty 'features'!
    rows = requests.get(url, headers={'User-Agent': 'curl/7.64.0'}, verify=False).json()['features']
    fl = pandas.DataFrame([row['attributes'] for row in rows])
    fl['Date'] = [pandas.Period(datetime.datetime.fromtimestamp(d // 1000), freq='D') for d in fl.Date1]
    fl = fl[['Date', 'Deaths']].copy()
    all_dates = pandas.period_range(start='2020-03-01', end=fl.Date.max(), freq='D')
    fl = fl.set_index('Date').sort_index().reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    fl.columns = ['Date', 'Deaths']
    fl.Deaths = fl.Deaths.cumsum()
    return fl


def load_ga_data():
    uri = "https://ga-covid19.ondemand.sas.com/docs/ga_covid_data.zip"
    df = download_zipped_df(uri, 'epicurve_symptom_date.csv', parse_dates=['symptom date'])
    df = df[df.measure == 'state_total'][['symptom date', 'death_cum']].copy()
    df.columns = ['Date', 'Deaths']
    df.Date = [pandas.Period(str(v), freq='D') for v in df.Date]
    return df


def load_ks_data():
    uri = './DateOfDeath.xlsx'
    ks = pandas.read_excel(uri, sheet_name='Kansas', parse_dates=['Date'])
    ks = ks[['Date', 'Deaths']].copy()
    ks.Date = [pandas.Period(d.date(), freq='D') for d in ks.Date]
    return ks


def load_ia_data():
    uri = './DateOfDeath.xlsx'
    ia = pandas.read_excel(uri, sheet_name='Iowa', parse_dates=['Date'])
    ia = ia[['Date', 'Deaths']].copy()
    ia.Date = [pandas.Period(d.date(), freq='D') for d in ia.Date]
    return ia


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
    uri = (download_path('MI_Date_of_Death.xlsx'))
    mi = pandas.read_excel(uri, parse_dates=['Date'])
    mi = mi.groupby('Date').sum()[['Deaths.Cumulative']].reset_index()
    mi.columns = ['Date', 'Deaths']
    mi.Date = [pandas.Period(str(v), freq='D') for v in mi.Date]
    return mi


def load_mo_data():
    uri = ("https://results.mo.gov/t/COVID19/views/COVID-19DataforDownload/MetricsbyDateofDeath.csv")
    mo = pandas.read_csv(uri).iloc[1:-1, :]
    col = 'Measure Values' if 'Measure Values' in mo.columns else 'Confirmed Deaths'
    # print(mo.columns)
    mo = mo[['Dod', 'Measure Values']].copy()
    mo.columns = ['Date', 'Deaths']
    mo.Date = [pandas.Period(str(v), freq='D') for v in mo.Date]
    mo = mo[mo.Date >= pandas.Period('2020-01-01', freq='D')].set_index('Date').sort_index()
    mo.Deaths = [int(x) for x in mo.Deaths]
    mo.Deaths = mo.Deaths.cumsum()
    all_dates = pandas.period_range(start='2020-01-01', end=mo.index[-1], freq='D')
    mo = mo.reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    mo.columns = ['Date', 'Deaths']
    return mo


def load_ms_data():
    uri = './DateOfDeath.xlsx'
    ms = pandas.read_excel(uri, sheet_name='Mississippi', parse_dates=['Date'])
    ms = ms[['Date', 'Deaths']].copy()
    ms.Date = [pandas.Period(d.date(), freq='D') for d in ms.Date]
    return ms


def load_nc_data():
    uri = download_path('TABLE_DAILY_CASE&DEATHS_METRICS_data.csv')
    # nc = pandas.read_csv(uri, encoding='utf_16', sep='\t', parse_dates=['Date'])[['Date', 'Measure Values']]
    nc = pandas.read_csv(uri, parse_dates=['Date'])[['Date', 'Measure Values']]
    nc.Date = [pandas.Period(str(v), freq='D') for v in nc.Date]
    nc = nc.set_index('Date').sort_index()
    nc['Deaths'] = nc['Measure Values'].fillna(0.0).cumsum()
    return nc.reset_index()[['Date', 'Deaths']].copy()


def load_nd_data():
    uri = "https://www.health.nd.gov/sites/www/files/documents/Files/MSS/coronavirus/charts-data/PublicUseData.csv"
    nd = pandas.read_csv(uri, parse_dates=['Date'])[['Date', 'Total Deaths']]
    nd = nd.groupby('Date').sum().sort_index().cumsum().reset_index()
    nd.columns = ['Date', 'Deaths']
    nd.Date = [pandas.Period(str(v), freq='D') for v in nd.Date]
    return nd


def load_nj_data():
    uri = './DateOfDeath.xlsx'
    nj = pandas.read_excel(uri, sheet_name='New Jersey', parse_dates=['Date'])
    nj = nj[['Date', 'Deaths']].copy()
    nj.Date = [pandas.Period(d.date(), freq='D') for d in nj.Date]
    return nj


def load_nv_data():
    uri = download_path('Nevada Dashboard Extract.xlsx')
    nv = pandas.read_excel(uri, sheet_name='Deaths').iloc[:, [0, 1, 2]].copy()
    nv.columns = ['Date', 'Daily', 'Deaths']
    nv.Date = [pandas.Period(v, freq='D') for v in nv.Date]
    nv = nv.sort_values('Date')
    nv = nv[['Date', 'Deaths']].copy()
    all_dates = pandas.period_range(start='2020-03-01', end=nv.Date.max(), freq='D')
    nv = nv.set_index('Date').reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    nv.columns = ['Date', 'Deaths']
    return nv


def load_nv_data_old():
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


def load_oh_data():
    uri = "https://coronavirus.ohio.gov/static/dashboards/COVIDDeathData_CountyOfDeath.csv"
    oh = pandas.read_csv(uri, low_memory=False)
    oh = oh[['Date Of Death', 'Death Due To Illness Count - County Of Death']].copy()
    oh.columns = ['Date', 'Daily']
    oh.Daily = [int(d) for d in oh.Daily]
    oh = oh[(oh.Daily > 0) & (oh.Daily < 100)].copy()
    oh.Date = [pandas.Period(x, freq='D') for x in oh.Date]
    oh = oh.groupby('Date').sum()[['Daily']].sort_index()
    oh['Deaths'] = oh.Daily.cumsum()
    return oh.reset_index()[['Date', 'Deaths']].copy()


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


def load_ri_data():
    uri = download_path('COVID-19 Rhode Island Data.xlsx')
    ri = pandas.read_excel(uri, sheet_name='Trends', parse_dates=['Date'])[['Date', 'Date of death']]
    ri.Date = [pandas.Period(str(v), freq='D') for v in ri.Date]
    ri = ri.sort_values('Date')
    ri.columns = ['Date', 'Daily']
    ri['Deaths'] = ri.Daily.cumsum()
    ri = ri[['Date', 'Deaths']].set_index('Date')
    all_dates = pandas.period_range(start='2020-03-01', end=ri.index.max(), freq='D')
    ri = ri.reindex(all_dates, method='ffill').fillna(0.0).reset_index()
    ri.columns = ['Date', 'Deaths']
    return ri


def load_sc_data():
    uri = './DateOfDeath.xlsx'
    sc = pandas.read_excel(uri, sheet_name='South Carolina', parse_dates=['Date'])
    sc = sc[['Date', 'Deaths']].copy()
    sc.Date = [pandas.Period(d.date(), freq='D') for d in sc.Date]
    return sc


def load_sd_data():
    uri = './DateOfDeath.xlsx'
    sd = pandas.read_excel(uri, sheet_name='South Dakota', parse_dates=['Date'])
    sd = sd[['Date', 'Deaths']].copy()
    sd.Date = [pandas.Period(d.date(), freq='D') for d in sd.Date]
    return sd


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
    # df = df[df['County Name'] == 'Total'].copy()
    df = df.iloc[-1:, :].copy()
    num_cols = len(df.columns)
    df = df.iloc[0, 1:]
    index = pandas.period_range('2020-03-07', periods=len(df), freq='D')
    df = pandas.DataFrame([float(x) for x in df.values], index=index).reset_index()
    df.columns = ['Date', 'Deaths']
    df.Deaths = df.Deaths.fillna(method='ffill').fillna(0.)
    return df.copy()


def load_va_data():
    uri = ("https://data.virginia.gov/api/views/9d6i-p8gz/rows.csv?accessType=DOWNLOAD")
    va = pandas.read_csv(uri, parse_dates=['Event Date'])[['Event Date', 'Number of Deaths']].copy()
    va.columns = ['Date', 'Daily']
    va = va.groupby('Date').sum().sort_index()
    va = va.reset_index()
    va['Deaths'] = va.Daily.cumsum()
    va = va[['Date', 'Deaths']].copy()
    va.Date = [pandas.Period(v, freq='D') for v in va.Date]
    all_dates = pandas.period_range(start='2020-03-01', end=va.Date.max(), freq='D')
    va = va.set_index('Date').reindex(all_dates, method='ffill').fillna(0.0)
    va = va.reset_index()
    va.columns = ['Date', 'Deaths']
    return va.copy()


def download_zipped_df(uri, fname, parse_dates=None):
    response = urllib.request.urlopen(uri)
    zippedData = response.read()
    myzipfile = zipfile.ZipFile(io.BytesIO(zippedData))
    foofile = myzipfile.open(fname)
    return pandas.read_csv(foofile, parse_dates=parse_dates)


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
    NewYork=
        dict(
            DaysOfWeek = (),
            Holidays = (
                '04-30-2020', '05-01-2020', '05-02-2020',
                '05-03-2020', '05-04-2020', '05-05-2020',
                '05-23-2020', '05-24-2020', '05-25-2020',  # Memorial Day
                '2020-11-26', '2020-11-27', '2020-11-28', '2020-11-29', '2020-11-30', '2020-12-01', # Thanksgiving
                '2021-03-21', '2021-03-22', '2021-03-23',  # data processing issues in NYC
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
    SatSunMon=('CA', 'CO', 'IL', 'LA', 'MT', 'NM', ),
    SunMon=('AR', 'HI', 'KY', 'MD', 'MN', 'NE', 'NH', 'OK', 'OR', 'WA', 'WI',),
    NewYork=('NY', ),
    Wyoming=('WY', ),
)

# This will hold the series of dates per state that need smoothing
SMOOTH_DATES = dict()


# noinspection PyUnusedLocal
def load_nyt_data(uri, meta, all_dates):
    earliest_date, latest_date = all_dates[0], all_dates[-1]

    create_smooth_dates(earliest_date, latest_date)

    # Pull in state data from NY Times and reduce it to interesting columns,
    # joined with the metadata above
    uri = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
    uri_live = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-states.csv'

    return read_nyt_csv(uri, uri_live, meta, earliest_date, latest_date)


def read_nyt_csv(uri, uri_live, meta, earliest_date, latest_date):
    stats = pandas.read_csv(uri)[['date', 'state', 'deaths']]
    stats_live = pandas.read_csv(uri_live)[['date', 'state', 'deaths']]

    # Attach the live stats if the daily file has not yet rolled
    if len(stats_live) and (stats.date.max() < stats_live.date.max()):
        print("Pulling in live stats")
        stats = pandas.concat([stats, stats_live], sort=False)
        stats = stats.sort_values(['state', 'date'])
        stats.index = list(range(len(stats)))

    stats = stats[stats.state.isin(meta.State)][['date', 'state', 'deaths']]
    stats.columns = ['Date', 'State', 'Deaths']

    stats = stats.set_index(['State', 'Date']).sort_index()
    # Pull in the statistics for states
    stats = stats.join(meta.set_index('State')[['ST']])

    # Reduce to only those states allotted to NYT
    stats = stats[stats.ST.isin(NYT_STATES)]
    stats = stats.reset_index()[['ST', 'Date', 'Deaths']].copy()
    stats.Date = [pandas.Period(str(v)) for v in stats.Date]

    st_dfs = []
    for st, st_df in stats.groupby('ST'):
        df = calc_state_stats(st, st_df, meta)
        st_dfs.append(df)
    stats = pandas.concat(st_dfs)

    stats = stats[['Daily7', 'Deaths7']].dropna().copy()
    stats.columns = ['Daily', 'Deaths']

    return stats


def calc_state_stats(state, state_stats, meta):
    st = state_stats.set_index('Date').sort_index().copy()

    st['RawDeaths'] = st.Deaths
    st['RawInc'] = st.Deaths - st.Deaths.shift()

    st = st.reset_index().copy()

    # Correct for various jumps/dips in the reporting of death data
    STATE_DEATH_ADJUSTMENTS = (
        ('AR', 143, '2020-09-15'),
        ('CA', 750, '2021-02-24'),
        ('CO', 65, '2020-04-24'),
        ('CO', -29, '2020-04-25'),
        ('HI', 56, '2021-01-26'),
        ('IL', 123, '2020-06-08'),
        ('KY', 300, '2021-03-18'),
        ('KY', 200, '2021-03-19'),
        ('KY', 40, '2021-03-25'),
        ('LA', 40, '2020-04-14'),
        ('LA', 40, '2020-04-22'),
        ('MD', 68, '2020-04-15'),
        ('MT', 40, '2021-02-03'),
        ('NY', 608, '2020-06-30'),  # most apparently happened at least three weeks earlier
        ('NY', -113, '2020-08-06'),
        ('NY', -11, '2020-09-09'),
        ('NY', -11, '2020-09-19'),
        ('NY', -7, '2020-09-22'),
        ('OK', 20, '2021-02-27'),
        ('OK', 25, '2021-02-28'),
        ('OK', 25, '2021-03-01'),
        ('OK', 1640, '2021-04-07'),
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
        if pandas.Period(deaths_date) <= st.Date.max():
            spread_deaths(st, state_, deaths, deaths_date)

    # Blank out and forward fill entries for days with wimpy reporting
    dates = find_smooth_dates(state)
    if dates is not None:
        st = st.set_index('Date')
        indices = st.index.isin(dates)
        st.loc[indices, 'Deaths'] = numpy.nan
        st.Deaths = st.Deaths.fillna(method='ffill')
        st = st.reset_index().copy()

    # Smooth series that might not be reported daily in some states
    st.Deaths = smooth_series(st.Deaths)

    st['Daily'], st['Daily7'] = calc_mid_weekly_average(st.Deaths)
    __, st['Daily7'] = calc_mid_weekly_average(st.Daily7.cumsum())
    st['Deaths7'] = st.Daily7.cumsum()

    return st.reset_index().set_index(['ST', 'Date']).copy()


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
        (7.6, numpy.array([0.8, 0.9, 1.0, 1.2, 1.5, 1.2, 1.0])),
        (7.5, numpy.array([0.7, 0.8, 0.9, 1.0, 1.2, 1.7, 1.2])),
        (7.1, numpy.array([0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.7])),
    )
    mid7 = trailing7.shift(-3).copy()
    dailies = daily.iloc[-7:].values
    vals = [((dailies * factors).sum() / divisor) for divisor, factors in specs]
    mid7.iloc[-3:] = vals

    return daily, mid7


# noinspection DuplicatedCode
def get_infections_df(states, meta, death_lag, ifr_start, ifr_end, ifr_breaks, incubation, infectious,
                      max_confirmed_ratio=0.7):
    meta = meta.set_index('ST')
    avg_nursing = (meta.Nursing.sum() / meta.Pop.sum())

    ifr_breaks = [] if ifr_breaks is None else ifr_breaks
    new_states = []
    for st, state in states.reset_index().groupby('ST'):
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


if __name__ == '__main__':
    def __main():
        pandas.set_option('display.max_columns', 2000)
        pandas.set_option('display.max_rows', 2000)
        pandas.set_option('display.width', 2000)

        # Earliest date that there is sufficient data for all states, including MA
        EARLIEST_DATE = pandas.Period('2020-01-01', freq='D')

        # Set a latest date when the most recent days have garbage (like on or after holidays)
        LATEST_DATE = pandas.Period('2021-04-03', freq='D')

        latest_date, meta, all_stats = load_data(EARLIEST_DATE, LATEST_DATE)

        # print(all_stats)
        # print(all_stats.reset_index().groupby('Date').sum().Deaths)
        # print(all_stats.reset_index().groupby('State').sum().Daily)

        # Median number of days between being exposed and developing illness
        INCUBATION = 4

        # Number of days one is infectious (this isn't actually used yet)
        INFECTIOUS = 10

        # Median days in between exposure and death
        DEATH_LAG = 19

        # Here is where you set variables for IFR assumptions

        # Note that this IFR represents a country-wide average on any given day, but the IFRs
        # are actually adjusted up/down based on median age and nursing home residents per capita

        # This set represents my worst case scenario (in my 95% CI interval)
        # Start by setting the inital and final IFRs

        # This is my expected scenario
        IFR_S, IFR_E = 0.01, 0.003
        IFR_BREAKS = [['2020-04-30', 0.0085], ['2020-07-31', 0.005], ['2020-09-15', 0.004], ['2021-01-15', 0.004]]

        IFR_S_S, IFR_E_S = f'{100 * IFR_S:.1f}%', f'{100 * IFR_E:.2f}%'
        infected_states = get_infections_df(all_stats, meta, DEATH_LAG, IFR_S, IFR_E, IFR_BREAKS, INCUBATION, INFECTIOUS)
        EST_LINE = str(latest_date - (DEATH_LAG - 1))
        print(f"Total infected by {latest_date}: {int(infected_states.NewInf.sum()):,}")


    __main()
