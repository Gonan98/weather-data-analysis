import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_fallecidos():
    df = pd.read_csv('fallecidos_covid.csv', delimiter=';', parse_dates=['FECHA_FALLECIMIENTO'], usecols=['FECHA_FALLECIMIENTO', 'UBIGEO'], index_col='FECHA_FALLECIMIENTO', low_memory=False)
    df = df[df['UBIGEO'] == '150132'] # Solo SJL
    df = df.groupby(df.index).count()
    df = df.rename(columns={'UBIGEO': 'FALLECIDOS'})
    df = df.reindex(pd.date_range(start='2020-01-01', end='2021-12-31')).fillna(0)
    return df


def load_positivos():
    df = pd.read_csv('positivos_covid.csv', delimiter=';', parse_dates=['FECHA_RESULTADO'], usecols=['FECHA_RESULTADO', 'UBIGEO'], index_col='FECHA_RESULTADO')
    df = df[df['UBIGEO'] == 150132] # Solo SJL
    df = df.groupby(df.index).count()
    df = df.rename(columns={'UBIGEO': 'POSITIVOS'})
    df = df.reindex(pd.date_range(start='2020-01-01', end='2021-12-31')).fillna(0)
    return df


def load_pm25():
    df = pd.read_csv('pm25.csv', delimiter=';', parse_dates=['FECHA'], index_col='FECHA')
    df = df[(df.index >= '2015-01-01') & (df.index < '2021-01-01')]
    df = df.replace('S/D', np.nan)
    df['PM25'] = df['PM25'].astype(float)
    return df


def plot_df(df, xlabel='', ylabel=''):
    df.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def smooth(df):
    pass


def test():
    pass


def main():
    # fallecidos_df = load_fallecidos()
    # fallecidos_df.plot()
    # plt.grid()
    # plt.show()

    # positivos_df = load_positivos()
    # positivos_df.plot()
    # plt.grid()
    # plt.show()

    pm25_df = load_pm25()
    pm25_mean_series = pm25_df.groupby(pm25_df.index.date).mean()['PM25']
    pm25_min_series = pm25_df.groupby(pm25_df.index.date).min()['PM25']
    pm25_max_series = pm25_df.groupby(pm25_df.index.date).max()['PM25']

    pm25_df = pd.DataFrame({'Promedio': pm25_mean_series, 'Minimo': pm25_min_series, 'Maximo': pm25_max_series})
    plot_df(pm25_df, 'Fecha', 'PM2.5 (ug/m3)')
    # plt.grid()
    # plt.show()

if __name__ == '__main__':
    main()