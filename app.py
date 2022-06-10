from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# def covid_date_parser(date_time):
#     return datetime.strptime(date_time, '%Y%m%d')


def the_date_parser(date_time):
    return datetime.strptime(date_time, '%d/%m/%Y %H:%M')


def load_fallecidos() -> pd.DataFrame:
    df = pd.read_csv('fallecidos_covid.csv', delimiter=';', parse_dates=['FECHA_FALLECIMIENTO'], usecols=['FECHA_FALLECIMIENTO', 'UBIGEO'], index_col='FECHA_FALLECIMIENTO', low_memory=False)
    df = df[df['UBIGEO'] == '150132'] # Solo SJL
    df = df.groupby(df.index).count()
    df = df.rename(columns={'UBIGEO': 'FALLECIDOS'})
    df = df.reindex(pd.date_range(start='2020-01-01', end='2022-04-30')).fillna(0)
    return df


def load_positivos() -> pd.DataFrame:
    df = pd.read_csv('positivos_covid.csv', delimiter=';', parse_dates=['FECHA_RESULTADO'], usecols=['FECHA_RESULTADO', 'UBIGEO'], index_col='FECHA_RESULTADO')
    df = df[df['UBIGEO'] == 150132] # Solo SJL
    df = df.groupby(df.index).count()
    df = df.rename(columns={'UBIGEO': 'POSITIVOS'})
    df = df.reindex(pd.date_range(start='2020-01-01', end='2022-04-30')).fillna(0)
    return df


def load_pm25():
    df = pd.read_csv('pm25.csv', delimiter=';', parse_dates=['FECHA'], date_parser=the_date_parser, index_col='FECHA')
    df = df.replace('S/D', np.nan)
    df['PM25'] = df['PM25'].astype(float)
    return df


def load_meteorologico(column):
    df = pd.read_csv('meteorologico.csv', delimiter=';', parse_dates=['FECHA'], date_parser=the_date_parser, index_col='FECHA')
    df = df.replace('S/D', np.nan)
    df[column] = df[column].astype(float)
    return df


def plot_df(df, xlabel='', ylabel=''):
    df.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def correlacion(x_serie, y_serie, xlabel='', ylabel=''):
    n = len(x_serie)
    x = x_serie.to_numpy()
    y = y_serie.to_numpy()
    sumx = np.sum(x)
    sumy = np.sum(y)
    sumx2 = np.sum(x**2)
    sumy2 = np.sum(y**2)
    sumxy = np.sum(x*y)
    promx = sumx/n
    promy = sumy/n

    m = (n*sumxy - sumx*sumy)/(n*sumx2 - sumx**2)
    b = promy - m*promx

    sigmax = np.sqrt(sumx2/n - promx**2)
    sigmay = np.sqrt(sumy2/n - promy**2)
    sigmaxy = sumxy/n - promx*promy
    R2 = (sigmaxy/(sigmax*sigmay))**2



    plt.plot(x, y, 'o', label='Datos')
    plt.plot(x, m*x + b, '-', label='Regresion', color='red')
    plt.title(f'R2 = {round(R2, 7)}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.show()


def smooth(df, column_name):
    df[column_name] = df[column_name].rolling(7).mean()
    return df


def test():
    pass


def main():

    unidades_medida = {
        'TEMPERATURA_MEDIA': '°C',
        'HUMEDAD_RELATIVA': '%',
        'PRECIPITACION': 'mm',
        'VELOCIDAD_VIENTO': 'm/s',
        'DIRECCION_VIENTO': '°',
        'RADIACION_SOLAR_TOTAL': 'Wh/m2'
    }

    # Analisis de fallecidos
    fallecidos_df = load_fallecidos()
    fallecidos_smooth_df = smooth(load_fallecidos(), 'FALLECIDOS')
    plot_df(fallecidos_smooth_df, 'Fecha', 'Fallecidos')

    # Analisis de positivos
    positivos_df = load_positivos()
    positivos_smooth_df = smooth(load_positivos(), 'POSITIVOS')
    plot_df(positivos_smooth_df, 'Fecha', 'Positivos')

    # Analisis de PM25
    pm25_df = load_pm25()
    pm25_mean = pm25_df.resample('D').mean()
    pm25_min = pm25_df.resample('D').min()
    pm25_max = pm25_df.resample('D').max()


    plt.plot(pm25_mean, label='Promedio')
    plt.plot(pm25_min, label='Minimo')
    plt.plot(pm25_max, label='Maximo')
    plt.grid()
    plt.legend()
    plt.show()

    # Correlaciones entre variables en el 2020
    fallecidos_serie = fallecidos_df[fallecidos_df.index.year == 2020]
    positivos_serie = positivos_df[positivos_df.index.year == 2020]
    pm25_mean_2020 = pm25_mean[pm25_mean.index.year == 2020]
    
    fallecidos_pm25 = pd.merge(fallecidos_serie, pm25_mean_2020, left_index=True, right_index=True).dropna()
    positivos_pm25 = pd.merge(positivos_serie, pm25_mean_2020, left_index=True, right_index=True).dropna()

    correlacion(fallecidos_pm25['FALLECIDOS'], fallecidos_pm25['PM25'], 'PM2.5 (ug/m3)', 'Fallecidos')
    correlacion(positivos_pm25['POSITIVOS'], positivos_pm25['PM25'], 'PM2.5 (ug/m3)', 'Posivos')

    pm25_mean_7dias = pm25_mean.copy()
    for _ in range(7):
        pm25_mean_7dias = pm25_mean_7dias.shift(-1)
        pm25_mean_7dias = pm25_mean_7dias[pm25_mean_7dias.index.year == 2020]
        pm25_mean_7dias = pm25_mean_7dias.resample('D').mean()
        f_pm_2020 = pd.merge(fallecidos_serie, pm25_mean_7dias, left_index=True, right_index=True).dropna()

        correlacion(f_pm_2020['FALLECIDOS'], f_pm_2020['PM25'], 'PM2.5 (ug/m3)', 'Fallecidos')
    
    # Analisis de temperatura




if __name__ == '__main__':
    main()