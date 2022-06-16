from cProfile import label
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dpns = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
dp = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

def load_fallecidos() -> pd.DataFrame:
    df = pd.read_csv(
        'fallecidos_covid.csv',
        delimiter=';',
        parse_dates=['FECHA_FALLECIMIENTO'],
        usecols=['FECHA_FALLECIMIENTO', 'UBIGEO'],
        index_col='FECHA_FALLECIMIENTO',
        low_memory=False)

    df = df[df['UBIGEO'] == '150132'] # Solo SJL
    df = df.groupby(df.index).count()
    df = df.rename(columns={'UBIGEO': 'FALLECIDOS'})
    df = df.reindex(pd.date_range(start='2020-01-01', end='2022-04-30')).fillna(0)
    return df


def load_positivos() -> pd.DataFrame:
    df = pd.read_csv(
        'positivos_covid.csv',
        delimiter=';',
        parse_dates=['FECHA_RESULTADO'],
        usecols=['FECHA_RESULTADO', 'UBIGEO'],
        index_col='FECHA_RESULTADO')

    df = df[df['UBIGEO'] == 150132] # Solo SJL
    df = df.groupby(df.index).count()
    df = df.rename(columns={'UBIGEO': 'POSITIVOS'})
    df = df.reindex(pd.date_range(start='2020-01-01', end='2022-04-30')).fillna(0)
    return df


def load_pm25():
    df = pd.read_csv(
        'pm25.csv',
        delimiter=';',
        parse_dates=['FECHA'],
        date_parser=dpns,
        index_col='FECHA')

    df = df.replace('S/D', np.nan)
    df['PM25'] = df['PM25'].astype(float)
    return df


def load_meteorologico(column):
    df = pd.read_csv(
        'meteorologica.csv',
        delimiter=';',
        parse_dates={ 'FECHA_HORA': ['FECHA', 'HORA'] },
        date_parser=dp,
        usecols=['FECHA', 'HORA', column],
        index_col='FECHA_HORA')

    df = df.replace('S/D', np.nan)
    df[column] = df[column].astype(float)
    return df


def plot_df(df, variable_covid='', variable_meteorologica=''):
    ax1 = df[df.columns[0]].plot(y=df.columns[0])
    ax2 = df[df.columns[1]].plot(y=df.columns[1], ax=ax1, secondary_y=True)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel(variable_covid)
    ax2.set_ylabel(variable_meteorologica)
    plt.grid()
    plt.show()


def np_correlacion(df, column_x, column_y, xlabel='', ylabel='', title=''):
    x = df[column_x].to_numpy()
    y = df[column_y].to_numpy()

    r = df[column_x].corr(df[column_y])

    plt.plot(x, y, 'o', label='Datos')

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    plt.plot(x, p(x), '-', label='Regresion', color='red')
    plt.title(f'{title}\nR² = {round(r**2, 7)}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.show()


def smooth_plot(df, column_name, xlabel='', ylabel=''):
    df[column_name].rolling(7).mean().plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def analisis_covid_meteorologico(covid_serie, variable_meteorologica, variable_covid, xlabel='', ylabel=''):
    mt_format = {
        'TEMPERATURA_MEDIA': 'Temperatura Media (°C)',
        'HUMEDAD_RELATIVA': 'Humedad Relativa (%)',
        'PRECIPITACION': 'Precipitacion (mm)',
        'VELOCIDAD_VIENTO': 'Velocidad del Viento (m/s)',
        'DIRECCION_VIENTO': 'Direccion del Viento (°)',
        'RADIACION_SOLAR_TOTAL': 'Radiacion Solar Total (Wh/m2)'
    }
    mt_df = load_meteorologico(variable_meteorologica)
    mt_df = mt_df.resample('D').mean()
    mt_df = mt_df[mt_df.index.year == 2020]

    covid_mt_df = pd.merge(covid_serie, mt_df, left_index=True, right_index=True)

    y1label = 'Numero de ' + variable_covid.capitalize()
    plot_df(covid_mt_df, y1label, mt_format[variable_meteorologica])

    np_correlacion(covid_mt_df.dropna(), variable_meteorologica, variable_covid, xlabel, ylabel)


def np_correlacion_7_dias(covid_serie, meteorologico_df, column_x, column_y, xlabel='', ylabel=''):
    DIAS = 7
    HORAS = 24
    for i in range(DIAS):
        meteorologico_df = meteorologico_df.shift(-HORAS)
        meteorologico_mean = meteorologico_df[meteorologico_df.index.year == 2020].resample('D').mean()
        covid_meteorologico_df = pd.merge(covid_serie, meteorologico_mean, left_index=True, right_index=True)
        np_correlacion(covid_meteorologico_df.dropna(), column_x, column_y, xlabel, ylabel, title=f'Correlacion corriendo {i+1} dia(s) atrás')


def main():


    # Analisis de fallecidos
    fallecidos_df = load_fallecidos()
    smooth_plot(fallecidos_df, 'FALLECIDOS' , xlabel='Fecha', ylabel='Fallecidos')

    # Analisis de positivos
    positivos_df = load_positivos()
    smooth_plot(positivos_df, 'POSITIVOS' , xlabel='Fecha', ylabel='Positivos')

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
    
    fallecidos_pm25 = pd.merge(fallecidos_serie, pm25_mean_2020, left_index=True, right_index=True)
    positivos_pm25 = pd.merge(positivos_serie, pm25_mean_2020, left_index=True, right_index=True)

    plot_df(fallecidos_pm25, 'Fallecidos', 'PM25')
    plot_df(positivos_pm25, 'Positivos', 'PM25')
    
    # Drop 0 values
    fallecidos_pm25 = fallecidos_pm25[fallecidos_pm25['FALLECIDOS'] > 0].dropna()
    positivos_pm25 = positivos_pm25[positivos_pm25['POSITIVOS'] > 0].dropna()

    np_correlacion(fallecidos_pm25, 'PM25', 'FALLECIDOS', xlabel='PM2.5 (ug/m3)', ylabel='Fallecidos')
    np_correlacion(positivos_pm25, 'PM25', 'POSITIVOS', xlabel='PM2.5 (ug/m3)', ylabel='Positivos')

    # Analisis Correlacion 7 Dias Fallecidos vs PM2.5
    np_correlacion_7_dias(fallecidos_serie, pm25_df, 'PM25', 'FALLECIDOS', xlabel='PM2.5 (ug/m3)', ylabel='Fallecidos')

    # Analisis Correlacion 7 Dias Positivos vs PM2.5
    np_correlacion_7_dias(positivos_serie, pm25_df, 'PM25', 'POSITIVOS', xlabel='PM2.5 (ug/m3)', ylabel='Positivos')
    
    # Analisis de temperatura
    analisis_covid_meteorologico(fallecidos_serie, 'TEMPERATURA_MEDIA', 'FALLECIDOS', xlabel='Temperatura (°C)', ylabel='Fallecidos')
    analisis_covid_meteorologico(positivos_serie, 'TEMPERATURA_MEDIA', 'POSITIVOS', xlabel='Temperatura (°C)', ylabel='Positivos')

    # Analisis de humedad
    analisis_covid_meteorologico(fallecidos_serie, 'HUMEDAD_RELATIVA', 'FALLECIDOS', xlabel='Humedad (%)', ylabel='Fallecidos')
    analisis_covid_meteorologico(positivos_serie, 'HUMEDAD_RELATIVA', 'POSITIVOS', xlabel='Humedad (%)', ylabel='Positivos')

    humedad_relativa_df = load_meteorologico('HUMEDAD_RELATIVA')
    np_correlacion_7_dias(fallecidos_serie, humedad_relativa_df, 'HUMEDAD_RELATIVA', 'FALLECIDOS', xlabel='Humedad (%)', ylabel='Fallecidos')


if __name__ == '__main__':
    main()
    #test()