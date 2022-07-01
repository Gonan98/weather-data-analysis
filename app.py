from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dpns = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
dp = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

variable_dict = {
    'FALLECIDOS': 'Numero de Fallecidos',
    'POSITIVOS': 'Numero de Casos Positivos',
    'PM25': 'PM 2.5 μg/m³',
    'TEMPERATURA_MEDIA': 'Temperatura Media (°C)',
    'HUMEDAD_RELATIVA': 'Humedad Relativa (%)',
    'PRECIPITACION': 'Precipitacion (mm)',
    'VELOCIDAD_VIENTO': 'Velocidad del Viento (m/s)',
    'DIRECCION_VIENTO': 'Direccion del Viento (°)',
    'RADIACION_SOLAR_TOTAL': 'Radiacion Solar Total (Wh/m²)'
}

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


def plot_df(df):
    variable_y1 = df.columns[0]
    variable_y2 = df.columns[1]
    ax1 = df[variable_y1].plot(y=variable_dict[variable_y1])
    ax2 = df[variable_y2].plot(y=variable_dict[variable_y2], ax=ax1, secondary_y=True)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel(variable_dict[variable_y1])
    ax2.set_ylabel(variable_dict[variable_y2])
    plt.grid()
    plt.show()


def np_correlacion(df, title=''):

    variable_x = df.columns[1] # Variable Meteorologica
    variable_y = df.columns[0] # Variable Covid

    x = df[variable_x].to_numpy()
    y = df[variable_y].to_numpy()
    r = df[variable_x].corr(df[variable_y])

    plt.plot(x, y, 'o', label='Datos')

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    plt.plot(x, p(x), '-', label='Regresion', color='red')
    plt.title(f'{title}\nR² = {round(r**2, 7)}')
    plt.xlabel(variable_dict[variable_x])
    plt.ylabel(variable_dict[variable_y])
    plt.grid()
    plt.legend()
    plt.show()


def smooth_plot(df):
    variable_covid = df.columns[0]
    new_df = df[variable_covid].rolling(7, min_periods=1).mean()
    new_df.plot()
    plt.xlabel('Fecha')
    plt.ylabel(variable_dict[variable_covid])
    plt.grid()
    plt.show()


def analisis_covid_meteorologico(covid_serie, variable_meteorologica):
    mt_df = load_meteorologico(variable_meteorologica)
    mt_df = mt_df.resample('D').mean()
    mt_df = mt_df[mt_df.index.year == 2020]

    covid_mt_df = pd.merge(covid_serie, mt_df, left_index=True, right_index=True)
    plot_df(covid_mt_df)
    np_correlacion(covid_mt_df.dropna())


def np_correlacion_dias(covid_serie, meteorologico_df, dias=60):
    HORAS = 24
    mt_df = meteorologico_df.copy()
    for i in range(dias):
        mt_df = mt_df.shift(HORAS)
        mt_mean_df = mt_df[(mt_df.index >= '2020-04-01') & (mt_df.index <= '2020-12-31')].resample('D').mean()
        covid_meteorologico_df = pd.merge(covid_serie, mt_mean_df, left_index=True, right_index=True).dropna()
        np_correlacion(covid_meteorologico_df, title=f'Correlacion corriendo {i+1} dia(s) atrás')


def np_correlacion_promedio_dias(covid_serie, meteorologico_df, dias=60):
    HORAS = 24
    mt_df = meteorologico_df.copy()
    for i in range(dias):
        mt_df = mt_df.shift(HORAS)
        mt_mean_df = mt_df[(mt_df.index >= '2020-04-01') & (mt_df.index <= '2020-12-31')].resample('D').mean()
        mt_mean_df = mt_mean_df.rolling(i+2, min_periods=1).mean()
        covid_meteorologico_df = pd.merge(covid_serie, mt_mean_df, left_index=True, right_index=True).dropna()
        np_correlacion(covid_meteorologico_df, title=f'Correlacion corriendo {i+1} dia(s) atrás promediado')



def main():

    # Analisis de fallecidos
    fallecidos_df = load_fallecidos()
    smooth_plot(fallecidos_df)

    # Analisis de positivos
    positivos_df = load_positivos()
    smooth_plot(positivos_df)

    # Analisis de PM25
    pm25_df = load_pm25()
    pm25_mean = pm25_df.resample('D').mean()
    pm25_min = pm25_df.resample('D').min()
    pm25_max = pm25_df.resample('D').max()

    plt.plot(pm25_mean, label='Promedio')
    plt.plot(pm25_min, label='Minimo')
    plt.plot(pm25_max, label='Maximo')
    plt.xlabel('Años')
    plt.ylabel(variable_dict['PM25'])
    plt.grid()
    plt.legend()
    plt.show()

    # Correlaciones entre variables en el 2020
    fallecidos_serie = fallecidos_df[fallecidos_df.index.year == 2020]
    positivos_serie = positivos_df[positivos_df.index.year == 2020]
    pm25_mean_2020 = pm25_mean[pm25_mean.index.year == 2020]
    
    fallecidos_pm25 = pd.merge(fallecidos_serie, pm25_mean_2020, left_index=True, right_index=True)
    positivos_pm25 = pd.merge(positivos_serie, pm25_mean_2020, left_index=True, right_index=True)

    plot_df(fallecidos_pm25)
    plot_df(positivos_pm25)
    
    # Drop 0 values
    fallecidos_pm25 = fallecidos_pm25[fallecidos_pm25['FALLECIDOS'] > 0].dropna()
    positivos_pm25 = positivos_pm25[positivos_pm25['POSITIVOS'] > 0].dropna()

    np_correlacion(fallecidos_pm25)
    np_correlacion(positivos_pm25)

    # Analisis Correlacion 7 Dias Fallecidos vs PM2.5
    np_correlacion_dias(fallecidos_serie, pm25_df)
    np_correlacion_promedio_dias(fallecidos_serie, pm25_df)

    # Analisis Correlacion 7 Dias Positivos vs PM2.5
    np_correlacion_dias(positivos_serie, pm25_df)
    np_correlacion_promedio_dias(positivos_serie, pm25_df)

    # Analisis de temperatura
    analisis_covid_meteorologico(fallecidos_serie, 'TEMPERATURA_MEDIA')
    analisis_covid_meteorologico(positivos_serie, 'TEMPERATURA_MEDIA')

    # Analisis de humedad
    analisis_covid_meteorologico(fallecidos_serie, 'HUMEDAD_RELATIVA')
    analisis_covid_meteorologico(positivos_serie, 'HUMEDAD_RELATIVA')

    # humedad_relativa_df = load_meteorologico('HUMEDAD_RELATIVA')
    # np_correlacion_dias(fallecidos_serie, humedad_relativa_df)


if __name__ == '__main__':
    main()