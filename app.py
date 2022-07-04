from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dpns = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
dp = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

variable_dict = {
        'FALLECIDOS': 'Fallecidos',
        'POSITIVOS': 'Casos Positivos',
        'PM25': 'PM 2.5 μg/m³',
        'TEMPERATURA_MEDIA': 'Temperatura Media (°C)',
        'HUMEDAD_RELATIVA': 'Humedad Relativa (%)',
        'PRECIPITACION': 'Precipitacion (mm)',
        'VELOCIDAD_VIENTO': 'Velocidad del Viento (m/s)',
        'DIRECCION_VIENTO': 'Direccion del Viento (°)',
        'RADIACION_SOLAR_TOTAL': 'Radiacion Solar Total (Wh/m²)'
    }

legend_dict = {
        'FALLECIDOS': 'Fallecidos',
        'POSITIVOS': 'Positivos',
        'PM25': 'PM 2.5',
        'TEMPERATURA_MEDIA': 'Temperatura Media',
        'HUMEDAD_RELATIVA': 'Humedad Relativa',
        'PRECIPITACION': 'Precipitacion',
        'VELOCIDAD_VIENTO': 'Velocidad del Viento',
        'DIRECCION_VIENTO': 'Direccion del Viento',
        'RADIACION_SOLAR_TOTAL': 'Radiacion Solar Total'
    }

def load_fallecidos():
    df = pd.read_csv(
        'fallecidos_covid.csv',
        delimiter=';',
        parse_dates=['FECHA_FALLECIMIENTO'],
        usecols=['FECHA_FALLECIMIENTO', 'UBIGEO'],
        index_col='FECHA_FALLECIMIENTO',
        low_memory=False)

    df = df[df['UBIGEO'] == '150132'] # Solo SJL
    serie = df.index.value_counts().sort_index()
    serie = serie.rename('FALLECIDOS')
    serie = serie.reindex(pd.date_range(start='2020-01-01', end='2022-05-31')).fillna(0)
    return serie


def load_positivos():
    df = pd.read_csv(
        'positivos_covid.csv',
        delimiter=';',
        parse_dates=['FECHA_RESULTADO'],
        usecols=['FECHA_RESULTADO', 'UBIGEO'],
        index_col='FECHA_RESULTADO')

    df = df[df['UBIGEO'] == 150132] # Solo SJL
    serie = df.index.value_counts().sort_index()
    serie = serie.rename('POSITIVOS')
    serie = serie.reindex(pd.date_range(start='2020-01-01', end='2022-05-31')).fillna(0)
    return serie


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
    var_y1 = df.columns[0]
    var_y2 = df.columns[1]
    ax1 = df[var_y1].plot(y=variable_dict[var_y1])
    ax2 = df[var_y2].plot(y=variable_dict[var_y2], ax=ax1, secondary_y=True)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel(variable_dict[var_y1])
    ax2.set_ylabel(variable_dict[var_y2])
    ax1.legend([legend_dict[var_y1]], loc='upper left')
    ax2.legend([legend_dict[var_y2]], loc='upper right')
    plt.grid()
    plt.show()


def np_correlacion(df, title=''):

    var_x = df.columns[1] # Variable Meteorologica
    var_y = df.columns[0] # Variable Covid

    x = df[var_x].to_numpy()
    y = df[var_y].to_numpy()
    r = df[var_x].corr(df[var_y])

    plt.plot(x, y, 'o', label='Datos')

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    plt.plot(x, p(x), '-', label='Regresion', color='red')
    plt.title(f'{title}\nR² = {round(r**2, 7)}')
    plt.xlabel(variable_dict[var_x])
    plt.ylabel(variable_dict[var_y])
    plt.grid()
    plt.legend()
    plt.show()


def smooth_plot(serie):
    serie.rolling(7, min_periods=1).mean().plot()
    plt.xlabel('Fecha')
    plt.ylabel(variable_dict[serie.name])
    plt.grid()
    plt.show()


def analisis_covid_meteorologico(covid_serie, variable_meteorologica):
    mt_df = load_meteorologico(variable_meteorologica)
    mt_df = mt_df.resample('D').mean()
    mt_df = mt_df[mt_df.index.year == 2020]

    covid_mt_df = pd.merge(covid_serie, mt_df, left_index=True, right_index=True)
    plot_df(covid_mt_df)
    np_correlacion(covid_mt_df.dropna())


def np_correlacion_desplazada(covid_serie, mt_df, start_date, end_date, dias=7, avg=False):
    
    HORAS = 24
    var_df = mt_df.copy()
    
    for i in range(1, dias):
        
        var_df = var_df.shift(periods=-HORAS)
        var_mean_df = var_df[(var_df.index >= start_date) & (var_df.index < end_date)].resample('D').mean()
        
        if avg:
            var_mean_df = var_mean_df.rolling(i+1, min_periods=1).mean()
                        
        covid_mt_df = pd.merge(covid_serie, var_mean_df, left_index=True, right_index=True).dropna()
        np_correlacion(covid_mt_df, title=f'Correlacion corriendo {i} día(s) atrás promediado' if avg else f'Correlacion corriendo {i} día(s) atrás')


def analisis_pm25(pm25, start_year, end_year=None):
    ECA = 50
    
    if end_year is None:
        pm25df = pm25[pm25.index >= start_year].copy()
    else:
        pm25df = pm25[(pm25.index >= start_year) & (pm25.index >= end_year)].copy()
        
    pm25df.plot()
    plt.axhline(ECA, color='orange', label='ECA')
    plt.xlabel('Fecha')
    plt.ylabel(variable_dict['PM25'])
    plt.grid()
    plt.legend(['Promedio PM 2.5', 'ECA'])
    plt.show()


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

    """
    Analisis Primera Ola
    """
    fallecidos_ola_1 = fallecidos_df[fallecidos_df.index.year == 2020]
    positivos_ola_1 = positivos_df[positivos_df.index.year == 2020]
    pm25_ola_1 = pm25_mean[pm25_mean.index.year == 2020]
    
    f_pm25_o1 = pd.merge(fallecidos_ola_1, pm25_ola_1, left_index=True, right_index=True)
    p_pm25_o1 = pd.merge(positivos_ola_1, pm25_ola_1, left_index=True, right_index=True)
    
    plot_df(f_pm25_o1)
    plot_df(p_pm25_o1)

    np_correlacion(f_pm25_o1.dropna())
    np_correlacion(p_pm25_o1.dropna())
    
    # Analisis Correlacion 30 Dias Fallecidos vs PM2.5
    np_correlacion_desplazada(fallecidos_ola_1, pm25_df, start_date='2020-01-01', end_date='2021-01-01', dias=7)
    np_correlacion_desplazada(fallecidos_ola_1, pm25_df, start_date='2020-01-01', end_date='2021-01-01', dias=7, avg=True)
    
    # Analisis Correlacion 30 Dias Positivos vs PM2.5
    np_correlacion_desplazada(positivos_ola_1, pm25_df, start_date='2020-01-01', end_date='2021-01-01', dias=7)
    np_correlacion_desplazada(positivos_ola_1, pm25_df, start_date='2020-01-01', end_date='2021-01-01', dias=7, avg=True)
    
    """
    Analisis Segunda Ola
    """
    fallecidos_ola_2 = fallecidos_df[(fallecidos_df.index >= '2021-01-01') & (fallecidos_df.index < '2021-07-01')]
    positivos_ola_2 = positivos_df[(positivos_df.index >= '2021-01-01') & (positivos_df.index < '2021-07-01')]
    pm25_ola_2 = pm25_mean[(pm25_mean.index >= '2021-01-01') & (pm25_mean.index < '2021-07-01')]
    
    f_pm25_o2 = pd.merge(fallecidos_ola_2, pm25_ola_2, left_index=True, right_index=True)
    p_pm25_o2 = pd.merge(positivos_ola_2, pm25_ola_2, left_index=True, right_index=True)
    
    plot_df(f_pm25_o2)
    plot_df(p_pm25_o2)
    
    np_correlacion(f_pm25_o2.dropna())
    np_correlacion(p_pm25_o2.dropna())
    
    # Analisis Correlacion 30 Dias Fallecidos vs PM2.5
    np_correlacion_desplazada(fallecidos_ola_2, pm25_df, start_date='2021-01-01', end_date='2021-07-01', dias=7)
    np_correlacion_desplazada(fallecidos_ola_2, pm25_df, start_date='2021-01-01', end_date='2021-07-01', dias=7, avg=True)

    # Analisis Correlacion 30 Dias Positivos vs PM2.5
    np_correlacion_desplazada(positivos_ola_2, pm25_df, start_date='2021-01-01', end_date='2021-07-01', dias=7)
    np_correlacion_desplazada(positivos_ola_2, pm25_df, start_date='2021-01-01', end_date='2021-07-01', dias=7, avg=True)
    
    # PM 2.5 ECA
    analisis_pm25(pm25_mean, '2020-01-01')

    """
    Analisis de temperatura
    """
    
    # PRIMERA OLA
    temperatura_df = load_meteorologico('TEMPERATURA_MEDIA')
    analisis_covid_meteorologico(fallecidos_ola_1, 'TEMPERATURA_MEDIA')
    analisis_covid_meteorologico(positivos_ola_1, 'TEMPERATURA_MEDIA')
    
    np_correlacion_desplazada(fallecidos_ola_1, temperatura_df, start_date='2021-01-01', end_date='2021-07-01', dias=7)
    np_correlacion_desplazada(fallecidos_ola_1, temperatura_df, start_date='2021-01-01', end_date='2021-07-01', dias=7, avg=True)
    
    np_correlacion_desplazada(positivos_ola_1, temperatura_df, start_date='2021-01-01', end_date='2021-07-01', dias=7)
    np_correlacion_desplazada(positivos_ola_1, temperatura_df, start_date='2021-01-01', end_date='2021-07-01', dias=7, avg=True)
    
    
    # SEGUNDA OLA
    analisis_covid_meteorologico(fallecidos_ola_2, 'TEMPERATURA_MEDIA')
    analisis_covid_meteorologico(positivos_ola_2, 'TEMPERATURA_MEDIA')
    
    np_correlacion_desplazada(fallecidos_ola_2, temperatura_df, start_date='2021-01-01', end_date='2021-07-01', dias=7)
    np_correlacion_desplazada(fallecidos_ola_2, temperatura_df, start_date='2021-01-01', end_date='2021-07-01', dias=7, avg=True)
    
    np_correlacion_desplazada(positivos_ola_2, temperatura_df, start_date='2021-01-01', end_date='2021-07-01', dias=7)
    np_correlacion_desplazada(positivos_ola_2, temperatura_df, start_date='2021-01-01', end_date='2021-07-01', dias=7, avg=True)
    


if __name__ == '__main__':
    main()