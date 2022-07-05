from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dpns = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
dp = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

INICIO_PRIMERA_OLA = '2020-04-01'
FIN_PRIMERA_OLA = '2021-01-01'

INICIO_SEGUNDA_OLA = FIN_PRIMERA_OLA
FIN_SEGUNDA_OLA = '2021-07-01'

PM25 = 'PM25'
TEMPERATURA_MEDIA = 'TEMPERATURA_MEDIA'

DIAS_ATRAS = 7

var_dict = {
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


def plot_df(df, title=''):
    var_y1 = df.columns[0]
    var_y2 = df.columns[1]
    ax1 = df[var_y1].plot(y=var_dict[var_y1])
    ax2 = df[var_y2].plot(y=var_dict[var_y2], ax=ax1, secondary_y=True)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel(var_dict[var_y1])
    ax2.set_ylabel(var_dict[var_y2])
    ax1.legend([legend_dict[var_y1]], loc='upper left')
    ax2.legend([legend_dict[var_y2]], loc='upper right')
    plt.title(title)
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
    plt.xlabel(var_dict[var_x])
    plt.ylabel(var_dict[var_y])
    plt.grid()
    plt.legend()
    plt.show()


def smooth_plot(serie):
    serie.rolling(7, min_periods=1).mean().plot()
    plt.xlabel('Fecha')
    plt.ylabel(var_dict[serie.name])
    plt.grid()
    plt.show()


def analizar(covid_serie, var_name, start_date, end_date, dias_desplazados):
        
    if (var_name == PM25):
        df = load_pm25()
    else:
        df = load_meteorologico(var_name)
    
    df = df.resample('D').mean()
    filtered_df = filter_between_dates(df, start_date, end_date)

    covid_mt_df = pd.merge(covid_serie, filtered_df, how='left', left_index=True, right_index=True)
    plot_df(covid_mt_df)
    np_correlacion(covid_mt_df.dropna())
    np_correlacion_desplazada(covid_serie, df, start_date, end_date, dias=dias_desplazados)
    np_correlacion_desplazada(covid_serie, df, start_date, end_date, dias=dias_desplazados, avg=True)


def np_correlacion_desplazada(covid_serie, mt_df, start_date, end_date, dias, avg=False):
    
    # HORAS = 24
    OFFSET = 1
    df = mt_df.copy()
    
    for i in range(1, dias):
        
        df = df.shift(periods=-OFFSET)
        avg_df = filter_between_dates(df, start_date, end_date)
        
        if avg:
            avg_df = avg_df.rolling(i+1, min_periods=1).mean()
                        
        covid_mt_df = pd.merge(covid_serie, avg_df, how='left', left_index=True, right_index=True).dropna()
        np_correlacion(covid_mt_df, title=f'Correlacion corriendo {i} día(s) atrás promediado' if avg else f'Correlacion corriendo {i} día(s) atrás')


def eca(pm25, start_date, end_date, title):
    ECA = 50
    
    filtered_pm25 = filter_between_dates(pm25, start_date, end_date) 
        
    filtered_pm25.plot()
    plt.axhline(ECA, color='orange', label='ECA')
    plt.xlabel('Fecha')
    plt.ylabel(var_dict['PM25'])
    plt.title(title)
    plt.grid()
    plt.legend(['Promedio PM 2.5', 'ECA'])
    plt.show()
    
    
def inca(pm25, start_date, end_date, title):
    BUENA = 50
    MODERADA = 100
    MALA = 500
    
    filtered_pm25 = filter_between_dates(pm25, start_date, end_date)
    filtered_pm25['PM25'] = filtered_pm25['PM25'] * 100 / 25
    filtered_pm25.plot()
    plt.axhline(BUENA, color='green', label='Buena')
    plt.axhline(MODERADA, color='yellow', label='Moderada')
    plt.axhline(MALA, color='orange', label='Mala')
    plt.xlabel('Fecha')
    plt.ylabel(var_dict['PM25'])
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def filter_between_dates(df, start, end):
    return df[(df.index >= start) & (df.index < end)]
    

def main():

    # Analisis de fallecidos
    fallecidos_serie = load_fallecidos()
    smooth_plot(fallecidos_serie)

    # Analisis de positivos
    positivos_serie = load_positivos()
    smooth_plot(positivos_serie)

    # Analisis de PM25
    pm25_df = load_pm25()
    pm25_avg = pm25_df.resample('D').mean()
    pm25_min = pm25_df.resample('D').min()
    pm25_max = pm25_df.resample('D').max()

    plt.plot(pm25_avg, label='Promedio')
    plt.plot(pm25_min, label='Minimo')
    plt.plot(pm25_max, label='Maximo')
    plt.xlabel('Años')
    plt.ylabel(var_dict['PM25'])
    plt.grid()
    plt.legend()
    plt.show()
    
    """
    ANALISIS PRIMERA OLA
    """
    
    fallecidos_ola_1 = filter_between_dates(fallecidos_serie, INICIO_PRIMERA_OLA, FIN_PRIMERA_OLA)
    positivos_ola_1 = filter_between_dates(positivos_serie, INICIO_PRIMERA_OLA, FIN_PRIMERA_OLA)
    
    # ECA
    eca(pm25_avg, INICIO_PRIMERA_OLA, FIN_PRIMERA_OLA, 'ECA PM 2.5 PRIMERA OLA')
    
    # INCA
    inca(pm25_avg, INICIO_PRIMERA_OLA, FIN_PRIMERA_OLA, 'INCA PM 2.5 PRIMERA OLA')
    
    # Fallecidos vs PM 2.5
    analizar(fallecidos_ola_1, PM25, INICIO_PRIMERA_OLA, FIN_PRIMERA_OLA, DIAS_ATRAS)

    # Positivos vs PM 2.5
    analizar(positivos_ola_1, PM25, INICIO_PRIMERA_OLA, FIN_PRIMERA_OLA, DIAS_ATRAS)
    
    # Fallecidos vs Temperatura
    analizar(fallecidos_ola_1, TEMPERATURA_MEDIA, INICIO_PRIMERA_OLA, FIN_PRIMERA_OLA, DIAS_ATRAS)
    
    # Positivos vs Temperatura Media
    analizar(positivos_ola_1, TEMPERATURA_MEDIA, INICIO_PRIMERA_OLA, FIN_PRIMERA_OLA, DIAS_ATRAS)
    
    """
    ANALISIS SEGUNDA OLA
    """
    
    fallecidos_ola_2 = filter_between_dates(fallecidos_serie, INICIO_SEGUNDA_OLA, FIN_SEGUNDA_OLA)
    positivos_ola_2 = filter_between_dates(positivos_serie, INICIO_SEGUNDA_OLA, FIN_SEGUNDA_OLA)
    
    # ECA
    eca(pm25_avg, INICIO_SEGUNDA_OLA, FIN_SEGUNDA_OLA, 'ECA PM 2.5 SEGUNDA OLA')
    
    # INCA
    inca(pm25_avg, INICIO_SEGUNDA_OLA, FIN_SEGUNDA_OLA, 'INCA PM 2.5 SEGUNDA OLA')
    
    # Fallecidos vs PM 2.5
    analizar(fallecidos_ola_2, PM25, INICIO_SEGUNDA_OLA, FIN_SEGUNDA_OLA, DIAS_ATRAS)

    # Positivos vs PM 2.5
    analizar(positivos_ola_2, PM25, INICIO_SEGUNDA_OLA, FIN_SEGUNDA_OLA, DIAS_ATRAS)
    
    # Fallecidos vs Temperatura
    analizar(fallecidos_ola_2, TEMPERATURA_MEDIA, INICIO_SEGUNDA_OLA, FIN_SEGUNDA_OLA, DIAS_ATRAS)
    
    # Positivos vs Temperatura Media
    analizar(positivos_ola_2, TEMPERATURA_MEDIA, INICIO_SEGUNDA_OLA, FIN_SEGUNDA_OLA, DIAS_ATRAS) 


if __name__ == '__main__':
    main()