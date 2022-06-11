from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def date_parser_no_seconds(date_time):
    return datetime.strptime(date_time, '%d/%m/%Y %H:%M')


def date_parser_normal(date_time):
    return datetime.strptime(date_time, '%d/%m/%Y %H:%M:%S')


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
        date_parser=date_parser_no_seconds,
        index_col='FECHA')

    df = df.replace('S/D', np.nan)
    df['PM25'] = df['PM25'].astype(float)
    return df


def load_meteorologico(column):
    df = pd.read_csv(
        'meteorologica.csv',
        delimiter=';',
        parse_dates={ 'FECHA_HORA': ['FECHA', 'HORA'] },
        date_parser=date_parser_normal,
        usecols=['FECHA', 'HORA', column],
        index_col='FECHA_HORA')

    df = df.replace('S/D', np.nan)
    df[column] = df[column].astype(float)
    return df


def plot_df(df, xlabel='', ylabel=''):
    df.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def np_correlacion(df, column_x, column_y, xlabel='', ylabel=''):
    x = df[column_x].to_numpy()
    y = df[column_y].to_numpy()

    r = df[column_x].corr(df[column_y])

    plt.plot(x, y, 'o', label='Datos')

    m, b = np.polyfit(x, y, 1)

    plt.plot(x, m*x + b, '-', label='Regresion', color='red')
    plt.title(f'R = {round(r, 7)}')
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


def test():
    mt = load_meteorologico('TEMPERATURA_MEDIA')
    print(mt)


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
    
    fallecidos_pm25 = pd.merge(fallecidos_serie, pm25_mean_2020, left_index=True, right_index=True).dropna()
    positivos_pm25 = pd.merge(positivos_serie, pm25_mean_2020, left_index=True, right_index=True).dropna()

    np_correlacion(fallecidos_pm25, 'PM25', 'FALLECIDOS', xlabel='PM2.5 (ug/m3)', ylabel='Fallecidos')
    np_correlacion(positivos_pm25, 'PM25', 'POSITIVOS', xlabel='PM2.5 (ug/m3)', ylabel='Positivos')

    # pm25_mean_7dias = pm25_mean.copy()
    # for _ in range(7):
    #     pm25_mean_7dias = pm25_mean_7dias.shift(-1)
    #     pm25_mean_7dias = pm25_mean_7dias[pm25_mean_7dias.index.year == 2020]
    #     pm25_mean_7dias = pm25_mean_7dias.resample('D').mean()
    #     f_pm_2020 = pd.merge(fallecidos_serie, pm25_mean_7dias, left_index=True, right_index=True).dropna()

    #     correlacion(f_pm_2020['FALLECIDOS'], f_pm_2020['PM25'], 'PM2.5 (ug/m3)', 'Fallecidos')
    
    # Analisis de temperatura
    temperatura_df = load_meteorologico('TEMPERATURA_MEDIA')
    temperatura_df = temperatura_df.resample('D').mean()
    temperatura_df = temperatura_df[temperatura_df.index.year == 2020]

    fallecidos_temperatura = pd.merge(fallecidos_serie, temperatura_df, left_index=True, right_index=True)

    plot_df(fallecidos_temperatura)
    np_correlacion(fallecidos_temperatura.dropna(), 'TEMPERATURA_MEDIA', 'FALLECIDOS', xlabel='Temperatura (°C)', ylabel='Fallecidos')

    # Analisis de humedad
    humedad_df = load_meteorologico('HUMEDAD_RELATIVA')
    humedad_df = humedad_df.resample('D').mean()
    humedad_df = humedad_df[humedad_df.index.year == 2020]

    fallecidos_humedad = pd.merge(fallecidos_serie, humedad_df, left_index=True, right_index=True)

    plot_df(fallecidos_humedad)
    np_correlacion(fallecidos_humedad.dropna(), 'HUMEDAD_RELATIVA', 'FALLECIDOS', xlabel='Humedad Relativa (%)', ylabel='Fallecidos')

    # Analisis de precipitacion
    precipitacion_df = load_meteorologico('PRECIPITACION')
    precipitacion_df = precipitacion_df.resample('D').mean()
    precipitacion_df = precipitacion_df[precipitacion_df.index.year == 2020]

    fallecidos_precipitacion = pd.merge(fallecidos_serie, precipitacion_df, left_index=True, right_index=True)

    plot_df(fallecidos_precipitacion)
    np_correlacion(fallecidos_precipitacion.dropna(), 'PRECIPITACION', 'FALLECIDOS', xlabel='Precipitacion (mm)', ylabel='Fallecidos')

    # Analisis de velocidad de viento
    velocidad_df = load_meteorologico('VELOCIDAD_VIENTO')
    velocidad_df = velocidad_df.resample('D').mean()
    velocidad_df = velocidad_df[velocidad_df.index.year == 2020]

    fallecidos_velocidad = pd.merge(fallecidos_serie, velocidad_df, left_index=True, right_index=True)

    plot_df(fallecidos_velocidad)
    np_correlacion(fallecidos_velocidad.dropna(), 'VELOCIDAD_VIENTO', 'FALLECIDOS', xlabel='Velocidad de Viento (m/s)', ylabel='Fallecidos')

    # Analisis de direccion de viento
    direccion_df = load_meteorologico('DIRECCION_VIENTO')
    direccion_df = direccion_df.resample('D').mean()
    direccion_df = direccion_df[direccion_df.index.year == 2020]

    fallecidos_direccion = pd.merge(fallecidos_serie, direccion_df, left_index=True, right_index=True)

    plot_df(fallecidos_direccion)
    np_correlacion(fallecidos_direccion.dropna(), 'DIRECCION_VIENTO', 'FALLECIDOS', xlabel='Direccion de Viento (°)', ylabel='Fallecidos')

    # Analisis de radiacion solar total
    radiacion_df = load_meteorologico('RADIACION_SOLAR_TOTAL')
    radiacion_df = radiacion_df.resample('D').mean()
    radiacion_df = radiacion_df[radiacion_df.index.year == 2020]

    fallecidos_radiacion = pd.merge(fallecidos_serie, radiacion_df, left_index=True, right_index=True)

    plot_df(fallecidos_radiacion)
    np_correlacion(fallecidos_radiacion.dropna(), 'RADIACION_SOLAR_TOTAL', 'FALLECIDOS', xlabel='Radiacion Solar Total (W/m2)', ylabel='Fallecidos')



if __name__ == '__main__':
    main()
    #test()