import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    """
    Clase para cargar y procesar datos históricos del ticker GGAL.
    """
    
    def __init__(self, ticker="GGAL", start_date="2000-01-01"):
        """
        Inicializa el cargador de datos.
        
        Args:
            ticker (str): Símbolo del ticker a descargar (por defecto: "GGAL")
            start_date (str): Fecha de inicio para los datos históricos (formato: "YYYY-MM-DD")
        """
        self.ticker = ticker
        self.start_date = start_date
        self.data = None
        self.medias = ((63, 422), (38, 350), (72, 506))
        self.ventana = 100
        
    def download_data(self, end_date=None):
        """
        Descarga datos históricos del ticker.
        
        Args:
            end_date (str, opcional): Fecha final para los datos (formato: "YYYY-MM-DD")
                                     Si es None, se usa la fecha actual.
        
        Returns:
            pandas.DataFrame: DataFrame con los datos históricos
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        self.data = yf.download(self.ticker, start=self.start_date, end=end_date)
        return self.data
    
    def calculate_features(self):
        """
        Calcula características técnicas para el modelo.
        
        Returns:
            pandas.DataFrame: DataFrame con las características calculadas
        """
        if self.data is None:
            raise ValueError("Debe descargar los datos primero usando download_data()")
        
        # Calcular RSI
        dif = self.data['Adj Close'].diff()
        win = pd.DataFrame(np.where(dif > 0, dif, 0))
        loss = pd.DataFrame(np.where(dif < 0, abs(dif), 0))
        ema_win = win.ewm(alpha=1/14).mean()
        ema_loss = loss.ewm(alpha=1/14).mean()
        rs = ema_win / ema_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.index = self.data.index
        
        # Calcular otras características
        self.data['pctChange'] = self.data['Adj Close'].pct_change()
        self.data['fw'] = self.data['Adj Close'].shift(-self.ventana)/self.data['Adj Close']-1
        self.data['rsi'] = rsi/100
        self.data['roll_vol'] = self.data['pctChange'].rolling(60).std() * 60**0.5
        self.data['ema_vol'] = self.data['pctChange'].ewm(span=300).std() * 300**0.5
        self.data['cruce_1'] = self.data['Adj Close'].rolling(self.medias[0][0]).mean()/self.data['Adj Close'].rolling(self.medias[0][1]).mean()-1
        self.data['cruce_2'] = self.data['Adj Close'].rolling(self.medias[1][0]).mean()/self.data['Adj Close'].rolling(self.medias[1][1]).mean()-1
        self.data['cruce_3'] = self.data['Adj Close'].rolling(self.medias[2][0]).mean()/self.data['Adj Close'].rolling(self.medias[2][1]).mean()-1
        
        return self.data
    
    def prepare_training_data(self):
        """
        Prepara los datos para entrenamiento, incluyendo la variable objetivo.
        
        Returns:
            tuple: (X, y) donde X son las características y y es la variable objetivo
        """
        if self.data is None:
            raise ValueError("Debe descargar los datos primero usando download_data()")
        
        # Crear variable objetivo
        self.data['target'] = 0
        self.data.loc[self.data.fw >= 0, 'target'] = 1
        
        # Guardar una copia completa
        self.data_full = self.data.copy()
        
        # Limpiar datos
        data_clean = self.data.round(4).dropna()
        
        # Preparar X e y
        y = list(data_clean['target'])
        X = data_clean[['rsi', 'roll_vol', 'ema_vol', 'cruce_1', 'cruce_2', 'cruce_3']]
        
        return X, y
    
    def get_latest_features(self):
        """
        Obtiene las características más recientes para predicción.
        
        Returns:
            pandas.DataFrame: DataFrame con las características más recientes
        """
        if self.data is None:
            raise ValueError("Debe descargar los datos primero usando download_data()")
        
        latest_data = self.data.iloc[-1:][['rsi', 'roll_vol', 'ema_vol', 'cruce_1', 'cruce_2', 'cruce_3']]
        return latest_data
    
    def get_historical_data(self, days=100):
        """
        Obtiene los datos históricos para los últimos n días.
        
        Args:
            days (int): Número de días a recuperar
            
        Returns:
            pandas.DataFrame: DataFrame con los datos históricos
        """
        if self.data is None:
            raise ValueError("Debe descargar los datos primero usando download_data()")
        
        return self.data.iloc[-days:]
