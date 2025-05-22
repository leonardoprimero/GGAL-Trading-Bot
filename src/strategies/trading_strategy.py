import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TradingStrategy:
    """
    Clase para implementar estrategias de trading basadas en el modelo de predicción.
    """
    
    def __init__(self, confidence_threshold=0.7):
        """
        Inicializa la estrategia de trading.
        
        Args:
            confidence_threshold (float): Umbral de confianza para generar señales (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.positions = []
        self.current_position = None
    
    def generate_signals(self, predictions, probabilities, dates, prices):
        """
        Genera señales de trading basadas en las predicciones del modelo.
        
        Args:
            predictions (numpy.ndarray): Predicciones del modelo (0 o 1)
            probabilities (numpy.ndarray): Probabilidades de las predicciones
            dates (pandas.DatetimeIndex): Fechas correspondientes a las predicciones
            prices (pandas.Series): Precios correspondientes a las fechas
            
        Returns:
            pandas.DataFrame: DataFrame con las señales generadas
        """
        signals = pd.DataFrame(index=dates)
        signals['price'] = prices
        signals['prediction'] = predictions
        signals['probability'] = [prob[1] if len(prob) > 1 else 0.5 for prob in probabilities]
        signals['signal'] = 0
        
        # Generar señales basadas en predicciones y umbral de confianza
        signals.loc[
            (signals['prediction'] == 1) & 
            (signals['probability'] >= self.confidence_threshold), 
            'signal'
        ] = 1  # Señal de compra
        
        signals.loc[
            (signals['prediction'] == 0) & 
            (signals['probability'] >= self.confidence_threshold), 
            'signal'
        ] = -1  # Señal de venta
        
        return signals
    
    def backtest(self, signals, initial_capital=10000.0, position_size=0.2):
        """
        Realiza un backtest de la estrategia con datos históricos.
        
        Args:
            signals (pandas.DataFrame): DataFrame con señales generadas
            initial_capital (float): Capital inicial para el backtest
            position_size (float): Tamaño de posición como fracción del capital (0.0-1.0)
            
        Returns:
            pandas.DataFrame: DataFrame con resultados del backtest
        """
        # Crear DataFrame para el backtest
        backtest = signals.copy()
        backtest['capital'] = initial_capital
        backtest['position'] = 0
        backtest['holdings'] = 0
        backtest['cash'] = initial_capital
        backtest['returns'] = 0
        
        # Simular operaciones
        for i in range(1, len(backtest)):
            # Por defecto, mantener valores del día anterior
            backtest.iloc[i, backtest.columns.get_loc('position')] = backtest.iloc[i-1, backtest.columns.get_loc('position')]
            backtest.iloc[i, backtest.columns.get_loc('cash')] = backtest.iloc[i-1, backtest.columns.get_loc('cash')]
            
            # Procesar señales
            if backtest.iloc[i, backtest.columns.get_loc('signal')] == 1 and backtest.iloc[i-1, backtest.columns.get_loc('position')] <= 0:
                # Señal de compra
                available_cash = backtest.iloc[i-1, backtest.columns.get_loc('cash')]
                position_value = available_cash * position_size
                shares_to_buy = position_value / backtest.iloc[i, backtest.columns.get_loc('price')]
                
                backtest.iloc[i, backtest.columns.get_loc('position')] = shares_to_buy
                backtest.iloc[i, backtest.columns.get_loc('cash')] = available_cash - position_value
                
            elif backtest.iloc[i, backtest.columns.get_loc('signal')] == -1 and backtest.iloc[i-1, backtest.columns.get_loc('position')] > 0:
                # Señal de venta
                shares_to_sell = backtest.iloc[i-1, backtest.columns.get_loc('position')]
                sale_value = shares_to_sell * backtest.iloc[i, backtest.columns.get_loc('price')]
                
                backtest.iloc[i, backtest.columns.get_loc('position')] = 0
                backtest.iloc[i, backtest.columns.get_loc('cash')] = backtest.iloc[i-1, backtest.columns.get_loc('cash')] + sale_value
            
            # Actualizar valor de las posiciones y capital total
            backtest.iloc[i, backtest.columns.get_loc('holdings')] = backtest.iloc[i, backtest.columns.get_loc('position')] * backtest.iloc[i, backtest.columns.get_loc('price')]
            backtest.iloc[i, backtest.columns.get_loc('capital')] = backtest.iloc[i, backtest.columns.get_loc('cash')] + backtest.iloc[i, backtest.columns.get_loc('holdings')]
            
            # Calcular retornos diarios
            backtest.iloc[i, backtest.columns.get_loc('returns')] = backtest.iloc[i, backtest.columns.get_loc('capital')] / backtest.iloc[i-1, backtest.columns.get_loc('capital')] - 1
        
        # Calcular métricas de rendimiento
        backtest['cumulative_returns'] = (1 + backtest['returns']).cumprod() - 1
        
        return backtest
    
    def calculate_performance_metrics(self, backtest):
        """
        Calcula métricas de rendimiento para el backtest.
        
        Args:
            backtest (pandas.DataFrame): DataFrame con resultados del backtest
            
        Returns:
            dict: Métricas de rendimiento
        """
        # Calcular retorno total
        total_return = backtest['cumulative_returns'].iloc[-1]
        
        # Calcular retorno anualizado
        days = (backtest.index[-1] - backtest.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Calcular volatilidad anualizada
        daily_std = backtest['returns'].std()
        annual_volatility = daily_std * (252 ** 0.5)  # Asumiendo 252 días de trading al año
        
        # Calcular ratio de Sharpe (asumiendo tasa libre de riesgo = 0)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calcular drawdown máximo
        cumulative_returns = backtest['cumulative_returns']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calcular número de operaciones
        signals = backtest['signal']
        trades = signals[signals != 0].count()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades
        }
    
    def get_latest_signal(self, prediction, probability, price):
        """
        Obtiene la señal más reciente basada en la última predicción.
        
        Args:
            prediction (int): Última predicción del modelo (0 o 1)
            probability (float): Probabilidad de la predicción
            price (float): Precio actual
            
        Returns:
            dict: Información de la señal generada
        """
        signal = 0
        
        if prediction == 1 and probability >= self.confidence_threshold:
            signal = 1  # Compra
        elif prediction == 0 and probability >= self.confidence_threshold:
            signal = -1  # Venta
        
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'price': price,
            'prediction': prediction,
            'probability': probability,
            'signal': signal,
            'action': 'COMPRAR' if signal == 1 else 'VENDER' if signal == -1 else 'MANTENER'
        }
