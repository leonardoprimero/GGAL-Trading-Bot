from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os
import joblib

class ModelTrainer:
    """
    Clase para entrenar y evaluar el modelo de clasificación para GGAL.
    """
    
    def __init__(self, max_depth=6, criterion='entropy', random_state=1):
        """
        Inicializa el entrenador del modelo.
        
        Args:
            max_depth (int): Profundidad máxima del árbol de decisión
            criterion (str): Criterio para medir la calidad de la división ('entropy' o 'gini')
            random_state (int): Semilla para reproducibilidad
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            criterion=self.criterion, 
            max_depth=self.max_depth, 
            random_state=self.random_state
        )
        
    def train(self, X, y, test_size=0.4):
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X (pandas.DataFrame): Características para entrenamiento
            y (list): Variable objetivo
            test_size (float): Proporción de datos para prueba (0.0-1.0)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # Entrenar modelo
        self.model.fit(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def predict(self, X):
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X (pandas.DataFrame): Características para predicción
            
        Returns:
            numpy.ndarray: Predicciones (0 o 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Realiza predicciones de probabilidad con el modelo entrenado.
        
        Args:
            X (pandas.DataFrame): Características para predicción
            
        Returns:
            numpy.ndarray: Probabilidades para cada clase
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo con datos de prueba.
        
        Args:
            X_test (pandas.DataFrame): Características de prueba
            y_test (list): Variable objetivo de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        y_pred = self.predict(X_test)
        
        # Calcular matriz de confusión
        resultados = list(zip(y_test, y_pred))
        
        tn = 0  # Verdaderos negativos
        fp = 0  # Falsos positivos
        fn = 0  # Falsos negativos
        tp = 0  # Verdaderos positivos
        
        for real, prediccion in resultados:
            if (real == 0) & (prediccion == 0):
                tn += 1
            if (real == 0) & (prediccion == 1):
                fp += 1
            if (real == 1) & (prediccion == 0):
                fn += 1
            if (real == 1) & (prediccion == 1):
                tp += 1
        
        # Calcular métricas
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            }
        }
    
    def save_model(self, filepath):
        """
        Guarda el modelo entrenado en un archivo.
        
        Args:
            filepath (str): Ruta donde guardar el modelo
            
        Returns:
            bool: True si se guardó correctamente
        """
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar el modelo
        joblib.dump(self.model, filepath)
        return True
    
    def load_model(self, filepath):
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath (str): Ruta del modelo guardado
            
        Returns:
            bool: True si se cargó correctamente
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"El archivo de modelo {filepath} no existe")
        
        self.model = joblib.load(filepath)
        return True
