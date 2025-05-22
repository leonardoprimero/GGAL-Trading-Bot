import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, 
                            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                            QFileDialog, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
                            QCheckBox, QProgressBar, QSplitter, QFrame, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QDateTime
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor

# Importar módulos propios
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import DataLoader
from models.model_trainer import ModelTrainer
from strategies.trading_strategy import TradingStrategy

class MatplotlibCanvas(FigureCanvas):
    """Clase para integrar gráficos de Matplotlib en PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)

class DataThread(QThread):
    """Hilo para cargar datos sin bloquear la interfaz"""
    update_progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, ticker, start_date):
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        
    def run(self):
        try:
            self.update_progress.emit(10)
            data_loader = DataLoader(ticker=self.ticker, start_date=self.start_date)
            self.update_progress.emit(30)
            data = data_loader.download_data()
            self.update_progress.emit(60)
            data = data_loader.calculate_features()
            self.update_progress.emit(90)
            X, y = data_loader.prepare_training_data()
            self.update_progress.emit(100)
            self.finished_signal.emit((data_loader, data, X, y))
        except Exception as e:
            self.error_signal.emit(str(e))

class TrainingThread(QThread):
    """Hilo para entrenar el modelo sin bloquear la interfaz"""
    update_progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, X, y, max_depth, criterion, test_size):
        super().__init__()
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.criterion = criterion
        self.test_size = test_size
        
    def run(self):
        try:
            self.update_progress.emit(10)
            model_trainer = ModelTrainer(max_depth=self.max_depth, criterion=self.criterion)
            self.update_progress.emit(30)
            X_train, X_test, y_train, y_test = model_trainer.train(self.X, self.y, test_size=self.test_size)
            self.update_progress.emit(60)
            metrics = model_trainer.evaluate(X_test, y_test)
            self.update_progress.emit(90)
            self.update_progress.emit(100)
            self.finished_signal.emit((model_trainer, X_train, X_test, y_train, y_test, metrics))
        except Exception as e:
            self.error_signal.emit(str(e))

class BacktestThread(QThread):
    """Hilo para ejecutar backtest sin bloquear la interfaz"""
    update_progress = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, model_trainer, data, confidence_threshold, initial_capital, position_size):
        super().__init__()
        self.model_trainer = model_trainer
        self.data = data
        self.confidence_threshold = confidence_threshold
        self.initial_capital = initial_capital
        self.position_size = position_size
        
    def run(self):
        try:
            self.update_progress.emit(10)
            # Preparar datos para backtest
            X = self.data[['rsi', 'roll_vol', 'ema_vol', 'cruce_1', 'cruce_2', 'cruce_3']].dropna()
            self.update_progress.emit(30)
            
            # Generar predicciones
            predictions = self.model_trainer.predict(X)
            probabilities = self.model_trainer.predict_proba(X)
            self.update_progress.emit(50)
            
            # Crear estrategia y generar señales
            strategy = TradingStrategy(confidence_threshold=self.confidence_threshold)
            signals = strategy.generate_signals(
                predictions, 
                probabilities, 
                X.index, 
                self.data.loc[X.index, 'Adj Close']
            )
            self.update_progress.emit(70)
            
            # Ejecutar backtest
            backtest_results = strategy.backtest(
                signals, 
                initial_capital=self.initial_capital,
                position_size=self.position_size
            )
            self.update_progress.emit(90)
            
            # Calcular métricas
            performance_metrics = strategy.calculate_performance_metrics(backtest_results)
            self.update_progress.emit(100)
            
            self.finished_signal.emit((strategy, signals, backtest_results, performance_metrics))
        except Exception as e:
            self.error_signal.emit(str(e))

class GGALTradingBotGUI(QMainWindow):
    """Interfaz gráfica principal para el bot de trading de GGAL"""
    
    def __init__(self):
        super().__init__()
        
        # Variables de estado
        self.data_loader = None
        self.data = None
        self.X = None
        self.y = None
        self.model_trainer = None
        self.strategy = None
        self.signals = None
        self.backtest_results = None
        
        # Configurar la ventana principal
        self.setWindowTitle("GGAL Trading Bot")
        self.setGeometry(100, 100, 1200, 800)
        
        # Crear el widget de pestañas
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Crear pestañas
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_backtest_tab()
        self.setup_live_trading_tab()
        
        # Mostrar mensaje de bienvenida
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Muestra un mensaje de bienvenida al iniciar la aplicación"""
        msg = QMessageBox()
        msg.setWindowTitle("Bienvenido a GGAL Trading Bot")
        msg.setText("Bienvenido al Bot de Trading para GGAL")
        msg.setInformativeText("Esta aplicación te permite entrenar modelos de trading para el ticker GGAL, "
                              "realizar backtests y ejecutar operaciones en tiempo real.\n\n"
                              "Para comenzar, ve a la pestaña 'Datos' y carga los datos históricos.")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
    
    def setup_data_tab(self):
        """Configura la pestaña de datos"""
        data_tab = QWidget()
        layout = QVBoxLayout()
        
        # Grupo de configuración
        config_group = QGroupBox("Configuración de Datos")
        config_layout = QFormLayout()
        
        # Ticker
        self.ticker_input = QLineEdit("GGAL")
        config_layout.addRow("Ticker:", self.ticker_input)
        
        # Fecha de inicio
        self.start_date_input = QLineEdit("2000-01-01")
        config_layout.addRow("Fecha de inicio:", self.start_date_input)
        
        # Botón de carga
        self.load_button = QPushButton("Cargar Datos")
        self.load_button.clicked.connect(self.load_data)
        config_layout.addRow("", self.load_button)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Barra de progreso
        self.data_progress = QProgressBar()
        self.data_progress.setValue(0)
        layout.addWidget(self.data_progress)
        
        # Área de visualización de datos
        self.data_view = QTabWidget()
        
        # Pestaña de tabla
        self.data_table = QTableWidget()
        self.data_view.addTab(self.data_table, "Tabla")
        
        # Pestaña de gráfico
        self.data_plot_widget = QWidget()
        self.data_plot_layout = QVBoxLayout()
        self.data_canvas = MatplotlibCanvas(width=5, height=4, dpi=100)
        self.data_plot_layout.addWidget(self.data_canvas)
        self.data_plot_widget.setLayout(self.data_plot_layout)
        self.data_view.addTab(self.data_plot_widget, "Gráfico")
        
        layout.addWidget(self.data_view)
        
        data_tab.setLayout(layout)
        self.tabs.addTab(data_tab, "Datos")
    
    def setup_model_tab(self):
        """Configura la pestaña del modelo"""
        model_tab = QWidget()
        layout = QVBoxLayout()
        
        # Grupo de configuración del modelo
        model_config_group = QGroupBox("Configuración del Modelo")
        model_config_layout = QFormLayout()
        
        # Profundidad máxima
        self.max_depth_input = QSpinBox()
        self.max_depth_input.setRange(1, 20)
        self.max_depth_input.setValue(6)
        model_config_layout.addRow("Profundidad máxima:", self.max_depth_input)
        
        # Criterio
        self.criterion_input = QComboBox()
        self.criterion_input.addItems(["entropy", "gini"])
        model_config_layout.addRow("Criterio:", self.criterion_input)
        
        # Tamaño de prueba
        self.test_size_input = QDoubleSpinBox()
        self.test_size_input.setRange(0.1, 0.9)
        self.test_size_input.setSingleStep(0.1)
        self.test_size_input.setValue(0.4)
        model_config_layout.addRow("Tamaño de prueba:", self.test_size_input)
        
        # Botón de entrenamiento
        self.train_button = QPushButton("Entrenar Modelo")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        model_config_layout.addRow("", self.train_button)
        
        model_config_group.setLayout(model_config_layout)
        layout.addWidget(model_config_group)
        
        # Barra de progreso
        self.model_progress = QProgressBar()
        self.model_progress.setValue(0)
        layout.addWidget(self.model_progress)
        
        # Grupo de resultados
        results_group = QGroupBox("Resultados del Modelo")
        results_layout = QVBoxLayout()
        
        # Tabla de métricas
        self.metrics_table = QTableWidget(5, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.metrics_table.setItem(0, 0, QTableWidgetItem("Precisión"))
        self.metrics_table.setItem(1, 0, QTableWidgetItem("Recall"))
        self.metrics_table.setItem(2, 0, QTableWidgetItem("F1-Score"))
        self.metrics_table.setItem(3, 0, QTableWidgetItem("Exactitud"))
        self.metrics_table.setItem(4, 0, QTableWidgetItem("Matriz de Confusión"))
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.metrics_table)
        
        # Botones para guardar/cargar modelo
        buttons_layout = QHBoxLayout()
        self.save_model_button = QPushButton("Guardar Modelo")
        self.save_model_button.clicked.connect(self.save_model)
        self.save_model_button.setEnabled(False)
        buttons_layout.addWidget(self.save_model_button)
        
        self.load_model_button = QPushButton("Cargar Modelo")
        self.load_model_button.clicked.connect(self.load_model)
        buttons_layout.addWidget(self.load_model_button)
        
        results_layout.addLayout(buttons_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        model_tab.setLayout(layout)
        self.tabs.addTab(model_tab, "Modelo")
    
    def setup_backtest_tab(self):
        """Configura la pestaña de backtest"""
        backtest_tab = QWidget()
        layout = QVBoxLayout()
        
        # Grupo de configuración de backtest
        backtest_config_group = QGroupBox("Configuración de Backtest")
        backtest_config_layout = QFormLayout()
        
        # Umbral de confianza
        self.confidence_threshold_input = QDoubleSpinBox()
        self.confidence_threshold_input.setRange(0.5, 1.0)
        self.confidence_threshold_input.setSingleStep(0.05)
        self.confidence_threshold_input.setValue(0.7)
        backtest_config_layout.addRow("Umbral de confianza:", self.confidence_threshold_input)
        
        # Capital inicial
        self.initial_capital_input = QDoubleSpinBox()
        self.initial_capital_input.setRange(1000, 1000000)
        self.initial_capital_input.setSingleStep(1000)
        self.initial_capital_input.setValue(10000)
        backtest_config_layout.addRow("Capital inicial:", self.initial_capital_input)
        
        # Tamaño de posición
        self.position_size_input = QDoubleSpinBox()
        self.position_size_input.setRange(0.1, 1.0)
        self.position_size_input.setSingleStep(0.1)
        self.position_size_input.setValue(0.2)
        backtest_config_layout.addRow("Tamaño de posición:", self.position_size_input)
        
        # Botón de ejecución
        self.run_backtest_button = QPushButton("Ejecutar Backtest")
        self.run_backtest_button.clicked.connect(self.run_backtest)
        self.run_backtest_button.setEnabled(False)
        backtest_config_layout.addRow("", self.run_backtest_button)
        
        backtest_config_group.setLayout(backtest_config_layout)
        layout.addWidget(backtest_config_group)
        
        # Barra de progreso
        self.backtest_progress = QProgressBar()
        self.backtest_progress.setValue(0)
        layout.addWidget(self.backtest_progress)
        
        # Área de resultados
        self.backtest_results_tabs = QTabWidget()
        
        # Pestaña de métricas
        self.backtest_metrics_widget = QWidget()
        self.backtest_metrics_layout = QVBoxLayout()
        self.backtest_metrics_table = QTableWidget(6, 2)
        self.backtest_metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.backtest_metrics_table.setItem(0, 0, QTableWidgetItem("Retorno Total"))
        self.backtest_metrics_table.setItem(1, 0, QTableWidgetItem("Retorno Anualizado"))
        self.backtest_metrics_table.setItem(2, 0, QTableWidgetItem("Volatilidad Anualizada"))
        self.backtest_metrics_table.setItem(3, 0, QTableWidgetItem("Ratio de Sharpe"))
        self.backtest_metrics_table.setItem(4, 0, QTableWidgetItem("Drawdown Máximo"))
        self.backtest_metrics_table.setItem(5, 0, QTableWidgetItem("Número de Operaciones"))
        self.backtest_metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.backtest_metrics_layout.addWidget(self.backtest_metrics_table)
        self.backtest_metrics_widget.setLayout(self.backtest_metrics_layout)
        self.backtest_results_tabs.addTab(self.backtest_metrics_widget, "Métricas")
        
        # Pestaña de gráfico de equity
        self.backtest_equity_widget = QWidget()
        self.backtest_equity_layout = QVBoxLayout()
        self.backtest_equity_canvas = MatplotlibCanvas(width=5, height=4, dpi=100)
        self.backtest_equity_layout.addWidget(self.backtest_equity_canvas)
        self.backtest_equity_widget.setLayout(self.backtest_equity_layout)
        self.backtest_results_tabs.addTab(self.backtest_equity_widget, "Equity")
        
        # Pestaña de señales
        self.backtest_signals_widget = QWidget()
        self.backtest_signals_layout = QVBoxLayout()
        self.backtest_signals_canvas = MatplotlibCanvas(width=5, height=4, dpi=100)
        self.backtest_signals_layout.addWidget(self.backtest_signals_canvas)
        self.backtest_signals_widget.setLayout(self.backtest_signals_layout)
        self.backtest_results_tabs.addTab(self.backtest_signals_widget, "Señales")
        
        layout.addWidget(self.backtest_results_tabs)
        
        backtest_tab.setLayout(layout)
        self.tabs.addTab(backtest_tab, "Backtest")
    
    def setup_live_trading_tab(self):
        """Configura la pestaña de trading en vivo"""
        live_tab = QWidget()
        layout = QVBoxLayout()
        
        # Grupo de estado actual
        status_group = QGroupBox("Estado Actual")
        status_layout = QFormLayout()
        
        # Último precio
        self.last_price_label = QLabel("N/A")
        status_layout.addRow("Último precio:", self.last_price_label)
        
        # Última predicción
        self.last_prediction_label = QLabel("N/A")
        status_layout.addRow("Última predicción:", self.last_prediction_label)
        
        # Última señal
        self.last_signal_label = QLabel("N/A")
        status_layout.addRow("Última señal:", self.last_signal_label)
        
        # Botón de actualización
        self.update_button = QPushButton("Actualizar Datos")
        self.update_button.clicked.connect(self.update_live_data)
        self.update_button.setEnabled(False)
        status_layout.addRow("", self.update_button)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Grupo de configuración de trading
        trading_config_group = QGroupBox("Configuración de Trading")
        trading_config_layout = QFormLayout()
        
        # Modo automático
        self.auto_mode_checkbox = QCheckBox()
        trading_config_layout.addRow("Modo automático:", self.auto_mode_checkbox)
        
        # Intervalo de actualización
        self.update_interval_input = QSpinBox()
        self.update_interval_input.setRange(1, 60)
        self.update_interval_input.setValue(5)
        trading_config_layout.addRow("Intervalo (min):", self.update_interval_input)
        
        # Botones de trading
        trading_buttons_layout = QHBoxLayout()
        self.start_trading_button = QPushButton("Iniciar Trading")
        self.start_trading_button.clicked.connect(self.start_trading)
        self.start_trading_button.setEnabled(False)
        trading_buttons_layout.addWidget(self.start_trading_button)
        
        self.stop_trading_button = QPushButton("Detener Trading")
        self.stop_trading_button.clicked.connect(self.stop_trading)
        self.stop_trading_button.setEnabled(False)
        trading_buttons_layout.addWidget(self.stop_trading_button)
        
        trading_config_layout.addRow("", trading_buttons_layout)
        
        trading_config_group.setLayout(trading_config_layout)
        layout.addWidget(trading_config_group)
        
        # Registro de operaciones
        log_group = QGroupBox("Registro de Operaciones")
        log_layout = QVBoxLayout()
        
        self.trading_log = QTextEdit()
        self.trading_log.setReadOnly(True)
        log_layout.addWidget(self.trading_log)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        live_tab.setLayout(layout)
        self.tabs.addTab(live_tab, "Trading en Vivo")
    
    def load_data(self):
        """Carga los datos históricos"""
        ticker = self.ticker_input.text()
        start_date = self.start_date_input.text()
        
        if not ticker or not start_date:
            QMessageBox.warning(self, "Error", "Por favor, ingrese un ticker y una fecha de inicio válidos.")
            return
        
        # Deshabilitar botón mientras se cargan los datos
        self.load_button.setEnabled(False)
        self.data_progress.setValue(0)
        
        # Crear y ejecutar hilo para cargar datos
        self.data_thread = DataThread(ticker, start_date)
        self.data_thread.update_progress.connect(self.data_progress.setValue)
        self.data_thread.finished_signal.connect(self.on_data_loaded)
        self.data_thread.error_signal.connect(self.on_data_error)
        self.data_thread.start()
    
    def on_data_loaded(self, result):
        """Callback cuando los datos se han cargado correctamente"""
        self.data_loader, self.data, self.X, self.y = result
        
        # Actualizar tabla de datos
        self.update_data_table()
        
        # Actualizar gráfico de datos
        self.update_data_plot()
        
        # Habilitar botones relevantes
        self.load_button.setEnabled(True)
        self.train_button.setEnabled(True)
        
        QMessageBox.information(self, "Éxito", f"Datos cargados correctamente. {len(self.data)} registros encontrados.")
    
    def on_data_error(self, error_msg):
        """Callback cuando hay un error al cargar los datos"""
        self.load_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error al cargar datos: {error_msg}")
    
    def update_data_table(self):
        """Actualiza la tabla de datos con los datos cargados"""
        if self.data is None:
            return
        
        # Preparar tabla
        self.data_table.setRowCount(min(100, len(self.data)))  # Mostrar solo los primeros 100 registros
        self.data_table.setColumnCount(len(self.data.columns))
        self.data_table.setHorizontalHeaderLabels(self.data.columns)
        
        # Llenar tabla
        for i in range(min(100, len(self.data))):
            for j in range(len(self.data.columns)):
                value = str(self.data.iloc[i, j])
                self.data_table.setItem(i, j, QTableWidgetItem(value))
        
        # Ajustar tamaño de columnas
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    def update_data_plot(self):
        """Actualiza el gráfico de datos"""
        if self.data is None:
            return
        
        # Limpiar gráfico anterior
        self.data_canvas.axes.clear()
        
        # Graficar precio de cierre
        self.data_canvas.axes.plot(self.data.index, self.data['Adj Close'], label='Precio de Cierre')
        
        # Configurar gráfico
        self.data_canvas.axes.set_title('Precio Histórico de GGAL')
        self.data_canvas.axes.set_xlabel('Fecha')
        self.data_canvas.axes.set_ylabel('Precio (USD)')
        self.data_canvas.axes.legend()
        self.data_canvas.axes.grid(True)
        
        # Rotar etiquetas de fecha para mejor visualización
        plt.setp(self.data_canvas.axes.get_xticklabels(), rotation=45)
        
        # Ajustar layout
        self.data_canvas.fig.tight_layout()
        
        # Actualizar canvas
        self.data_canvas.draw()
    
    def train_model(self):
        """Entrena el modelo con los datos cargados"""
        if self.X is None or self.y is None:
            QMessageBox.warning(self, "Error", "No hay datos disponibles para entrenar el modelo.")
            return
        
        # Obtener parámetros
        max_depth = self.max_depth_input.value()
        criterion = self.criterion_input.currentText()
        test_size = self.test_size_input.value()
        
        # Deshabilitar botón mientras se entrena el modelo
        self.train_button.setEnabled(False)
        self.model_progress.setValue(0)
        
        # Crear y ejecutar hilo para entrenar modelo
        self.training_thread = TrainingThread(self.X, self.y, max_depth, criterion, test_size)
        self.training_thread.update_progress.connect(self.model_progress.setValue)
        self.training_thread.finished_signal.connect(self.on_model_trained)
        self.training_thread.error_signal.connect(self.on_model_error)
        self.training_thread.start()
    
    def on_model_trained(self, result):
        """Callback cuando el modelo se ha entrenado correctamente"""
        self.model_trainer, X_train, X_test, y_train, y_test, metrics = result
        
        # Actualizar tabla de métricas
        self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{metrics['precision']:.4f}"))
        self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{metrics['recall']:.4f}"))
        self.metrics_table.setItem(2, 1, QTableWidgetItem(f"{metrics['f1']:.4f}"))
        self.metrics_table.setItem(3, 1, QTableWidgetItem(f"{metrics['accuracy']:.4f}"))
        
        cm = metrics['confusion_matrix']
        cm_text = f"VN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}, VP: {cm['tp']}"
        self.metrics_table.setItem(4, 1, QTableWidgetItem(cm_text))
        
        # Habilitar botones relevantes
        self.train_button.setEnabled(True)
        self.save_model_button.setEnabled(True)
        self.run_backtest_button.setEnabled(True)
        self.update_button.setEnabled(True)
        
        QMessageBox.information(self, "Éxito", "Modelo entrenado correctamente.")
    
    def on_model_error(self, error_msg):
        """Callback cuando hay un error al entrenar el modelo"""
        self.train_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error al entrenar modelo: {error_msg}")
    
    def save_model(self):
        """Guarda el modelo entrenado en un archivo"""
        if self.model_trainer is None:
            QMessageBox.warning(self, "Error", "No hay modelo para guardar.")
            return
        
        # Abrir diálogo para seleccionar archivo
        filepath, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo", "", "Modelo (*.joblib)")
        
        if filepath:
            try:
                self.model_trainer.save_model(filepath)
                QMessageBox.information(self, "Éxito", f"Modelo guardado en {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al guardar modelo: {str(e)}")
    
    def load_model(self):
        """Carga un modelo previamente guardado"""
        # Abrir diálogo para seleccionar archivo
        filepath, _ = QFileDialog.getOpenFileName(self, "Cargar Modelo", "", "Modelo (*.joblib)")
        
        if filepath:
            try:
                self.model_trainer = ModelTrainer()
                self.model_trainer.load_model(filepath)
                
                # Habilitar botones relevantes
                self.save_model_button.setEnabled(True)
                self.run_backtest_button.setEnabled(True)
                self.update_button.setEnabled(True)
                
                QMessageBox.information(self, "Éxito", f"Modelo cargado desde {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al cargar modelo: {str(e)}")
    
    def run_backtest(self):
        """Ejecuta un backtest con el modelo entrenado"""
        if self.model_trainer is None:
            QMessageBox.warning(self, "Error", "No hay modelo para ejecutar backtest.")
            return
        
        if self.data is None:
            QMessageBox.warning(self, "Error", "No hay datos para ejecutar backtest.")
            return
        
        # Obtener parámetros
        confidence_threshold = self.confidence_threshold_input.value()
        initial_capital = self.initial_capital_input.value()
        position_size = self.position_size_input.value()
        
        # Deshabilitar botón mientras se ejecuta el backtest
        self.run_backtest_button.setEnabled(False)
        self.backtest_progress.setValue(0)
        
        # Crear y ejecutar hilo para backtest
        self.backtest_thread = BacktestThread(
            self.model_trainer, 
            self.data, 
            confidence_threshold, 
            initial_capital, 
            position_size
        )
        self.backtest_thread.update_progress.connect(self.backtest_progress.setValue)
        self.backtest_thread.finished_signal.connect(self.on_backtest_finished)
        self.backtest_thread.error_signal.connect(self.on_backtest_error)
        self.backtest_thread.start()
    
    def on_backtest_finished(self, result):
        """Callback cuando el backtest ha finalizado correctamente"""
        self.strategy, self.signals, self.backtest_results, performance_metrics = result
        
        # Actualizar tabla de métricas
        self.backtest_metrics_table.setItem(0, 1, QTableWidgetItem(f"{performance_metrics['total_return']:.2%}"))
        self.backtest_metrics_table.setItem(1, 1, QTableWidgetItem(f"{performance_metrics['annual_return']:.2%}"))
        self.backtest_metrics_table.setItem(2, 1, QTableWidgetItem(f"{performance_metrics['annual_volatility']:.2%}"))
        self.backtest_metrics_table.setItem(3, 1, QTableWidgetItem(f"{performance_metrics['sharpe_ratio']:.2f}"))
        self.backtest_metrics_table.setItem(4, 1, QTableWidgetItem(f"{performance_metrics['max_drawdown']:.2%}"))
        self.backtest_metrics_table.setItem(5, 1, QTableWidgetItem(f"{performance_metrics['trades']}"))
        
        # Actualizar gráfico de equity
        self.update_equity_plot()
        
        # Actualizar gráfico de señales
        self.update_signals_plot()
        
        # Habilitar botones relevantes
        self.run_backtest_button.setEnabled(True)
        self.start_trading_button.setEnabled(True)
        
        QMessageBox.information(self, "Éxito", "Backtest ejecutado correctamente.")
    
    def on_backtest_error(self, error_msg):
        """Callback cuando hay un error al ejecutar el backtest"""
        self.run_backtest_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error al ejecutar backtest: {error_msg}")
    
    def update_equity_plot(self):
        """Actualiza el gráfico de equity"""
        if self.backtest_results is None:
            return
        
        # Limpiar gráfico anterior
        self.backtest_equity_canvas.axes.clear()
        
        # Graficar equity
        self.backtest_equity_canvas.axes.plot(
            self.backtest_results.index, 
            self.backtest_results['capital'], 
            label='Capital'
        )
        
        # Configurar gráfico
        self.backtest_equity_canvas.axes.set_title('Evolución del Capital')
        self.backtest_equity_canvas.axes.set_xlabel('Fecha')
        self.backtest_equity_canvas.axes.set_ylabel('Capital (USD)')
        self.backtest_equity_canvas.axes.legend()
        self.backtest_equity_canvas.axes.grid(True)
        
        # Rotar etiquetas de fecha para mejor visualización
        plt.setp(self.backtest_equity_canvas.axes.get_xticklabels(), rotation=45)
        
        # Ajustar layout
        self.backtest_equity_canvas.fig.tight_layout()
        
        # Actualizar canvas
        self.backtest_equity_canvas.draw()
    
    def update_signals_plot(self):
        """Actualiza el gráfico de señales"""
        if self.signals is None or self.backtest_results is None:
            return
        
        # Limpiar gráfico anterior
        self.backtest_signals_canvas.axes.clear()
        
        # Graficar precio
        self.backtest_signals_canvas.axes.plot(
            self.signals.index, 
            self.signals['price'], 
            label='Precio', 
            color='blue'
        )
        
        # Graficar señales de compra
        buy_signals = self.signals[self.signals['signal'] == 1]
        self.backtest_signals_canvas.axes.scatter(
            buy_signals.index, 
            buy_signals['price'], 
            color='green', 
            marker='^', 
            s=100, 
            label='Compra'
        )
        
        # Graficar señales de venta
        sell_signals = self.signals[self.signals['signal'] == -1]
        self.backtest_signals_canvas.axes.scatter(
            sell_signals.index, 
            sell_signals['price'], 
            color='red', 
            marker='v', 
            s=100, 
            label='Venta'
        )
        
        # Configurar gráfico
        self.backtest_signals_canvas.axes.set_title('Señales de Trading')
        self.backtest_signals_canvas.axes.set_xlabel('Fecha')
        self.backtest_signals_canvas.axes.set_ylabel('Precio (USD)')
        self.backtest_signals_canvas.axes.legend()
        self.backtest_signals_canvas.axes.grid(True)
        
        # Rotar etiquetas de fecha para mejor visualización
        plt.setp(self.backtest_signals_canvas.axes.get_xticklabels(), rotation=45)
        
        # Ajustar layout
        self.backtest_signals_canvas.fig.tight_layout()
        
        # Actualizar canvas
        self.backtest_signals_canvas.draw()
    
    def update_live_data(self):
        """Actualiza los datos en vivo"""
        if self.data_loader is None or self.model_trainer is None:
            QMessageBox.warning(self, "Error", "No hay datos o modelo disponible.")
            return
        
        try:
            # Descargar datos actualizados
            self.data_loader.download_data()
            self.data_loader.calculate_features()
            
            # Obtener características más recientes
            latest_features = self.data_loader.get_latest_features()
            
            if latest_features.isnull().values.any():
                QMessageBox.warning(self, "Error", "Los datos más recientes contienen valores nulos.")
                return
            
            # Realizar predicción
            prediction = self.model_trainer.predict(latest_features)[0]
            probability = self.model_trainer.predict_proba(latest_features)[0][1]
            
            # Obtener último precio
            last_price = self.data_loader.data['Adj Close'].iloc[-1]
            
            # Generar señal
            signal_info = self.strategy.get_latest_signal(prediction, probability, last_price)
            
            # Actualizar etiquetas
            self.last_price_label.setText(f"${last_price:.2f}")
            self.last_prediction_label.setText(f"{'Subir' if prediction == 1 else 'Bajar'} ({probability:.2%})")
            self.last_signal_label.setText(signal_info['action'])
            
            # Agregar al registro
            log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Precio: ${last_price:.2f}, "
            log_entry += f"Predicción: {'Subir' if prediction == 1 else 'Bajar'} ({probability:.2%}), "
            log_entry += f"Señal: {signal_info['action']}"
            self.trading_log.append(log_entry)
            
            QMessageBox.information(self, "Éxito", "Datos actualizados correctamente.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al actualizar datos: {str(e)}")
    
    def start_trading(self):
        """Inicia el trading automático"""
        QMessageBox.information(self, "Información", "La funcionalidad de trading automático está en desarrollo.")
        self.start_trading_button.setEnabled(False)
        self.stop_trading_button.setEnabled(True)
    
    def stop_trading(self):
        """Detiene el trading automático"""
        QMessageBox.information(self, "Información", "Trading automático detenido.")
        self.start_trading_button.setEnabled(True)
        self.stop_trading_button.setEnabled(False)

def main():
    app = QApplication(sys.argv)
    window = GGALTradingBotGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
