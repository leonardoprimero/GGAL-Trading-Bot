# GGAL Trading Bot

![GGAL Trading Bot](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Flogo.webp?alt=media&token=53ca6d46-6501-4207-bb7b-f8c26c8c0c72)

## Descripción

GGAL Trading Bot es una aplicación profesional de trading algorítmico diseñada específicamente para el ticker GGAL (Grupo Financiero Galicia). Utiliza técnicas de aprendizaje automático para predecir movimientos de precios y generar señales de compra/venta, permitiendo a los traders tomar decisiones informadas o automatizar sus estrategias.

La aplicación combina análisis técnico, machine learning y backtesting en una interfaz gráfica intuitiva y completa, facilitando tanto el análisis como la ejecución de operaciones.

## Características Principales

- **Análisis de Datos Históricos**: Descarga y visualización de datos históricos de GGAL.
- **Indicadores Técnicos**: Cálculo automático de RSI, volatilidad y cruces de medias móviles.
- **Modelo Predictivo**: Implementación de árboles de decisión para predecir movimientos de precios.
- **Backtesting**: Evaluación de estrategias con datos históricos y métricas de rendimiento.
- **Interfaz Gráfica**: Visualización intuitiva de datos, señales y resultados.
- **Trading en Vivo**: Generación de señales en tiempo real para tomar decisiones de trading.
- **Personalización**: Ajuste de parámetros del modelo y estrategias según preferencias.

## Capturas de Pantalla

![Pantalla de Datos](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Fdata_screen.webp?alt=media&token=d3e72699-11ae-4466-9b65-6d98c6ac967e)
*Visualización de datos históricos y análisis técnico*

![Pantalla de Modelo](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Fmodel_screen.webp?alt=media&token=afb40be9-4b22-48c5-a344-33f9aef02281)
*Entrenamiento y evaluación del modelo predictivo*

![Pantalla de Backtest](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Fbacktest_screen.webp?alt=media&token=dc80b555-d35a-4de4-981b-61c9122695c8)
*Resultados de backtesting y análisis de rendimiento*

![Pantalla de Trading](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Ftrading_screen.webp?alt=media&token=ad79e8a0-8fdb-4c22-9b02-0d8a9649978c)
*Interfaz de trading en vivo con señales en tiempo real*

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Dependencias

El proyecto requiere las siguientes bibliotecas:

```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
scikit-learn>=0.23.0
yfinance>=0.1.63
PyQt5>=5.15.0
joblib>=0.16.0
```

### Instalación Paso a Paso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/leoprimero/GGAL-Trading-Bot.git
   cd GGAL-Trading-Bot
   ```

2. Crear un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   
   # En Windows
   venv\Scripts\activate
   
   # En macOS/Linux
   source venv/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecutar la aplicación:
   ```bash
   python src/main.py
   ```

## Guía de Uso

### Carga de Datos

1. En la pestaña "Datos", ingrese el ticker (por defecto GGAL) y la fecha de inicio.
2. Haga clic en "Cargar Datos" para descargar y procesar los datos históricos.
3. Explore los datos en formato tabular o gráfico.

### Entrenamiento del Modelo

1. En la pestaña "Modelo", configure los parámetros del árbol de decisión:
   - Profundidad máxima: controla la complejidad del modelo
   - Criterio: método para evaluar divisiones (entropy o gini)
   - Tamaño de prueba: porcentaje de datos para validación
2. Haga clic en "Entrenar Modelo" para iniciar el entrenamiento.
3. Revise las métricas de rendimiento (precisión, recall, F1-score, etc.).
4. Opcionalmente, guarde el modelo para uso futuro.

### Backtesting

1. En la pestaña "Backtest", configure los parámetros de la estrategia:
   - Umbral de confianza: nivel mínimo de probabilidad para generar señales
   - Capital inicial: monto para simular operaciones
   - Tamaño de posición: porcentaje del capital a invertir en cada operación
2. Haga clic en "Ejecutar Backtest" para evaluar la estrategia.
3. Analice los resultados:
   - Métricas de rendimiento (retorno total, Sharpe ratio, etc.)
   - Gráfico de equity (evolución del capital)
   - Visualización de señales en el gráfico de precios

### Trading en Vivo

1. En la pestaña "Trading en Vivo", haga clic en "Actualizar Datos" para obtener la información más reciente.
2. Revise la última predicción y señal generada.
3. Configure el modo automático y el intervalo de actualización si desea automatizar el proceso.
4. Utilice los botones "Iniciar Trading" y "Detener Trading" para controlar la ejecución.
5. Consulte el registro de operaciones para ver el historial de señales y acciones.

## Estructura del Proyecto

```
GGAL-Trading-Bot/
│
├── docs/                      # Documentación
│   └── images/                # Imágenes para documentación
│
├── notebooks/                 # Jupyter notebooks para análisis exploratorio
│
├── src/                       # Código fuente
│   ├── data/                  # Módulos para manejo de datos
│   │   └── data_loader.py     # Carga y procesamiento de datos
│   │
│   ├── models/                # Módulos para modelos predictivos
│   │   └── model_trainer.py   # Entrenamiento y evaluación de modelos
│   │
│   ├── strategies/            # Módulos para estrategias de trading
│   │   └── trading_strategy.py # Implementación de estrategias
│   │
│   ├── gui/                   # Interfaz gráfica
│   │   └── main_app.py        # Aplicación principal
│   │
│   ├── utils/                 # Utilidades generales
│   │
│   └── main.py                # Punto de entrada principal
│
├── tests/                     # Pruebas unitarias y de integración
│
├── .gitignore                 # Archivos ignorados por git
├── LICENSE                    # Licencia del proyecto
├── README.md                  # Este archivo
└── requirements.txt           # Dependencias del proyecto
```

## Fundamentos Técnicos

### Indicadores Utilizados

- **RSI (Relative Strength Index)**: Mide la velocidad y cambio de los movimientos de precios.
- **Volatilidad**: Calculada como desviación estándar de los retornos diarios.
- **Cruces de Medias Móviles**: Relaciones entre medias móviles de diferentes períodos.

### Modelo Predictivo

El bot utiliza un árbol de decisión para clasificar los movimientos futuros del precio en dos categorías:
- **Clase 1**: El precio subirá en el horizonte temporal definido.
- **Clase 0**: El precio bajará en el horizonte temporal definido.

Los árboles de decisión son particularmente útiles en trading por su interpretabilidad y capacidad para capturar relaciones no lineales entre indicadores técnicos.

### Estrategia de Trading

La estrategia básica implementada sigue estas reglas:
1. **Señal de Compra**: Cuando el modelo predice subida (clase 1) con probabilidad superior al umbral configurado.
2. **Señal de Venta**: Cuando el modelo predice bajada (clase 0) con probabilidad superior al umbral configurado.
3. **Sin Acción**: Cuando la probabilidad está por debajo del umbral de confianza.

## Personalización y Extensión

### Ajuste de Parámetros

Los principales parámetros que puede ajustar para optimizar el rendimiento son:

- **Ventana de Predicción**: Modifique la variable `ventana` en `data_loader.py` para cambiar el horizonte temporal de predicción.
- **Profundidad del Árbol**: Ajuste `max_depth` en la interfaz para controlar la complejidad del modelo.
- **Umbral de Confianza**: Modifique este valor para ser más o menos conservador en la generación de señales.

### Añadir Nuevos Indicadores

Para incorporar nuevos indicadores técnicos:

1. Modifique la clase `DataLoader` en `data_loader.py` para calcular el nuevo indicador.
2. Asegúrese de incluir el nuevo indicador en el conjunto de características para el modelo.

### Implementar Nuevas Estrategias

Para crear estrategias alternativas:

1. Extienda la clase `TradingStrategy` en `trading_strategy.py` o cree una nueva clase que implemente la lógica deseada.
2. Modifique la interfaz gráfica para permitir la selección entre diferentes estrategias.

## Consideraciones de Riesgo

Este software se proporciona únicamente con fines educativos e informativos. Tenga en cuenta:

- **No es Asesoramiento Financiero**: Las señales generadas no constituyen recomendaciones de inversión.
- **Riesgo de Mercado**: El trading de acciones implica riesgo de pérdida de capital.
- **Limitaciones del Modelo**: Los modelos predictivos tienen limitaciones y no pueden anticipar eventos extraordinarios del mercado.
- **Validación**: Siempre realice backtesting exhaustivo y validación antes de utilizar cualquier estrategia con capital real.

## Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haga un fork del repositorio
2. Cree una rama para su funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Realice sus cambios y haga commit (`git commit -m 'Añadir nueva funcionalidad'`)
4. Haga push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abra un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para más detalles.

## Autor

**Leonardo Caliva** - [@leoprimero](https://github.com/leoprimero)

## Agradecimientos

- Agradecimiento especial a la comunidad de trading algorítmico y machine learning.
- Basado en conceptos de aprendizaje supervisado y algoritmos de clasificación para trading.
