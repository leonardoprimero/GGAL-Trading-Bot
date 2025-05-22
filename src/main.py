import sys
import os

# Añadir directorio src al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.main_app import main

if __name__ == "__main__":
    main()
