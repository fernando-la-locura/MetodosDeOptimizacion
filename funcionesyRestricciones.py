# EJERCICIO 03 = Administración de Tiempo en Proyectos
import numpy as np
import matplotlib.pyplot as plt

def analizar_tiempo_proyecto():
    def plot_region_factible():
        x = np.linspace(0, 12, 100)
        y1 = np.full_like(x, 6)  # y ≥ 6
        y2 = 12 - x  # x + y ≤ 12
        
        plt.figure(figsize=(10, 8))
        plt.plot(x, y1, 'b-', label='y ≥ 6')
        plt.plot(x, y2, 'r-', label='x + y ≤ 12')
        plt.axvline(x=4, color='g', label='x ≥ 4')
        
        # Región factible
        y_factible = np.maximum(y1, 0)
        plt.fill_between(x[x >= 4], y_factible[x >= 4], 
                        y2[x >= 4], alpha=0.3)
        
        plt.grid(True)
        plt.xlabel('Reuniones (x)')
        plt.ylabel('Documentación (y)')
        plt.title('Región Factible')
        plt.legend()
        plt.show()
    
    return plot_region_factible()
analizar_tiempo_proyecto()


# EJERCICIO 04 =  Producción de Assets para Videojuegos

from scipy.optimize import linprog

def optimizar_produccion_assets():
    # Coeficientes de la función objetivo
    c = [-1, -1]  # Negativo para maximización
    
    # Restricciones
    A = [[2, 3]]  # 2P1 + 3P2 ≤ 18
    b = [18]
    
    # Límites
    bounds = [(0, None), (0, None)]
    
    # Resolver
    resultado = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='simplex')
    
    return {
        'modelos_3d': resultado.x[0],
        'texturas': resultado.x[1],
        'valor_maximo': -resultado.fun
    }

# Llamada a la optimización
resultados = optimizar_produccion_assets()
print("Modelos 3D:", resultados['modelos_3d'])
print("Texturas:", resultados['texturas'])
print("Valor máximo:", resultados['valor_maximo'])

# EJERCICIO 05 =  Producción de Hardware

import numpy as np
import matplotlib.pyplot as plt

def optimizar_produccion_hardware():
    def analizar_combinaciones():
        # Puntos extremos
        max_a = 50 / 5  # Máximo dispositivos A
        max_b = 50 / 10  # Máximo dispositivos B
        
        # Generar combinaciones factibles
        combinaciones = []
        for a in range(int(max_a) + 1):
            for b in range(int(max_b) + 1):
                if 5*a + 10*b <= 50:
                    combinaciones.append((a, b))
        
        return {
            'max_tipo_a': max_a,
            'max_tipo_b': max_b,
            'combinaciones': combinaciones
        }
    
    return analizar_combinaciones()

def plot_region_hardware():
    x = np.linspace(0, 10, 100)
    y = (50 - 5*x) / 10
    
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, 'r-', label='5A + 10B = 50')
    plt.fill_between(x, 0, y, where=y >= 0, alpha=0.3)
    plt.grid(True)
    plt.xlabel('Dispositivos A')
    plt.ylabel('Dispositivos B')
    plt.title('Región Factible - Hardware')
    plt.legend()
    plt.show()

# Ejecución y visualización
resultado = optimizar_produccion_hardware()
print("Máximo tipo A:", resultado['max_tipo_a'])
print("Máximo tipo B:", resultado['max_tipo_b'])
print("Combinaciones factibles:", resultado['combinaciones'])

# Gráfico
plot_region_hardware()
