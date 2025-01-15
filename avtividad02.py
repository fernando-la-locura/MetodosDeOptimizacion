import matplotlib.pyplot as plt
import numpy as np

# 1. Precio de vivienda: P = mA + b
def precio_vivienda(A, m=1000, b=50000):
    return m * A + b

def graficar_precio_vivienda():
    A = np.linspace(0, 200, 100)
    P = precio_vivienda(A)
    plt.figure(figsize=(10, 6))
    plt.plot(A, P)
    plt.title('Precio de Vivienda vs Área Construida')
    plt.xlabel('Área construida (A)')
    plt.ylabel('Precio (P)')
    plt.grid(True)
    
    print("\nPrecio de vivienda para diferentes áreas:")
    for area in [50, 100, 150, 200]:
        print(f"Área: {area}m² -> Precio: ${precio_vivienda(area):,.2f}")

# 2. Ganancia mensual: G = aN - b
def ganancia_mensual(N, a=50, b=1000):
    return a * N - b

def graficar_ganancia_mensual():
    N = np.linspace(0, 100, 100)
    G = ganancia_mensual(N)
    plt.figure(figsize=(10, 6))
    plt.plot(N, G)
    plt.title('Ganancia Mensual vs Número de Predicciones')
    plt.xlabel('Número de predicciones (N)')
    plt.ylabel('Ganancia (G)')
    plt.grid(True)
    
    print("\nGanancia mensual para diferentes predicciones:")
    for pred in [20, 40, 60, 80, 100]:
        print(f"Predicciones: {pred} -> Ganancia: ${ganancia_mensual(pred):,.2f}")

# 3. Tiempo de procesamiento: T = kD + c
def tiempo_procesamiento(D, k=0.1, c=2):
    return k * D + c

def graficar_tiempo_procesamiento():
    D = np.linspace(0, 1000, 100)
    T = tiempo_procesamiento(D)
    plt.figure(figsize=(10, 6))
    plt.plot(D, T)
    plt.title('Tiempo de Procesamiento vs Tamaño de Datos')
    plt.xlabel('Tamaño de datos (D)')
    plt.ylabel('Tiempo (T)')
    plt.grid(True)
    
    print("\nTiempo de procesamiento para diferentes tamaños de datos:")
    for datos in [200, 400, 600, 800, 1000]:
        print(f"Datos: {datos} -> Tiempo: {tiempo_procesamiento(datos):.2f} segundos")

# 4. Costo de almacenamiento: C = pD + f
def costo_almacenamiento(D, p=0.5, f=100):
    return p * D + f

def graficar_costo_almacenamiento():
    D = np.linspace(0, 1000, 100)
    C = costo_almacenamiento(D)
    plt.figure(figsize=(10, 6))
    plt.plot(D, C)
    plt.title('Costo de Almacenamiento vs Cantidad de Datos')
    plt.xlabel('Cantidad de datos (D)')
    plt.ylabel('Costo (C)')
    plt.grid(True)
    
    print("\nCosto de almacenamiento para diferentes cantidades de datos:")
    for datos in [200, 400, 600, 800, 1000]:
        print(f"Datos: {datos}GB -> Costo: ${costo_almacenamiento(datos):.2f}")

# 5. Medición calibrada: M = aR + b
def medicion_calibrada(R, a=1.2, b=0.5):
    return a * R + b

def graficar_medicion_calibrada():
    R = np.linspace(0, 100, 100)
    M = medicion_calibrada(R)
    plt.figure(figsize=(10, 6))
    plt.plot(R, M)
    plt.title('Medición Calibrada vs Medición en Crudo')
    plt.xlabel('Medición en crudo (R)')
    plt.ylabel('Medición calibrada (M)')
    plt.grid(True)
    
    print("\nMedición calibrada para diferentes valores en crudo:")
    for valor in [20, 40, 60, 80, 100]:
        print(f"Valor crudo: {valor} -> Valor calibrado: {medicion_calibrada(valor):.2f}")

# 6. Tiempo de respuesta: T = mS + b
def tiempo_respuesta(S, m=0.05, b=1):
    return m * S + b

def graficar_tiempo_respuesta():
    S = np.linspace(0, 100, 100)
    T = tiempo_respuesta(S)
    plt.figure(figsize=(10, 6))
    plt.plot(S, T)
    plt.title('Tiempo de Respuesta vs Solicitudes Simultáneas')
    plt.xlabel('Solicitudes simultáneas (S)')
    plt.ylabel('Tiempo de respuesta (T)')
    plt.grid(True)
    
    print("\nTiempo de respuesta para diferentes solicitudes simultáneas:")
    for sol in [20, 40, 60, 80, 100]:
        print(f"Solicitudes: {sol} -> Tiempo: {tiempo_respuesta(sol):.2f} segundos")

# 7. Ingresos de plataforma: I = pS + b
def ingresos_plataforma(S, p=10, b=500):
    return p * S + b

def graficar_ingresos_plataforma():
    S = np.linspace(0, 1000, 100)
    I = ingresos_plataforma(S)
    plt.figure(figsize=(10, 6))
    plt.plot(S, I)
    plt.title('Ingresos vs Número de Suscriptores')
    plt.xlabel('Número de suscriptores (S)')
    plt.ylabel('Ingresos (I)')
    plt.grid(True)
    
    print("\nIngresos para diferentes números de suscriptores:")
    for susc in [200, 400, 600, 800, 1000]:
        print(f"Suscriptores: {susc} -> Ingresos: ${ingresos_plataforma(susc):,.2f}")

# 8. Energía consumida: E = kO + b
def energia_consumida(O, k=0.2, b=50):
    return k * O + b

def graficar_energia_consumida():
    O = np.linspace(0, 500, 100)
    E = energia_consumida(O)
    plt.figure(figsize=(10, 6))
    plt.plot(O, E)
    plt.title('Energía Consumida vs Operaciones Realizadas')
    plt.xlabel('Operaciones realizadas (O)')
    plt.ylabel('Energía consumida (E)')
    plt.grid(True)
    
    print("\nEnergía consumida para diferentes números de operaciones:")
    for ops in [100, 200, 300, 400, 500]:
        print(f"Operaciones: {ops} -> Energía: {energia_consumida(ops):.2f} unidades")

# 9. Número de likes: L = mF + b
def numero_likes(F, m=0.3, b=10):
    return m * F + b

def graficar_numero_likes():
    F = np.linspace(0, 1000, 100)
    L = numero_likes(F)
    plt.figure(figsize=(10, 6))
    plt.plot(F, L)
    plt.title('Número de Likes vs Número de Seguidores')
    plt.xlabel('Número de seguidores (F)')
    plt.ylabel('Número de likes (L)')
    plt.grid(True)
    
    print("\nNúmero de likes para diferentes cantidades de seguidores:")
    for seg in [200, 400, 600, 800, 1000]:
        print(f"Seguidores: {seg} -> Likes esperados: {numero_likes(seg):.2f}")

# 10. Costo de entrenamiento ML: C = pI + c
def costo_entrenamiento(I, p=2, c=1000):
    return p * I + c

def graficar_costo_entrenamiento():
    I = np.linspace(0, 1000, 100)
    C = costo_entrenamiento(I)
    plt.figure(figsize=(10, 6))
    plt.plot(I, C)
    plt.title('Costo de Entrenamiento vs Número de Iteraciones')
    plt.xlabel('Número de iteraciones (I)')
    plt.ylabel('Costo total (C)')
    plt.grid(True)
    
    print("\nCosto de entrenamiento para diferentes números de iteraciones:")
    for iter in [200, 400, 600, 800, 1000]:
        print(f"Iteraciones: {iter} -> Costo: ${costo_entrenamiento(iter):,.2f}")

# Ejecutar todas las gráficas
if __name__ == "__main__":
    graficar_precio_vivienda()
    graficar_ganancia_mensual()
    graficar_tiempo_procesamiento()
    graficar_costo_almacenamiento()
    graficar_medicion_calibrada()
    graficar_tiempo_respuesta()
    graficar_ingresos_plataforma()
    graficar_energia_consumida()
    graficar_numero_likes()
    graficar_costo_entrenamiento()
    plt.show()