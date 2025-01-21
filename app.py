import streamlit as st
import numpy as np
from typing import List, Tuple

def parse_equations(equations_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte el texto de las ecuaciones en matrices A y b
    """
    lines = equations_text.strip().split('\n')
    n = len(lines)
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i, line in enumerate(lines):
        # Eliminar espacios y dividir por '='
        left, right = line.replace(' ', '').split('=')
        
        # Procesar el lado derecho
        b[i] = float(right)
        
        # Procesar el lado izquierdo
        terms = left.replace('-', '+-').split('+')
        for term in terms:
            if term:
                if 'x' in term:
                    var_idx = int(term.split('x')[1]) - 1
                    coef = term.split('x')[0]
                    A[i, var_idx] = float(coef if coef not in ['', '-'] else '-1' if coef == '-' else '1')

    return A, b

def cramer(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resuelve el sistema usando el método de Cramer
    """
    n = len(b)
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-10:
        raise ValueError("El determinante es cero. El sistema no tiene solución única.")
    
    x = np.zeros(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / det_A
    
    return x

def gauss_jordan(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resuelve el sistema usando el método de Gauss-Jordan
    """
    n = len(b)
    # Crear matriz aumentada
    Ab = np.column_stack((A, b))
    
    # Eliminación hacia adelante
    for i in range(n):
        # Pivoteo parcial
        max_element = abs(Ab[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(Ab[k][i]) > max_element:
                max_element = abs(Ab[k][i])
                max_row = k
        
        if max_element == 0:
            raise ValueError("El sistema no tiene solución única.")
        
        # Intercambiar filas si es necesario
        if max_row != i:
            Ab[i], Ab[max_row] = Ab[max_row], Ab[i]
        
        # Hacer el elemento pivote igual a 1
        Ab[i] = Ab[i] / Ab[i][i]
        
        # Eliminar la variable en las otras ecuaciones
        for j in range(n):
            if i != j:
                Ab[j] = Ab[j] - Ab[i] * Ab[j][i]
    
    return Ab[:, -1]

def sustitucion(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resuelve el sistema usando el método de sustitución hacia atrás
    """
    n = len(b)
    x = np.zeros(n)
    
    # Convertir a matriz triangular superior
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError("Elemento diagonal es cero. El método no puede continuar.")
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]
    
    # Sustitución hacia atrás
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i][i]
    
    return x

# Configuración de la página
st.set_page_config(page_title="Resolutor de Sistemas de Ecuaciones", layout="wide")

# Título
st.title("Resolutor de Sistemas de Ecuaciones Lineales")

# Instrucciones
st.markdown("""
### Instrucciones:
1. Ingrese sus ecuaciones una por línea
2. Use el formato: ax1 + bx2 + cx3 = d
3. Ejemplo:
```
2x1 + 1x2 + 4x3 = 7
3x1 - 2x2 + 1x3 = 4
1x1 + 5x2 - 2x3 = -2
```
""")

# Área de texto para ingresar ecuaciones
equations = st.text_area("Ingrese sus ecuaciones:", height=150)

if st.button("Resolver"):
    try:
        # Parsear ecuaciones
        A, b = parse_equations(equations)
        
        # Mostrar el sistema en forma matricial
        st.subheader("Sistema en forma matricial:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Matriz A:")
            st.write(A)
        with col2:
            st.write("Vector b:")
            st.write(b)
        
        # Resolver usando diferentes métodos
        methods = {
            "Método de Cramer": cramer,
            "Método de Gauss-Jordan": gauss_jordan,
            "Método de Sustitución": sustitucion
        }
        
        st.subheader("Soluciones:")
        for method_name, method_func in methods.items():
            try:
                solution = method_func(A.copy(), b.copy())
                st.write(f"\n{method_name}:")
                for i, xi in enumerate(solution):
                    st.write(f"x{i+1} = {xi:.4f}")
            except Exception as e:
                st.error(f"{method_name}: {str(e)}")
                
    except Exception as e:
        st.error(f"Error al procesar las ecuaciones: {str(e)}")
