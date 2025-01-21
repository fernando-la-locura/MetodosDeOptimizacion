import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple
import base64
from io import BytesIO

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

def cramer(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Resuelve el sistema usando el método de Cramer
    """
    n = len(b)
    det_A = np.linalg.det(A)
    steps = []
    matrices = [A.copy()]
    steps.append(f"Determinante de A: {det_A:.4f}")
    
    if abs(det_A) < 1e-10:
        raise ValueError("El determinante es cero. El sistema no tiene solución única.")
    
    x = np.zeros(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        matrices.append(Ai)
        det_Ai = np.linalg.det(Ai)
        x[i] = det_Ai / det_A
        steps.append(f"Determinante de A{i+1}: {det_Ai:.4f}")
        steps.append(f"x{i+1} = {det_Ai:.4f} / {det_A:.4f} = {x[i]:.4f}")
    
    return x, steps, matrices

def gauss_jordan(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Resuelve el sistema usando el método de Gauss-Jordan
    """
    n = len(b)
    Ab = np.column_stack((A, b))
    steps = []
    matrices = [Ab.copy()]
    
    for i in range(n):
        # Pivoteo parcial
        max_element = abs(Ab[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(Ab[k][i]) > max_element:
                max_element = abs(Ab[k][i])
                max_row = k
        
        if max_element < 1e-10:
            raise ValueError("El sistema no tiene solución única.")
        
        # Intercambiar filas si es necesario
        if max_row != i:
            Ab[i], Ab[max_row] = Ab[max_row].copy(), Ab[i].copy()
            steps.append(f"Intercambio de filas {i+1} y {max_row+1}")
            matrices.append(Ab.copy())
        
        # Hacer el elemento pivote igual a 1
        pivot = Ab[i][i]
        Ab[i] = Ab[i] / pivot
        steps.append(f"Normalización de la fila {i+1}")
        matrices.append(Ab.copy())
        
        # Eliminar la variable en las otras ecuaciones
        for j in range(n):
            if i != j:
                factor = Ab[j][i]
                Ab[j] = Ab[j] - factor * Ab[i]
                steps.append(f"Eliminación en fila {j+1}")
                matrices.append(Ab.copy())
    
    return Ab[:, -1], steps, matrices

def sustitucion(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """
    Resuelve el sistema usando el método de sustitución hacia atrás
    """
    n = len(b)
    Ab = np.column_stack((A, b))
    steps = []
    matrices = [Ab.copy()]
    
    # Convertir a matriz triangular superior
    for i in range(n):
        if abs(A[i][i]) < 1e-10:
            raise ValueError("Elemento diagonal es cero. El método no puede continuar.")
        for j in range(i + 1, n):
            factor = Ab[j][i] / Ab[i][i]
            Ab[j] = Ab[j] - factor * Ab[i]
            steps.append(f"Eliminación en fila {j+1}")
            matrices.append(Ab.copy())
    
    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i][-1] - np.dot(Ab[i, i+1:-1], x[i+1:])) / Ab[i][i]
        steps.append(f"x{i+1} = {x[i]:.4f}")
    
    return x, steps, matrices

def plot_matrix(matrix: np.ndarray, title: str, key_suffix: str = "") -> None:
    """
    Crea una visualización de la matriz usando plotly con un identificador único
    """
    fig = go.Figure(data=[go.Heatmap(
        z=matrix,
        text=[[f"{val:.4f}" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale="RdBu",
        showscale=False
    )])
    
    fig.update_layout(
        title=title,
        height=300,
        width=400,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig, key=f"matrix_plot_{title}_{key_suffix}")

def download_results(results_df: pd.DataFrame) -> str:
    """
    Genera un enlace de descarga para los resultados en formato CSV
    """
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'data:file/csv;base64,{b64}'

# Configuración de la página
st.set_page_config(
    page_title="Resolutor de Sistemas de Ecuaciones",
    page_icon="📊",
    layout="wide"
)

# Título y descripción
st.title("📊 Resolutor de Sistemas de Ecuaciones Lineales")
st.markdown("""
Esta aplicación resuelve sistemas de ecuaciones lineales utilizando tres métodos diferentes:
- Método de Cramer
- Método de Gauss-Jordan
- Método de Sustitución

Ingrese sus ecuaciones en el formato especificado y compare los resultados de los diferentes métodos.
""")

# Sidebar con instrucciones
with st.sidebar:
    st.header("Instrucciones")
    st.markdown("""
    1. Ingrese cada ecuación en una nueva línea
    2. Use el formato: ax1 + bx2 + cx3 = d
    3. Ejemplo:
        ```
        2x1 + 3x2 = 18
        x1 - x2 = 1
        ```
    4. Los coeficientes pueden ser números enteros o decimales
    5. Use signos + y - para los términos
    """)
    
    # Ejemplos predefinidos
    st.header("Ejemplos")
    if st.button("Cargar ejemplo 2x2"):
        example = """2x1 + 3x2 = 18
x1 - x2 = 1"""
        st.session_state['equations'] = example
    
    if st.button("Cargar ejemplo 3x3"):
        example = """x1 + 2x2 + x3 = 6
2x1 + 3x2 + x3 = 11
x2 + 2x3 = 5"""
        st.session_state['equations'] = example

# Área principal
col1, col2 = st.columns([2, 1])

with col1:
    # Entrada de ecuaciones
    if 'equations' not in st.session_state:
        st.session_state['equations'] = ""
    
    equations = st.text_area(
        "Ingrese sus ecuaciones:",
        value=st.session_state['equations'],
        height=150,
        key="equation_input"
    )

    if st.button("Resolver", type="primary"):
        try:
            # Parsear ecuaciones
            A, b = parse_equations(equations)
            
            # Mostrar sistema en forma matricial
            st.subheader("Sistema en forma matricial")
            col_matrix1, col_matrix2 = st.columns(2)
            
            with col_matrix1:
                plot_matrix(A, "Matriz A", "initial_A")
            
            with col_matrix2:
                plot_matrix(b.reshape(-1, 1), "Vector b", "initial_b")
            
            # Resolver usando diferentes métodos
            methods = {
                "Método de Cramer": cramer,
                "Método de Gauss-Jordan": gauss_jordan,
                "Método de Sustitución": sustitucion
            }
            
            results = []
            
            # Crear pestañas para cada método
            tabs = st.tabs(list(methods.keys()))
            
            for tab_idx, (tab, (method_name, method_func)) in enumerate(zip(tabs, methods.items())):
                with tab:
                    try:
                        solution, steps, matrices = method_func(A.copy(), b.copy())
                        
                        # Mostrar solución
                        st.subheader("Solución")
                        for i, xi in enumerate(solution):
                            st.write(f"x{i+1} = {xi:.4f}")
                            results.append({
                                "Método": method_name,
                                "Variable": f"x{i+1}",
                                "Valor": xi
                            })
                        
                        # Mostrar pasos
                        with st.expander("Ver pasos"):
                            for step_idx, (step, matrix) in enumerate(zip(steps, matrices)):
                                st.write(f"Paso {step_idx+1}: {step}")
                                plot_matrix(matrix, f"Matriz - Paso {step_idx+1}", 
                                          f"method_{tab_idx}_step_{step_idx}")
                    
                    except Exception as e:
                        st.error(f"{method_name}: {str(e)}")
            
            # Crear DataFrame con resultados
            if results:
                results_df = pd.DataFrame(results)
                
                # Mostrar tabla comparativa
                st.subheader("Comparación de resultados")
                st.dataframe(results_df.pivot(index='Variable', columns='Método', values='Valor'))
                
                # Botón de descarga
                st.download_button(
                    label="Descargar resultados (CSV)",
                    data=download_results(results_df),
                    file_name="resultados_ecuaciones.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error al procesar las ecuaciones: {str(e)}")

with col2:
    # Información adicional y ayuda
    st.header("Características")
    st.markdown("""
    - Resolución paso a paso
    - Visualización matricial
    - Comparación de métodos
    - Exportación de resultados
    - Ejemplos predefinidos
    """)
    
    st.header("Limitaciones")
    st.markdown("""
    - El sistema debe tener el mismo número de ecuaciones que de incógnitas
    - El determinante de la matriz de coeficientes no debe ser cero
    - Los coeficientes deben ser números válidos
    """)