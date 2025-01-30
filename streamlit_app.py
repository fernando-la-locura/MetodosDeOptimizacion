import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import numpy as np

def calculate_second_derivative(expr_str):
    """Calcula la segunda derivada de una expresión."""
    x = sp.Symbol('x')
    try:
        expr = parse_expr(expr_str)
        first_deriv = sp.diff(expr, x)
        second_deriv = sp.diff(first_deriv, x)
        return str(second_deriv)
    except:
        return None

def is_convex(second_deriv_str, x_range):
    """Verifica si la función es convexa en el rango dado."""
    if second_deriv_str is None:
        return None
    
    x = sp.Symbol('x')
    second_deriv = parse_expr(second_deriv_str)
    
    # Evaluar la segunda derivada en varios puntos del rango
    x_points = np.linspace(x_range[0], x_range[1], 100)
    try:
        values = [float(second_deriv.subs(x, xi)) for xi in x_points]
        return all(v >= -1e-10 for v in values)  # Consideramos valores muy cercanos a 0 como positivos
    except:
        return None

def evaluate_function(expr_str, x_values):
    """Evalúa la función para un conjunto de valores x."""
    x = sp.Symbol('x')
    expr = parse_expr(expr_str)
    return [float(expr.subs(x, xi)) for xi in x_values]

def plot_function_and_chord(expr_str, x_range, x1, x2):
    """Crea una gráfica de la función y la cuerda entre dos puntos."""
    x_values = np.linspace(x_range[0], x_range[1], 200)
    y_values = evaluate_function(expr_str, x_values)
    
    # Calcular puntos de la cuerda
    y1 = evaluate_function(expr_str, [x1])[0]
    y2 = evaluate_function(expr_str, [x2])[0]
    
    fig = go.Figure()
    
    # Gráfica de la función
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        name='f(x)',
        line=dict(color='blue')
    ))
    
    # Gráfica de la cuerda
    fig.add_trace(go.Scatter(
        x=[x1, x2],
        y=[y1, y2],
        name='Cuerda',
        line=dict(color='red', dash='dash')
    ))
    
    # Configuración del layout
    fig.update_layout(
        title='Función y Cuerda',
        xaxis_title='x',
        yaxis_title='y',
        showlegend=True,
        height=500
    )
    
    return fig

def main():
    # Configuración de la página
    st.set_page_config(page_title="Analizador de Convexidad", layout="wide")

    # Título
    st.title("📊 Analizador de Convexidad de Funciones")

    # Descripción
    st.markdown("""
    Este analizador te ayuda a determinar si una función es convexa o no. Utiliza el análisis de la segunda
    derivada y la visualización de la función junto con sus cuerdas para mostrar la convexidad.
    """)

    # Input de la función
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ingrese la función")
        function_input = st.text_input(
            "Función f(x)",
            value="x**2",
            help="Ingrese una función en términos de x. Ejemplo: x**2, exp(x), log(x)"
        )
        
        # Rango de x
        st.subheader("Rango de análisis")
        x_min = st.number_input("x mínimo", value=-5.0)
        x_max = st.number_input("x máximo", value=5.0)
        
        # Puntos para la cuerda
        st.subheader("Puntos para la cuerda")
        x1 = st.slider("x₁", min_value=float(x_min), max_value=float(x_max), value=float(x_min+1))
        x2 = st.slider("x₂", min_value=float(x_min), max_value=float(x_max), value=float(x_max-1))

    with col2:
        st.subheader("Ejemplos de funciones")
        st.markdown("""
        - `x**2` (función cuadrática)
        - `x**3` (función cúbica)
        - `exp(x)` (función exponencial)
        - `log(x)` (logaritmo natural)
        - `sin(x)` (función seno)
        """)

    # Análisis al presionar el botón
    if st.button("Analizar Convexidad", type="primary"):
        try:
            # Calcular segunda derivada
            second_deriv = calculate_second_derivative(function_input)
            
            if second_deriv is not None:
                st.subheader("Análisis de convexidad")
                
                # Mostrar segunda derivada
                st.write(f"Segunda derivada: f''(x) = {second_deriv}")
                
                # Verificar convexidad
                is_convex_result = is_convex(second_deriv, [x_min, x_max])
                
                if is_convex_result is not None:
                    if is_convex_result:
                        st.success("✅ La función es convexa en el rango especificado")
                    else:
                        st.warning("⚠️ La función no es convexa en el rango especificado")
                    
                    # Mostrar gráfica
                    st.subheader("Visualización")
                    fig = plot_function_and_chord(function_input, [x_min, x_max], x1, x2)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explicación
                    st.subheader("Interpretación")
                    st.markdown("""
                    - La línea azul muestra la función f(x)
                    - La línea roja punteada muestra la cuerda entre los puntos seleccionados
                    - Para una función convexa, la gráfica de la función debe estar por debajo o 
                      coincidir con cualquier cuerda que une dos puntos de la función
                    """)
                else:
                    st.error("No se pudo determinar la convexidad en el rango especificado")
            else:
                st.error("No se pudo calcular la segunda derivada")
                
        except Exception as e:
            st.error(f"Error al analizar la función: {str(e)}")

if __name__ == "__main__":
    main()