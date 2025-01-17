import tkinter as tk
from tkinter import messagebox
import numpy as np

def resolver_gauss_jordan():
    try:
        # Obtener el número de ecuaciones
        n = int(entry_num_ecuaciones.get())
        if n <= 0:
            raise ValueError("El número de ecuaciones debe ser mayor a 0.")
        
        # Leer los coeficientes de la matriz aumentada desde la entrada
        entradas = text_matriz.get("1.0", tk.END).strip().split("\n")
        matriz = []
        for linea in entradas:
            fila = list(map(float, linea.split()))
            if len(fila) != n + 1:
                raise ValueError("Cada fila debe tener exactamente n+1 valores (coeficientes + término independiente).")
            matriz.append(fila)
        matriz = np.array(matriz, dtype=float)
        
        # Realizar el método de Gauss-Jordan
        salida = []
        salida.append("# Matriz aumentada inicial\n" + str(matriz) + "\n")
        for i in range(n):
            pivote = matriz[i][i]
            if pivote == 0:
                raise ValueError(f"No se puede continuar, el pivote en la posición ({i+1},{i+1}) es cero.")
            matriz[i] = matriz[i] / pivote
            salida.append(f"# Normalizar fila {i+1}\n{matriz}\n")
            for j in range(n):
                if i != j:
                    factor = matriz[j][i]
                    matriz[j] = matriz[j] - factor * matriz[i]
                    salida.append(f"# Usar fila {i+1} para hacer cero el elemento en la posición ({j+1},{i+1})\n{matriz}\n")

        salida.append("# Matriz en forma escalonada reducida\n" + str(matriz) + "\n")
        soluciones = [f"x{i+1} = {matriz[i][-1]:.2f}" for i in range(n)]
        salida.append("Soluciones:\n" + "\n".join(soluciones))

        # Mostrar la salida en la interfaz
        text_salida.delete("1.0", tk.END)
        text_salida.insert(tk.END, "\n".join(salida))

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Configuración de la ventana principal
root = tk.Tk()
root.title("Método de Gauss-Jordan")
root.geometry("700x500")

# Widgets para el número de ecuaciones
frame_top = tk.Frame(root)
frame_top.pack(pady=10)
tk.Label(frame_top, text="Número de ecuaciones:").pack(side=tk.LEFT, padx=5)
entry_num_ecuaciones = tk.Entry(frame_top, width=5)
entry_num_ecuaciones.pack(side=tk.LEFT, padx=5)

# Widgets para ingresar la matriz aumentada
frame_matriz = tk.Frame(root)
frame_matriz.pack(pady=10)
tk.Label(frame_matriz, text="Ingrese la matriz aumentada (una fila por línea):").pack(anchor="w")
text_matriz = tk.Text(frame_matriz, width=80, height=10)
text_matriz.pack()

# Botón para resolver
btn_resolver = tk.Button(root, text="Resolver", command=resolver_gauss_jordan, bg="lightblue")
btn_resolver.pack(pady=10)

# Widgets para mostrar la salida
frame_salida = tk.Frame(root)
frame_salida.pack(pady=10)
tk.Label(frame_salida, text="Salida:").pack(anchor="w")
text_salida = tk.Text(frame_salida, width=80, height=15)
text_salida.pack()

# Ejecutar la aplicación
root.mainloop()