# Ejercicio 1: Mínimo de una función cuadrática en 1D
# g(x) = (x - 5)^2

# Definimos la función objetivo y su derivada
g <- function(x) { (x - 5)^2 }
g_prime <- function(x) { 2 * (x - 5) }

# Parámetros iniciales
x <- 10  # Punto inicial
eta <- 0.2  # Tasa de aprendizaje
iteraciones <- 5

# Almacenamos las iteraciones
resultados <- data.frame(k = 0, xk = x, g_xk = g(x))

# Iteraciones del método de descenso del gradiente
for (k in 1:iteraciones) {
  gradiente <- g_prime(x)
  x <- x - eta * gradiente
  resultados <- rbind(resultados, data.frame(k = k, xk = x, g_xk = g(x)))
}

# Visualización
print(resultados)

# Gráfico detallado
curve(g(x), from = 0, to = 10, col = "blue", lwd = 2, ylab = "g(x)", 
      main = "Descenso del Gradiente para g(x)", xlab = "x")
points(resultados$xk, resultados$g_xk, col = "red", pch = 19)
lines(resultados$xk, resultados$g_xk, col = "red", lty = 2)
text(resultados$xk, resultados$g_xk, labels = paste0("k=", resultados$k), 
     pos = 4, cex = 0.8, col = "darkgreen")
grid()


# Ejercicio 2: Ajuste de recta por mínimos cuadrados
# Datos
data <- data.frame(x = c(1, 2, 3, 4, 5), y = c(2, 2.8, 3.6, 4.5, 5.1))

# Función de costo
J <- function(beta0, beta1) {
  sum((data$y - (beta0 + beta1 * data$x))^2)
}

# Gradientes parciales
grad_beta0 <- function(beta0, beta1) {
  -2 * sum(data$y - (beta0 + beta1 * data$x))
}
grad_beta1 <- function(beta0, beta1) {
  -2 * sum((data$y - (beta0 + beta1 * data$x)) * data$x)
}

# Parámetros iniciales
beta0 <- 0
beta1 <- 0
eta <- 0.01
iteraciones <- 3

# Almacenamos las iteraciones
resultados <- data.frame(iteracion = 0, beta0 = beta0, beta1 = beta1, costo = J(beta0, beta1))

for (i in 1:iteraciones) {
  grad0 <- grad_beta0(beta0, beta1)
  grad1 <- grad_beta1(beta0, beta1)
  beta0 <- beta0 - eta * grad0
  beta1 <- beta1 - eta * grad1
  resultados <- rbind(resultados, data.frame(iteracion = i, beta0 = beta0, beta1 = beta1, costo = J(beta0, beta1)))
}

# Visualización
print(resultados)

# Gráfico de ajuste detallado
plot(data$x, data$y, main = "Ajuste por Descenso del Gradiente", col = "blue", pch = 19, xlab = "x", ylab = "y")
abline(a = beta0, b = beta1, col = "red", lwd = 2)
text(data$x, data$y, labels = round(data$y, 2), pos = 4, cex = 0.8, col = "darkgreen")
grid()


# Ejercicio 3: Clasificación logística
# Datos
x1 <- c(0.5, 1.5, 2.0, 3.0)
x2 <- c(1.0, 2.0, 2.5, 3.5)
y <- c(0, 0, 1, 1)
data <- data.frame(x1, x2, y)

# Modelo y funciones
sigmoid <- function(z) { 1 / (1 + exp(-z)) }
loss <- function(w) {
  -sum(data$y * log(sigmoid(w[1] + w[2] * data$x1 + w[3] * data$x2)) +
         (1 - data$y) * log(1 - sigmoid(w[1] + w[2] * data$x1 + w[3] * data$x2)))
}

# Gradientes
grad <- function(w) {
  preds <- sigmoid(w[1] + w[2] * data$x1 + w[3] * data$x2)
  c(sum(preds - data$y),
    sum((preds - data$y) * data$x1),
    sum((preds - data$y) * data$x2))
}

# Parámetros iniciales
w <- c(0, 0, 0)  # Incluye sesgo
eta <- 0.1
iteraciones <- 3

# Almacenamos las iteraciones
resultados <- data.frame(iteracion = 0, w1 = w[1], w2 = w[2], w3 = w[3], loss = loss(w))

for (i in 1:iteraciones) {
  g <- grad(w)
  w <- w - eta * g
  resultados <- rbind(resultados, data.frame(iteracion = i, w1 = w[1], w2 = w[2], w3 = w[3], loss = loss(w)))
}

# Visualización
print(resultados)

# Gráfico detallado de Clasificación Logística
plot(resultados$iteracion, resultados$loss, type = "o", col = "blue", lwd = 2, 
     xlab = "Iteración", ylab = "Loss", main = "Evolución de la Función de Costo")
text(resultados$iteracion, resultados$loss, labels = round(resultados$loss, 3), pos = 4, cex = 0.8, col = "darkgreen")
grid()


# Ejercicio 4: Descenso del Gradiente Estocástico (SGD) para regresión multivariable
# Generamos datos sintéticos para ilustración
set.seed(123)
N <- 1000
x <- matrix(rnorm(N * 2), ncol = 2)
y <- rowSums(x) + rnorm(N, sd = 0.5)
data <- data.frame(x1 = x[, 1], x2 = x[, 2], y)

# Función de costo
J <- function(w, data_batch) {
  mean((data_batch$y - (w[1] + w[2] * data_batch$x1 + w[3] * data_batch$x2))^2)
}

# Gradientes parciales para un minibatch
grad <- function(w, data_batch) {
  preds <- w[1] + w[2] * data_batch$x1 + w[3] * data_batch$x2
  errors <- preds - data_batch$y
  c(mean(errors), mean(errors * data_batch$x1), mean(errors * data_batch$x2))
}

# Parámetros iniciales
w <- c(0, 0, 0)  # Incluye sesgo
eta <- 0.01
batch_size <- 50
iteraciones <- 10

# Descenso del Gradiente Estocástico
for (i in 1:iteraciones) {
  minibatch <- data[sample(1:N, batch_size), ]
  g <- grad(w, minibatch)
  w <- w - eta * g
  cat("Iteración", i, "Parámetros:", round(w, 3), "\n")
}
