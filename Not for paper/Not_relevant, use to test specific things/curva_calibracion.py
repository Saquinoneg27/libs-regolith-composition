import numpy as np
import matplotlib.pyplot as plt

concentraciones = np.array([10.20, 9.24, 10.20, 8.00, 10.72])  # Mass fractions in %
intensidades = np.array([0.0017842371605054077,0.0016164225576409173,0.0018151061006792103,0.0014894821172533083,0.001606427198909733])  # a.u

# Ajuste de la curva de calibración
# Calculamos la pendiente y el intercepto de la recta
A = np.vstack([concentraciones, np.ones(len(concentraciones))]).T
pendiente, intercepto = np.linalg.lstsq(A, intensidades, rcond=None)[0]

# Predicciones de intensidades a partir de las concentraciones
intensidades_pred = pendiente * concentraciones + intercepto


# Cálculo del R cuadrado
ss_tot = np.sum((intensidades - np.mean(intensidades)) ** 2)
ss_res = np.sum((intensidades - intensidades_pred) ** 2)
r_cuadrado = 1 - (ss_res / ss_tot)

# Gráfico de la curva de calibración
plt.scatter(concentraciones, intensidades, label='Experimental Data')
plt.plot(concentraciones, intensidades_pred, color='red', label='Prediction')
plt.xlabel('Concentration (Mass fractions in %)')
plt.ylabel('Intensity (au)')
plt.title('Calibration curve for 40 mJ-Earth')
plt.legend()
plt.show()

# Imprimir la ecuación de la curva
print(f"Ecuación de la curva de calibración: Intensidad = {pendiente:.2f} * Concentración + {intercepto:.2f}")
print(f"Coeficiente de determinación (R^2): {r_cuadrado:.2f}")
