## Asignación de pesos a clases
Para compensar la falta de balanceo asignamos a cada clase un peso. La suma de éstos da 1 y se enmarca dentro de las 
estrategias semisupervisadas, utilizando el conjunto de entrenamiento para estimar la verdadera distribución de los
datos de test (éstos últimos se entienden como la verdadera distribución de los datos) 

## Calibrado de modelos
Una vez obtenido el OVA, para calibrar la confianza de los modelos aplicamos un desplazamiento constante por columna
obtenido mediante diferential evolution. 

## Selección de características
Mediante un algoritmo genético, con cruzado uniforme, seleccionamos las mejores características sobre una versión 
simplificada del problema (problema de regresión y un subconjunto de características) 