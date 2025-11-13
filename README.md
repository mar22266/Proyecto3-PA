# Proyecto Final

## Integrantes
- Sergio Orellana
- Rodrigo Mansilla
- Andre Marroquín


## Descripción del proyecto

El proyecto consiste en simular galaxias y estrellas utilizando *CUDA* para aplicar conceptos de computación paralela.

- Cada *galaxia* se representa como un *bloque* de hilos.
- Cada *estrella* se representa como un *hilo* dentro de ese bloque.
- A cada estrella se le asigna un *brillo* calculado en la GPU.

El trabajo se desarrolla en tres fases principales:

1. *Fase 1:* Cada hilo calcula e imprime el brillo de su estrella.
2. *Fase 2:* Las estrellas de una galaxia usan *memoria compartida* para almacenar sus brillos y se imprimen agrupadas por galaxia.
3. *Fase 3:* Las estrellas colaboran para calcular el *brillo promedio* de cada galaxia y se determina cuál galaxia es la más brillante.

