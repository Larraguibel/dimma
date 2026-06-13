# Next steps

- Añadir prueba de las cotas teóricas de alpha estacionariedad del paper
    - Con qué constante? solo se da el orden → Ajustar un polinomio inverso y que se eliga sola la cte. Nos enfocamos en forma
- Ojito con la constante de suavidad: si usas Relu es infinto. Usa funciones de activacion suaves com GeLU, sigmoide, etc.
    - Hecho
- **comparar rendimiento de Dp- SGD vs spider boost vs sin ruido**\|
- ~~probemos que pasa si pongo delta = 1/n\*\*2~~
- **Paper de 2024 de sparsity: Aunque no sea optimo, implementar los gradientes con proyección sobre DP spider.**
- Option B — T·(1+q) gradient steps of batch size b2 for the baseline. It<br>equates total gradient computations (the primary cost), is easy to compute from existing config<br>values, and is the standard convention in DP-SGD comparison papers. Echarle una miradita
- Subir esto en un github tipo pdf
- Dejar instrucciones par a generar llave MOK.
- añadir baseline a todos los plots (modelo sin DP)
- Separar poisson con truncated en dos archivos
