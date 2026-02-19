# Desarrollo Matemático — Ecuación de Poisson 2D
## Taller Sistemas Distribuidos 2026-1 — Universidad Distrital

---

## 1. Planteamiento del Problema

Se busca resolver la **ecuación de Poisson bidimensional**:

$$\nabla^2 \phi(x,y) = f(x,y), \quad (x,y) \in \Omega = [0,1]\times[0,1]$$

con término fuente:

$$f(x,y) = \sin(\pi x)\sin(\pi y)$$

y condiciones de frontera de Dirichlet homogéneas:

$$\phi(x,y) = 0, \quad \forall (x,y) \in \partial\Omega$$

El operador Laplaciano en 2D es:

$$\nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2}$$

---

## 2. Solución Analítica

Por el método de eigenfunciones, buscamos $\phi(x,y) = A\sin(\pi x)\sin(\pi y)$.

Aplicando el Laplaciano:

$$\nabla^2\left[A\sin(\pi x)\sin(\pi y)\right] = A\left(-\pi^2\sin(\pi x)\sin(\pi y)\right) + A\left(-\pi^2\sin(\pi x)\sin(\pi y)\right)$$

$$= -2\pi^2 A \sin(\pi x)\sin(\pi y) = \sin(\pi x)\sin(\pi y)$$

Por lo tanto:

$$-2\pi^2 A = 1 \implies A = -\frac{1}{2\pi^2}$$

La **solución exacta** es:

$$\boxed{\phi^*(x,y) = -\frac{\sin(\pi x)\sin(\pi y)}{2\pi^2}}$$

con $\|\phi^*\|_\infty = \frac{1}{2\pi^2} \approx 0.05066$.

---

## 3. Discretización por Diferencias Finitas Centradas

### 3.1 Malla Uniforme

Se discretiza el dominio $[0,1]\times[0,1]$ con una malla uniforme $N\times N$ de puntos **interiores**:

$$x_i = i \cdot h, \quad y_j = j \cdot h, \quad h = \frac{1}{N+1}, \quad i,j = 0,1,\ldots,N+1$$

Los **puntos interiores** son $i,j = 1,\ldots,N$ y los puntos de frontera $(i=0, i=N+1, j=0, j=N+1)$ tienen $\phi = 0$.

### 3.2 Aproximación del Laplaciano

La segunda derivada parcial se aproxima con la fórmula de diferencias finitas centradas de orden 2:

$$\frac{\partial^2 \phi}{\partial x^2}\bigg|_{(x_i,y_j)} \approx \frac{\phi_{i-1,j} - 2\phi_{i,j} + \phi_{i+1,j}}{h^2} + O(h^2)$$

$$\frac{\partial^2 \phi}{\partial y^2}\bigg|_{(x_i,y_j)} \approx \frac{\phi_{i,j-1} - 2\phi_{i,j} + \phi_{i,j+1}}{h^2} + O(h^2)$$

Sumando ambas aproximaciones, el **operador Laplaciano discreto** (estencil de 5 puntos) resulta:

$$\nabla^2_h \phi_{i,j} = \frac{\phi_{i-1,j} + \phi_{i+1,j} + \phi_{i,j-1} + \phi_{i,j+1} - 4\phi_{i,j}}{h^2}$$

### 3.3 Sistema Lineal

Igualando el Laplaciano discreto a la fuente evaluada en la malla:

$$\frac{\phi_{i-1,j} + \phi_{i+1,j} + \phi_{i,j-1} + \phi_{i,j+1} - 4\phi_{i,j}}{h^2} = f_{i,j}$$

donde $f_{i,j} = \sin(\pi i h)\sin(\pi j h)$.

Esto genera un sistema lineal $A\mathbf{u} = \mathbf{b}$ de $N^2$ ecuaciones, donde $A$ es la **matriz de Laplaciano discreto** (matriz dispersa pentadiagonal con estructura de bloque).

---

## 4. Esquema Iterativo de Jacobi

### 4.1 Derivación

Despejando $\phi_{i,j}$ en la ecuación discreta:

$$4\phi_{i,j} = \phi_{i-1,j} + \phi_{i+1,j} + \phi_{i,j-1} + \phi_{i,j+1} - h^2 f_{i,j}$$

El método de **Jacobi** actualiza todos los puntos interiores simultáneamente usando los valores de la iteración anterior. Si $\phi^{(k)}_{i,j}$ denota el valor en la iteración $k$, la ecuación discreta implementada explícitamente es:

$$\boxed{\phi^{(k+1)}_{i,j} = \frac{1}{4}\left[\phi^{(k)}_{i-1,j} + \phi^{(k)}_{i+1,j} + \phi^{(k)}_{i,j-1} + \phi^{(k)}_{i,j+1} - h^2 f_{i,j}\right], \quad i,j = 1,\ldots,N}$$

con condiciones de frontera: $\phi^{(k)}_{0,j} = \phi^{(k)}_{N+1,j} = \phi^{(k)}_{i,0} = \phi^{(k)}_{i,N+1} = 0$ para todo $k$.

### 4.2 Interpretación Matricial

El método de Jacobi equivale a la descomposición $A = D - (L+U)$ donde $D$ es la diagonal ($D = 4/h^2 \cdot I$), y la iteración es:

$$\phi^{(k+1)} = D^{-1}(L+U)\phi^{(k)} + D^{-1}\mathbf{b} = G_J\phi^{(k)} + \mathbf{c}$$

La **matriz de iteración de Jacobi** $G_J = D^{-1}(L+U)$ tiene radio espectral:

$$\rho(G_J) = \cos(\pi h) = \cos\!\left(\frac{\pi}{N+1}\right) \approx 1 - \frac{\pi^2}{2(N+1)^2}$$

Para $N=512$: $\rho \approx 0.999\,981$ → convergencia muy lenta para $N$ grande.

### 4.3 Criterio de Convergencia

El método se ejecuta hasta satisfacer:

$$\left\|\phi^{(k+1)} - \phi^{(k)}\right\|_\infty = \max_{1\le i,j\le N}\left|\phi^{(k+1)}_{i,j} - \phi^{(k)}_{i,j}\right| < \varepsilon = 10^{-6}$$

**Justificación**: La norma infinito de la diferencia entre iteraciones consecutivas es una medida del residuo de actualización. Cuando esta diferencia es suficientemente pequeña, el método ha alcanzado un estado estacionario (las correcciones ya no cambian la solución apreciablemente). Este criterio es equivalente a medir la norma del residuo $\|r^{(k)}\| = \|A\phi^{(k)} - b\|$ escalado por $h^2/4$.

**Observación importante**: Para $N$ grande, el paso $h$ es pequeño y la primera actualización desde $\phi^{(0)}=0$ tiene magnitud:

$$\left\|\phi^{(1)} - \phi^{(0)}\right\|_\infty = \frac{h^2}{4}\max_{i,j}|f_{i,j}| = \frac{h^2}{4} = \frac{1}{4(N+1)^2}$$

Para $N \geq 500$: $\frac{1}{4\cdot501^2} \approx 9.96\times10^{-7} < 10^{-6}$.

Esto significa que el criterio se satisface en pocas iteraciones para $N$ grande (la solución numérica converge rápido al óptimo para $h$ pequeño). **Para los experimentos de rendimiento se utiliza un número fijo de iteraciones** ($N_{iter}=100$) que garantiza tiempo de ejecución medible en todos los tamaños de malla.

---

## 5. Análisis de Localidad Espacial y Caché

### 5.1 Layout de Memoria

**C++ (row-major)**: El elemento $\phi[i][j]$ en una malla $(N+2)\times(N+2)$ se almacena en la posición lineal $i\cdot(N+2)+j$. Los elementos de una misma fila son **contiguos** en memoria.

**Fortran (column-major)**: El elemento `phi(i,j)` se almacena en posición $(j-1)\cdot(N+2)+(i-1)$. Los elementos de una misma **columna** son contiguos en memoria.

### 5.2 Línea de Caché en Apple M3

El Apple M3 tiene una línea de caché de **128 bytes = 16 doubles**. Esto implica que un acceso a $\phi_{i,j}$ carga automáticamente $\phi_{i,j}, \phi_{i,j+1}, \ldots, \phi_{i,j+15}$ en C++ (fila entera en trozos de 16) o $\phi_{i,j}, \phi_{i+1,j}, \ldots, \phi_{i+15,j}$ en Fortran.

### 5.3 Orden de Bucles Óptimo

#### C++ (row-major) — Bucle ij (ÓPTIMO):
```
for i = 1..N:          // fila exterior
    for j = 1..N:      // columna interior
        phi_new[i][j] = ...phi[i][j-1]...phi[i][j+1]...  // ACCESO SECUENCIAL
```
- `phi[i][j-1]`, `phi[i][j]`, `phi[i][j+1]`: **acceso secuencial** — misma línea de caché ✓
- `phi[i-1][j]`, `phi[i+1][j]`: stride $N+2$ — líneas de caché distintas, pero se cargan y **se reusan** a lo largo del bucle j ✓

#### C++ — Bucle ji (SUBÓPTIMO):
```
for j = 1..N:          // columna exterior
    for i = 1..N:      // fila interior
        phi_new[i][j] = ...phi[i-1][j]...phi[i+1][j]...  // ACCESO STRIDE-N
```
- `phi[i][j]`, `phi[i+1][j]`: stride $(N+2)$ doubles = $(N+2)\times8$ bytes ≈ **32 KB para $N=4096$** — cada acceso es un fallo de caché diferente ✗

#### Fortran (column-major) — Bucle ji (ÓPTIMO):
```fortran
do j = 2, N+1
    do i = 2, N+1
        phi_new(i,j) = ...phi(i-1,j)...phi(i+1,j)...  ! SECUENCIAL en columna
```
- `phi(i-1,j)`, `phi(i,j)`, `phi(i+1,j)`: **acceso secuencial** en columna ✓

### 5.4 Técnica de Blocking/Tiling

Para una malla $N\times N$ con bloque $B\times B$, el conjunto activo en caché por bloque es:

$$\text{Working set} = \underbrace{(B+2)^2}_{\phi^{(k)}} + \underbrace{B^2}_{\phi^{(k+1)}} + \underbrace{B^2}_{f} \text{ doubles}$$

Para Apple M3 (L1D = 128 KB, L2 = 16 MB):

| $B$ | Working set (KB) | Caché objetivo |
|-----|-----------------|---------------|
| 8   | 1.25 KB         | L1 ✓          |
| 16  | 4.0 KB          | L1 ✓          |
| 32  | 13.5 KB         | L1 ✓          |
| 64  | 51.8 KB         | L1 (ajustado) ✓ |
| 128 | 204 KB          | L2 ✓          |

---

## 6. Análisis de Intensidad Aritmética y Modelo Roofline

### 6.1 Operaciones Aritméticas por Punto

El núcleo del estencil de Jacobi por punto interior $(i,j)$:

| Operación | Costo |
|-----------|-------|
| 4 sumas/restas de vecinos | 4 FLOPs |
| 1 multiplicación $h^2 \cdot f_{ij}$ | 1 FLOP |
| 1 resta del término fuente | 1 FLOP |
| 1 multiplicación por $\frac{1}{4}$ | 1 FLOP |
| 1 resta para diferencia $\phi_{new}-\phi_{old}$ | 1 FLOP |
| 1 valor absoluto (comparación) | — |

**Total por punto: 8 FLOPs** (o 7 si se omite la diferencia).

Por iteración: $8 \times N^2$ FLOPs.

### 6.2 Transferencia de Datos

Con reuso de caché perfecto (cada elemento leído una sola vez):

| Dato | Accesos por punto |
|------|-------------------|
| `phi[i-1][j]`, `phi[i+1][j]`, `phi[i][j-1]`, `phi[i][j+1]` | 4 lecturas |
| `phi[i][j]` (para la resta) | 1 lectura |
| `f[i][j]` | 1 lectura |
| `phi_new[i][j]` | 1 escritura |

**Total: 7 transferencias × 8 bytes = 56 bytes/punto** (límite inferior con caché perfecta).

En la práctica, con acceso row-major para $N$ grande:
- Las líneas de caché se cargan por completo (16 doubles)
- El patrón de 5 puntos implica cargar $\approx 5$ líneas de caché por punto en el peor caso
- Efectivo: $\approx$ 40-80 bytes/punto dependiendo del tamaño $N$

### 6.3 Intensidad Aritmética

$$I = \frac{\text{FLOPs}}{\text{Bytes transferidos}} = \frac{8\text{ FLOP}}{56\text{ bytes}} \approx 0.143 \text{ FLOP/byte}$$

### 6.4 Modelo Roofline — Apple M3

Parámetros del Apple M3 (núcleos de rendimiento):

- **Pico computacional**: $\approx 4.05\text{ GHz} \times 8\text{ cores} \times 2\text{ unidades FMA} \times 2\text{ (FMA=2 ops)} \times 2\text{ (NEON 128-bit = 2 doubles)} = 259\text{ GFLOPS}$
- **Ancho de banda de memoria**: $\approx 100\text{ GB/s}$ (unified memory architecture)
- **Punto de quiebre (ridge point)**: $\frac{259\text{ GFLOPS}}{100\text{ GB/s}} = 2.59\text{ FLOP/byte}$

Con $I \approx 0.14\text{ FLOP/byte} \ll 2.59\text{ FLOP/byte}$, el algoritmo es **fuertemente memory-bound**:

$$\text{Rendimiento máximo alcanzable} = I \times B_W = 0.14 \times 100 = 14\text{ GFLOPS}$$

(solo el 5.4% del pico computacional)

El cuello de botella es el ancho de banda de memoria, no la capacidad de cómputo. Las optimizaciones de caché (blocking) tienen mayor impacto que las optimizaciones de ILP o vectorización.

---

## 7. Estimación del Número de Iteraciones para Convergencia

La tasa de convergencia de Jacobi para la ecuación de Poisson es:

$$\left\|\phi^{(k)} - \phi^*\right\|_\infty \leq \rho(G_J)^k \left\|\phi^{(0)} - \phi^*\right\|_\infty$$

Para reducir el error en un factor $\varepsilon$:

$$k \geq \frac{\log(\varepsilon)}{\log\rho(G_J)} \approx \frac{-\log(\varepsilon)}{\pi^2/[2(N+1)^2]} = \frac{2(N+1)^2\log(1/\varepsilon)}{\pi^2}$$

| $N$ | $h$ | $\rho(G_J)$ | $k_{\text{conv}}(\varepsilon=10^{-6})$ |
|-----|-----|------------|----------------------------------------|
| 64  | 1/65 | 0.98834 | ≈ 9,100 |
| 128 | 1/129 | 0.99709 | ≈ 36,600 |
| 256 | 1/257 | 0.99926 | ≈ 146,200 |
| 512 | 1/513 | 0.99981 | ≈ 584,500 |

Esto confirma la necesidad de usar un número fijo de iteraciones en los experimentos de rendimiento.

---

## 8. Error de Discretización

El error de truncamiento local del estencil de 5 puntos es de orden $O(h^2)$:

$$\left|\nabla^2_h \phi^* - f\right| \leq \frac{h^2}{12}\left(\frac{\partial^4\phi^*}{\partial x^4} + \frac{\partial^4\phi^*}{\partial y^4}\right) = \frac{h^2 \pi^4}{6}\phi^*$$

El error global de discretización es:

$$\|\phi_h - \phi^*\|_\infty \leq C h^2 \approx \frac{\pi^2}{12} h^2$$

Para validación numérica con $N=64$ ($h=1/65$):

$$\text{Error esperado} \approx \frac{\pi^2}{12} \times (1/65)^2 \approx 1.94 \times 10^{-4}$$
