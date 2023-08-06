"""
ESTACIÓN DE RADIOFRECUENCIA MARCIANA EN BANDA UHF
Autor: David Villa Blanco
Código: Emisión de pulso gaussiano por antena parabólica
"""

# Importamos las librerías
import numpy as np
import matplotlib.pyplot as plt

# DATOS DE LA SIMULACIÓN
c = 299792458 # Velocidad de la luz en el vacío
Dx = 5e-1 # Separación entre puntos del mallado espacial
Dt = Dx/(2*c) # Separación entre puntos del array temporal
npuntos = 400 # Número de puntos del mallado espacial
E0 = 1 # Amplitud pulso inicial
eps_0 = 8.854e-12 # permitividad eléctrica del vacío)
eps_exterior = 4.3 # Epsilon relativo marte
cond_exterior = 1e-12 # Conductividad marte
wavelength = 0.749481145 # Longitud de onda del pulso inicial
Dxp = 2*np.sqrt(np.log(2))*wavelength # Ancho espacial del pulso inicial (pulso gaussiano)
Dtp = Dxp/c # Anchura temporal del pulso inicial
t0p = 5*Dtp # Centro temporal del pulso inicial
npasos = int(2e3) # Número total de pasos de la simulación
pasos_entre_reps = 10 # Pasos entre repeticiones de la simulación
pausa_entre_reps = 0.01 # Pausa temporal entre actualización del plot

# ARRAYS VARIOS
t_array = np.linspace(0, Dt*(npasos-1), npasos) # Array temporal
x_array = np.linspace(0, Dx*(npuntos-1), npuntos) # Array espacial en x
y_array = np.linspace(0, Dx*(npuntos-1), npuntos) # Array espacial en y
yy, xx = np.meshgrid(y_array, x_array) # Mallado espacial
eps_array = eps_exterior*np.ones([npuntos,npuntos]) # Array de permitividad eléctrica (Se irá modificando en función de los elementos de la simulación y sus posiciones, inicializándose en eps_exterior)
cond_array = cond_exterior*np.ones([npuntos,npuntos]) # Array de conductividad eléctrica (Se irá modificando en función de los elementos de la simulación y sus posiciones, inicializándose en cond_exterior)
Ez = np.zeros((len(x_array), len(y_array))) # Mallado del campo eléctrico eje z
Hx = np.zeros((len(x_array), len(y_array))) # Mallado del campo magnético eje x
Hy = np.zeros((len(x_array), len(y_array))) # Mallado del campo magnético eje y
# Vectores de absorción (Condiciones frontera)
Ez_izq1 = np.zeros(npuntos)
Ez_izq2 = np.zeros(npuntos)
Ez_dcha1 = np.zeros(npuntos)
Ez_dcha2 = np.zeros(npuntos)
Ez_arriba1 = np.zeros(npuntos)
Ez_arriba2 = np.zeros(npuntos)
Ez_abajo1 = np.zeros(npuntos)
Ez_abajo2 = np.zeros(npuntos)
indice_pulso_x = np.array([int(npuntos-2)]) # Índice espacial del pulso en el eje x
indice_pulso_y = np.arange(0, npuntos, 1) # Índices espaciales del puslo en el eje y

# SIMULACIÓN ANTENA PARABÓLICA
a = 0.007 # Apertura de la parábola
k = Dx*npuntos/2 # Eje de simetría de la parábola
focox = 1/(4*a) # Posición x del foco de la parábola
focoy = k # Posición y del foco de la parábola
eps_antena_value = 1.7 # Permitividad electrica de la antena (y del alimentador, ambos hechos de Al)
cond_antena_value = 3.538e7 # Conductividad eléctrica de la antena (y del alimentador, ambos hechos de Al)
eps_soporte_value = 2.1 # Permitividad eléctrica del soporte (hecho de Teflón)
cond_soporte_value = 1e-24 # Conductividad eléctrica del soporte (hecho de Teflón)
# Posición ANTENA
y_ant = np.linspace(Dx*npuntos/4, 3*Dx*npuntos/4, int(npuntos/2)) # Array y antena
y_ant_ind = y_ant/Dx # Índices y antena
x_ant = a*(y_ant-k)**2 # Array x antena
x_ant_ind = x_ant/Dx # Índices x antena
# Posición SOPORTE INFERIOR
a1 = x_ant[-1] # Valor de x más alejado de la antena
a2 = focox
separacion = 4*Dx # Separación espacial en el eje x entre el alimentador y el pulso
b = (y_ant[int(len(y_ant)/2)]-y_ant[0])/(a2-a1+separacion) # Pendiente soportes
x_soporte = np.linspace(a1, a2+separacion, int((a2-a1+separacion)/Dx)) # Array x soporte inferior
x_soporte_ind = x_soporte/Dx # Índices x soporte inferior
y_soporte = b*(x_soporte-a1)+y_ant[0] # Array y soporte inferior
y_soporte_ind = y_soporte/Dx # Índices y soporte inferior
# Posición SOPORTE SUPERIOR
x_soporte2 = np.linspace(a1, a2+separacion, int((a2-a1+separacion)/Dx)) # Array x soporte superior
x_soporte_ind2 = x_soporte2/Dx # Índices x soporte superior
y_soporte2 = -b*(x_soporte-a1)+y_ant[-1] # Array y soporte superior
y_soporte_ind2 = y_soporte2/Dx # Índices y soporte superior
# Posición ALIMENTADOR INFERIOR
x_alimen = np.linspace(x_soporte[-6],x_soporte[-1], 6) # Array x alimentador inferior
x_alimen_ind =x_alimen/Dx # Índices x alimentador inferior
y_alimen = b*(x_alimen-a1)+y_ant[0] # Array y alimentador inferior
y_alimen_ind = y_alimen/Dx # Índices y alimentador inferior
# Posición ALIMENTADOR SUPERIOR
x_alimen2 = np.linspace(x_soporte[-6],x_soporte[-1], 6) # Array x alimentador superior
x_alimen_ind2 =x_alimen2/Dx # Índices x alimentador superior
y_alimen2 = -b*(x_alimen-a1)+y_ant[-1] # Array y alimentador superior
y_alimen_ind2 = y_alimen2/Dx # Índices y alimentador superior

# ACTUALIZACIÓN DE LOS ARRAYS DE CONDUCTIVIDAD Y PERMITIVIDAD ELÉCTRICAS
# Actualización antena
for ind in range(len(y_ant_ind)):
    eps_array[int(x_ant_ind[ind]),int(y_ant_ind[ind])] = eps_antena_value # Relleno PERMITIVIDAD ANTENA
    cond_array[int(x_ant_ind[ind]),int(y_ant_ind[ind])] = cond_antena_value # Relleno CONDUCTIVIDAD ANTENA
# Actualización soportes
for ind in range(len(y_soporte_ind)):
    eps_array[int(x_soporte_ind[ind]),int(y_soporte_ind[ind])] = eps_soporte_value # Relleno PERMITIVIDAD SOPORTE INFERIOR
    cond_array[int(x_soporte_ind[ind]),int(y_soporte_ind[ind])] = cond_soporte_value # Relleno CONDUCTIVIDAD SOPORTE INFERIOR
    eps_array[int(x_soporte_ind2[ind]),int(y_soporte_ind2[ind])] = eps_soporte_value # Relleno PERMITIVIDAD SOPORTE SUPERIOR
    cond_array[int(x_soporte_ind2[ind]),int(y_soporte_ind2[ind])] = cond_soporte_value # Relleno CONDUCTIVIDAD SOPORTE SUPERIOR
# Actualización alimentadores
for ind in range(len(y_alimen_ind)):
    eps_array[int(x_alimen_ind[ind]),int(y_alimen_ind[ind])] = eps_antena_value # Relleno PERMITIVIDAD ALIMENTADOR INFERIOR
    cond_array[int(x_alimen_ind[ind]),int(y_alimen_ind[ind])] = cond_antena_value # Relleno CONDUCTIVIDAD ALIMENTADOR INFERIOR
    eps_array[int(x_alimen_ind2[ind]),int(y_alimen_ind2[ind])] = eps_antena_value # Relleno PERMITIVIDAD ALIMENTADOR SUPERIOR
    cond_array[int(x_alimen_ind2[ind]),int(y_alimen_ind2[ind])] = cond_antena_value # Relleno CONDUCTIVIDAD ALIMENTADOR SUPERIOR

ctc = cond_array*Dt/(2*eps_0*eps_array) # Coeficiente de tiempo de conductancia

# CREAMOS EL PLOT (MAPA DE CALOR REPRESENTANDO E_z)
fig = plt.figure(figsize=(8,8)) # Creamos nuestra figura
ejes = fig.add_subplot() # Añadimos un subplot
# Ajustamos límites en ejes x e y
ejes.set_ylim(-1.5*E0, 1.5*E0)
ejes.set_xlim(0, Dx*npuntos)
levels = np.linspace(-0.1*E0,0.1*E0, 21) # Cantidad de colores distintos del mapa de calor
cs = ejes.contourf(xx,yy, np.clip(Ez, -0.1*E0, 0.1*E0), levels, cmap="jet") # Graficamos
bar = plt.colorbar(cs) # Mostramos la barra de colores

# SIMULACIÓN (BUCLE TEMPORAL PRINCIPAL)
for t_ind in range(len(t_array)):
    # Actualización del campo eléctrico
    Ez[1:,1:] = ((1-ctc[1:,1:])/(1+ctc[1:,1:]))*Ez[1:,1:]+(1/(2*eps_array[1:,1:]*(1+ctc[1:,1:])))*(Hy[1:,1:]-Hy[:-1,1:])-(1/(2*eps_array[1:,1:]*(1+ctc[1:,1:])))*(Hx[1:,1:]-Hx[1:,:-1])
    # Generamos el pulso plano durante dos unidades de tiempo
    if t_ind in np.arange(0,2,1):
        for i in range(npuntos):
            Ez[indice_pulso_x[0],indice_pulso_y[i]] = E0 # Fuente fuerte
    # Condiciones de contorno (absorbentes)
    # Lado izquierdo
    Ez[0,:][:] = Ez_izq1
    Ez_izq1[:] = Ez_izq2
    Ez_izq2[:] = Ez[1, :]
    # Lado derecho
    Ez[-1,:][:] = Ez_dcha1
    Ez_dcha1[:] = Ez_dcha2
    Ez_dcha2[:] = Ez[-2, :]
    # Lado superior
    Ez[:,0][:] = Ez_arriba1
    Ez_arriba1[:] = Ez_arriba2
    Ez_arriba2[:] = Ez[:, 1]
    # Lado inferior
    Ez[:,-1][:] = Ez_abajo1
    Ez_abajo1[:] = Ez_abajo2
    Ez_abajo2[:] = Ez[:, -2]
    # Actualización del campo magnético
    Hx[:,:-1] = Hx[:,:-1]-(1/2)*(Ez[:,1:]-Ez[:,:-1])
    Hy[:-1,:] = Hy[:-1,:]+(1/2)*(Ez[1:,:]-Ez[:-1,:])
    # Condicional de representación
    if t_ind%pasos_entre_reps == 0:
        # ACTUALIZAMOS MAPA DE CALOR
        ejes.cla()
        ejes.set_title("Recepción por antena parabólica de un pulso electromagnético plano")
        ejes.set_xlabel("x (m)")
        ejes.set_ylabel("y (m)")
        ejes.contourf(xx,yy, np.clip(Ez, -0.1*E0, 0.1*E0), levels, cmap="jet")
        ejes.plot(x_ant_ind*Dx, y_ant_ind*Dx, "w-") # Plot de la antena
        ejes.plot(x_soporte, y_soporte, "k-") # Plot del soporte inferior
        ejes.plot(x_alimen, y_alimen, "w-") # Plot del alimentador inferior
        ejes.plot(x_soporte2, y_soporte2, "k-") # Plot del soporte superior
        ejes.plot(x_alimen2, y_alimen2, "w-") # Plot del alimentador superior
        plt.pause(pausa_entre_reps) # Pausa entre actualizaciones del plot