# Trabajo Visión Computacional
                                       “ Detección y Conteo de Residuos Inorgánicos Contaminantes mediante YOLO v9“
   
    CURSO:
    Visión Computacional
      
    INTEGRANTES:
    Huarcaya Pumacayo, Victor Nikolai
    More Ayay, Dahayra Xiomara
    Lima Quispe, Alexandra Nancy
    Huarauya Fabian, Josue Eduardo

    
    PROFESOR:
    Montalvo Garcia, Peter Jonathan
                                                                       2024 - I
                  
# INICIO DEL PROYECTO

El presente proyecto de investigación ha logrado desarrollar una solución de visión artificial basada en el modelo YOLO v9, con el objetivo de detectar y cuantificar los diferentes tipos de residuos sólidos inorgánicos presentes en un video grabado en un entorno rural. Entre los hallazgos más relevantes, se destaca que el uso del modelo YOLO v9 ha probado ser efectivo en la tarea de detección y clasificación de residuos como plástico, botellas, papel, cartón y vidrio, alcanzando altos niveles de precisión.

<div align="center">
    <a href="./">
        <img src="https://lirp.cdn-website.com/777d988b/dms3rep/multi/opt/acumulacio-n+basura+1-1920w.jpeg" width="79%"/>
    </a>
</div>


## Instalación De Recursos y Directorio
Prestar atención a cada paso para que se pueda entrenar y poner en práctica el algoritmo.

<details><summary> <b>Expand</b> </summary>

``` shell
# Muestra la información de la GPU de NVIDIA instalada, útil para verificar la disponibilidad de recursos.
!nvidia-smi
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          Off | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0              46W / 400W |      2MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+

# Importa la biblioteca os para interactuar con el sistema operativo
import os

# Obtiene y guarda el directorio de trabajo actual en la variable HOME
HOME = os.getcwd()

# Imprime el directorio de trabajo actual
print(HOME)
/content
```

</details>


## Clonación de repositorio y del modelo
<details><summary> <b>Expand</b> </summary>

``` shell
# Clona el repositorio de YOLOv9 desde GitHub
!git clone https://github.com/SkalskiP/yolov9.git

# Cambia el directorio actual al clon del repositorio YOLOv9 (comentado para compatibilidad de Python)
 %cd yolov9

# Instala las dependencias necesarias desde el archivo requirements.txt
!pip install -r requirements.txt -q
Cloning into 'yolov9'...
remote: Enumerating objects: 325, done.
remote: Counting objects: 100% (218/218), done.
remote: Compressing objects: 100% (62/62), done.
remote: Total 325 (delta 159), reused 156 (delta 156), pack-reused 107
Receiving objects: 100% (325/325), 2.23 MiB | 45.77 MiB/s, done.
Resolving deltas: 100% (165/165), done.
/content/yolov9
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.3/207.3 kB 5.6 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.7/62.7 kB 8.0 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 44.7 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.3/21.3 MB 61.5 MB/s eta 0:00:00
```


## Instalando Roboflow
<details><summary> <b>Expand</b> </summary>

``` shell
# Instala la biblioteca roboflow para manejo de datasets
!pip install -q roboflow
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75.6/75.6 kB 3.1 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 178.7/178.7 kB 11.6 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.5/54.5 kB 7.5 MB/s eta 0:00:00

```

## Descargando Yolov9
<details><summary> <b>Expand</b> </summary>

``` shell
# Descarga los archivos de pesos del modelo YOLOv9 y los guarda en el directorio {HOME}/weights
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt

```








