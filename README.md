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

# Importa la biblioteca os para interactuar con el sistema operativo
import os

# Obtiene y guarda el directorio de trabajo actual en la variable HOME
HOME = os.getcwd()

# Imprime el directorio de trabajo actual
print(HOME)

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

```










