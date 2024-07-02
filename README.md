
# Trabajo Visión Computacional
                            “ Detección y Conteo de Residuos Inorgánicos Contaminantes mediante YOLO v9“

<p align="center">
  <img src="https://github.com/Dahayra13/Trabajo-Final---Visi-n-Computacional-/assets/119473082/296d322f-9eff-4bfd-9d24-6326e4b2aa65" alt="image">
</p>

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

El presente proyecto de investigación ha logrado desarrollar una solución de visión artificial basada en el modelo YOLO v9 a la problematíca de la inadecuada gestión de los residuos solidos inorganicos en las zonas rurales, con el objetivo de detectar y cuantificar los diferentes tipos de residuos presentes en un video grabado en un entorno rural. Entre los hallazgos más relevantes, se destaca que el uso del modelo YOLO v9 ha probado ser efectivo en la tarea de detección y clasificación de residuos como plástico, botellas, papel, cartón y vidrio, alcanzando altos niveles de precisión.


<p align="center">
  <img src="https://github.com/Dahayra13/Trabajo-Final---Visi-n-Computacional-/assets/119473082/8e61e261-460f-400e-bc42-2fb800895cca" alt="image">
</p>


## Enlaces del Proyecto

- *Enlace al repositorio de GitHub*: [Repositorio en GitHub](https://github.com/Dahayra13/Trabajo-Final---Visi-n-Computacional-)
- *Enlace al notebook de Google Colab*: [Notebook en Google Colab](https://colab.research.google.com/drive/1f77N-iooEK_Z09bRh6lAPqcaKKWAjXXv#scrollTo=PHgTdsLNiMln)
- *Enlace del drive del video*: [Video](https://drive.google.com/drive/folders/1ihkDUGI_Dft0ENr_GMcNu3HI7g0ivjIN?usp=sharing)
- *Enlace del dataset*: [Dataset](https://drive.google.com/file/d/1khsADRka8d5Jef6fQPqbqyit0p9TjtaV/view?usp=sharing )
- *Enlace del Informe*: [Informe](https://docs.google.com/document/d/1JrrReoecM7GFBO5QgYkxAQIJv6m7qVKy3WQm6zioSpw/edit)


## Instrucciones para ejecutar el proyecto
- Python 3.7 o superior
- Google Colab cambiar "Entorno de ejecución" de CPU a GPU
- Visual Studio Code
- Buscar un dataset con clases ya establecidas.

## Clases utilizadas
- cardboard
- combustible waste
- glass
- metal
- paper
- plastic
- trash

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

## Dirrección de los archivos 
<details><summary> <b>Expand</b> </summary>

``` shell
# Lista los archivos descargados en el directorio {HOME}/weights para verificar la descarga
!ls -la {HOME}/weights
total 402440
drwxr-xr-x 2 root root      4096 Jul  2 02:52 .
drwxr-xr-x 1 root root      4096 Jul  2 02:52 ..
-rw-r--r-- 1 root root  51508261 Feb 18 12:36 gelan-c.pt
-rw-r--r-- 1 root root 117203713 Feb 18 12:36 gelan-e.pt
-rw-r--r-- 1 root root 103153312 Feb 18 12:36 yolov9-c.pt
-rw-r--r-- 1 root root 140217688 Feb 18 12:36 yolov9-e.pt

# Descarga una imagen de ejemplo desde Roboflow y la guarda en el directorio {HOME}/data
!wget -P {HOME}/data -q https://media.roboflow.com/notebooks/examples/dog.jpeg

# Define la ruta de la imagen de origen para la detección
SOURCE_IMAGE_PATH = f"{HOME}/Basura.jpeg"

# Ejecuta el script detect.py con los pesos del modelo gelan-c.pt y una confianza mínima de 0.1 para realizar la detección en la imagen
!python detect.py --weights {HOME}/weights/gelan-c.pt --conf 0.1 --source {HOME}/data/Basura.jpeg --device 0
detect: weights=['/content/weights/gelan-c.pt'], source=/content/data/Basura.jpeg, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.1, iou_thres=0.45, max_det=1000, device=0, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
YOLOv5 🚀 1e33dbb Python-3.10.12 torch-2.3.0+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB, 40514MiB)

Fusing layers... 
Model summary: 467 layers, 25472640 parameters, 0 gradients, 102.8 GFLOPs
image 1/1 /content/data/Basura.jpeg: 640x640 1 suitcase, 5 bottles, 17.0ms
Speed: 0.6ms pre-process, 17.0ms inference, 793.4ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/exp


# Muestra la segunda imagen resultante después de la detección, ajustando el ancho a 600 píxeles
Image(filename=f"{HOME}/yolov9/runs/detect/exp2/Basura.jpeg", width=600)

```

<p align="center">
  <img src="https://github.com/Dahayra13/Trabajo-Final---Visi-n-Computacional-/blob/main/Imagenes/descarga%20(1).jfif" alt="image">
</p>







## Problemas encontrados y cómo fueron solucionados por el equipo
- Cambio del entorno de ejecución del Colab: El equipo migró el proyecto a un entorno local para superar las limitaciones de recursos en Colab y optimizar el entrenamiento de YOLOv9.
- Baja velocidad y rendimiento del Google Colab: Se trasladó el proyecto a un entorno local con hardware más potente, como una GPU dedicada, para mejorar la velocidad y rendimiento del modelo YOLOv9 durante el entrenamiento.




