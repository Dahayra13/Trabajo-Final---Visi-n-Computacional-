
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



## Cambia el directorio
<details><summary> <b>Expand</b> </summary>

``` shell
# Cambia el directorio actual al clon del repositorio YOLOv9 (comentado para compatibilidad de Python)
%cd {HOME}/yolov9
/content/yolov9
```



## Instalación de Roboflow
<details><summary> <b>Expand</b> </summary>

``` shell
# Instala nuevamente la biblioteca roboflow (posiblemente redundante)
!pip install roboflow

# Importa la clase Roboflow y configura el proyecto para descargar datos utilizando una clave de API
from roboflow import Roboflow
rf = Roboflow(api_key="owlqaOdSGhLb078zPaIw")
project = rf.workspace("trash-sorter").project("synthetic-trashes")
version = project.version(2)
dataset = version.download("yolov9")
Requirement already satisfied: roboflow in /usr/local/lib/python3.10/dist-packages (1.1.33)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from roboflow) (2024.6.2)
Requirement already satisfied: chardet==4.0.0 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.0.0)
Requirement already satisfied: idna==3.7 in /usr/local/lib/python3.10/dist-packages (from roboflow) (3.7)
Requirement already satisfied: cycler in /usr/local/lib/python3.10/dist-packages (from roboflow) (0.12.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.4.5)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from roboflow) (3.7.1)
Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.25.2)
Requirement already satisfied: opencv-python-headless==4.10.0.84 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.10.0.84)
Requirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from roboflow) (9.4.0)
Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.8.2)
Requirement already satisfied: python-dotenv in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.0.1)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.31.0)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.16.0)
Requirement already satisfied: urllib3>=1.26.6 in /usr/local/lib/python3.10/dist-packages (from roboflow) (2.0.7)
Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.10/dist-packages (from roboflow) (4.66.4)
Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from roboflow) (6.0.1)
Requirement already satisfied: requests-toolbelt in /usr/local/lib/python3.10/dist-packages (from roboflow) (1.0.0)
Requirement already satisfied: python-magic in /usr/local/lib/python3.10/dist-packages (from roboflow) (0.4.27)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (1.2.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (4.53.0)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (24.1)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->roboflow) (3.1.2)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->roboflow) (3.3.2)
loading Roboflow workspace...
loading Roboflow project...
Downloading Dataset Version Zip in Synthetic-trashes-2 to yolov9:: 100%|██████████| 44546/44546 [00:03<00:00, 12109.61it/s]

Extracting Dataset Version Zip to Synthetic-trashes-2 in yolov9:: 100%|██████████| 1618/1618 [00:00<00:00, 5114.40it/s]
```

## Cambia el directorio Y Verificación
<details><summary> <b>Expand</b> </summary>

``` shell
# Cambia el directorio actual al clon del repositorio YOLOv9 (comentado para compatibilidad de Python)
%cd {HOME}/yolov9
/content/yolov9
```

## Entrenamiento
<details><summary> <b>Expand</b> </summary>

``` shell
# Ejecuta el script train.py para entrenar el modelo con parámetros específicos como tamaño del lote, número de épocas, tamaño de imagen y pesos iniciales
!python train.py \
--batch 16 --epochs 30 --img 600 --device 0 --min-items 0 --close-mosaic 15 \
--data {dataset.location}/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
/content/yolov9
2024-07-02 02:54:52.939719: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-07-02 02:54:52.991276: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-07-02 02:54:52.991322: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-07-02 02:54:52.992907: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-07-02 02:54:53.000989: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-07-02 02:54:54.218419: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
train: weights=/content/weights/gelan-c.pt, cfg=models/detect/gelan-c.yaml, data=/content/yolov9/Synthetic-trashes-2/data.yaml, hyp=hyp.scratch-high.yaml, epochs=30, batch_size=16, imgsz=600, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=None, image_weights=False, device=0, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, flat_cos_lr=False, fixed_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, min_items=0, close_mosaic=15, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
YOLOv5 🚀 1e33dbb Python-3.10.12 torch-2.3.0+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB, 40514MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, cls_pw=1.0, dfl=1.5, obj_pw=1.0, iou_t=0.2, anchor_t=5.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.3
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLO 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLO 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
100% 755k/755k [00:00<00:00, 15.8MB/s]
Overriding model.yaml nc=80 with nc=8

                 from  n    params  module                                  arguments                     
  0                -1  1      1856  models.common.Conv                      [3, 64, 3, 2]                 
  1                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  2                -1  1    212864  models.common.RepNCSPELAN4              [128, 256, 128, 64, 1]        
  3                -1  1    164352  models.common.ADown                     [256, 256]                    
  4                -1  1    847616  models.common.RepNCSPELAN4              [256, 512, 256, 128, 1]       
  5                -1  1    656384  models.common.ADown                     [512, 512]                    
  6                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
  7                -1  1    656384  models.common.ADown                     [512, 512]                    
  8                -1  1   2857472  models.common.RepNCSPELAN4              [512, 512, 512, 256, 1]       
  9                -1  1    656896  models.common.SPPELAN                   [512, 512, 256]               
 10                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 11           [-1, 6]  1         0  models.common.Concat                    [1]                           
 12                -1  1   3119616  models.common.RepNCSPELAN4              [1024, 512, 512, 256, 1]      
 13                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 14           [-1, 4]  1         0  models.common.Concat                    [1]                           
 15                -1  1    912640  models.common.RepNCSPELAN4              [1024, 256, 256, 128, 1]      
 16                -1  1    164352  models.common.ADown                     [256, 256]                    
 17          [-1, 12]  1         0  models.common.Concat                    [1]                           
 18                -1  1   2988544  models.common.RepNCSPELAN4              [768, 512, 512, 256, 1]       
 19                -1  1    656384  models.common.ADown                     [512, 512]                    
 20           [-1, 9]  1         0  models.common.Concat                    [1]                           
 21                -1  1   3119616  models.common.RepNCSPELAN4              [1024, 512, 512, 256, 1]      
 22      [15, 18, 21]  1   5496808  models.yolo.DDetect                     [8, [256, 512, 512]]          
gelan-c summary: 621 layers, 25443240 parameters, 25443224 gradients, 103.2 GFLOPs

Transferred 931/937 items from /content/weights/gelan-c.pt
AMP: checks passed ✅
WARNING ⚠️ --img-size 600 must be multiple of max stride 32, updating to 608
optimizer: SGD(lr=0.01) with parameter groups 154 weight(decay=0.0), 161 weight(decay=0.0005), 160 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
train: Scanning /content/yolov9/Synthetic-trashes-2/train/labels... 563 images, 0 backgrounds, 0 corrupt: 100% 563/563 [00:00<00:00, 4104.86it/s]
train: New cache created: /content/yolov9/Synthetic-trashes-2/train/labels.cache
val: Scanning /content/yolov9/Synthetic-trashes-2/valid/labels... 150 images, 0 backgrounds, 0 corrupt: 100% 150/150 [00:00<00:00, 2731.35it/s]
val: New cache created: /content/yolov9/Synthetic-trashes-2/valid/labels.cache
Plotting labels to runs/train/exp/labels.jpg... 
Image sizes 608 train, 608 val
Using 8 dataloader workers
Logging results to runs/train/exp
Starting training for 30 epochs...
/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       0/29      10.4G      1.198      4.461      1.333        538        608:   0% 0/36 [00:04<?, ?it/s]WARNING ⚠️ TensorBoard graph visualization failure Sizes of tensors must match except in dimension 1. Expected size 36 but got size 37 for tensor number 1 in the list.
       0/29      19.6G     0.9542      2.832      1.168        107        608: 100% 36/36 [00:24<00:00,  1.49it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:03<00:00,  1.55it/s]
                   all        150       2589      0.592       0.33      0.286      0.243
/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/29      26.3G     0.8313      1.453      1.049        184        608: 100% 36/36 [00:09<00:00,  3.79it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.91it/s]
                   all        150       2589       0.61      0.502      0.512      0.435

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/29      26.3G     0.9007      1.225      1.036        182        608: 100% 36/36 [00:09<00:00,  3.81it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  2.98it/s]
                   all        150       2589      0.648      0.508      0.544      0.461

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/29      26.3G     0.8908      1.174      1.038        111        608: 100% 36/36 [00:09<00:00,  3.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.12it/s]
                   all        150       2589      0.742      0.581      0.654      0.561

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/29      26.3G     0.8545      1.012      1.029        100        608: 100% 36/36 [00:09<00:00,  3.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.19it/s]
                   all        150       2589      0.757      0.603       0.67      0.587

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/29      26.3G     0.8489     0.9699      1.032        115        608: 100% 36/36 [00:09<00:00,  3.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.13it/s]
                   all        150       2589      0.759      0.541      0.633      0.537

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/29      26.3G     0.8759     0.9671      1.041        124        608: 100% 36/36 [00:10<00:00,  3.58it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.18it/s]
                   all        150       2589       0.74       0.61      0.689      0.596

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/29      26.3G     0.8278     0.8481      1.012        210        608: 100% 36/36 [00:09<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.16it/s]
                   all        150       2589      0.809      0.673      0.751      0.655

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/29      26.3G     0.8092     0.8299      1.018         95        608: 100% 36/36 [00:09<00:00,  3.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.17it/s]
                   all        150       2589      0.841      0.686      0.762      0.651

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/29      26.3G     0.7976     0.8141      1.011        176        608: 100% 36/36 [00:09<00:00,  3.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.09it/s]
                   all        150       2589      0.866      0.652      0.748      0.646

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/29      28.1G      0.792      0.823      1.013        188        608: 100% 36/36 [00:09<00:00,  3.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.16it/s]
                   all        150       2589      0.804      0.657      0.745      0.648

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/29      28.1G     0.7602     0.7826     0.9951        105        608: 100% 36/36 [00:09<00:00,  3.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.19it/s]
                   all        150       2589      0.863      0.684      0.772      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/29      28.1G     0.7536     0.7584     0.9925         91        608: 100% 36/36 [00:09<00:00,  3.80it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.06it/s]
                   all        150       2589      0.874      0.659       0.76      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/29      28.1G     0.7502     0.7709     0.9989        114        608: 100% 36/36 [00:09<00:00,  3.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.11it/s]
                   all        150       2589       0.83      0.666      0.753      0.664

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/29      30.1G     0.7438     0.7256     0.9831        175        608: 100% 36/36 [00:09<00:00,  3.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.21it/s]
                   all        150       2589       0.83      0.693      0.771      0.683
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/29      30.1G     0.6118     0.6454          1         43        608: 100% 36/36 [00:07<00:00,  4.87it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.19it/s]
                   all        150       2589      0.856      0.677      0.773      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/29      30.1G     0.6015     0.6118     0.9773         54        608: 100% 36/36 [00:07<00:00,  4.82it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.19it/s]
                   all        150       2589      0.849      0.594      0.716      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/29      30.1G      0.609      0.607     0.9855         44        608: 100% 36/36 [00:07<00:00,  4.92it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.08it/s]
                   all        150       2589      0.864      0.667      0.776      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/29      30.1G     0.5915     0.5926     0.9771         52        608: 100% 36/36 [00:07<00:00,  4.92it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.18it/s]
                   all        150       2589      0.823      0.675      0.758      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/29      30.1G     0.5892     0.5844     0.9645         62        608: 100% 36/36 [00:07<00:00,  4.94it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.15it/s]
                   all        150       2589      0.885      0.679      0.782      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/29      30.1G     0.5755     0.5641     0.9564         47        608: 100% 36/36 [00:07<00:00,  4.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.06it/s]
                   all        150       2589      0.839      0.701      0.789      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/29      30.1G     0.5739      0.553     0.9653         40        608: 100% 36/36 [00:07<00:00,  5.04it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.17it/s]
                   all        150       2589      0.867       0.67      0.774      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/29      30.1G     0.5643     0.5479     0.9648         45        608: 100% 36/36 [00:07<00:00,  4.75it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.16it/s]
                   all        150       2589      0.902      0.692      0.801      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/29      30.1G      0.549     0.5167     0.9439         48        608: 100% 36/36 [00:07<00:00,  4.93it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.17it/s]
                   all        150       2589      0.879      0.685      0.788      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/29      30.1G     0.5383     0.5041     0.9441         45        608: 100% 36/36 [00:07<00:00,  4.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.22it/s]
                   all        150       2589      0.861      0.727      0.806      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/29      30.1G     0.5287     0.4773     0.9335         52        608: 100% 36/36 [00:07<00:00,  4.84it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.19it/s]
                   all        150       2589        0.9      0.705      0.808      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/29      30.1G     0.5204     0.4653      0.927         49        608: 100% 36/36 [00:07<00:00,  4.94it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.23it/s]
                   all        150       2589      0.922       0.68      0.812      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/29      30.1G     0.5182      0.471     0.9361         42        608: 100% 36/36 [00:07<00:00,  4.89it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.22it/s]
                   all        150       2589      0.876      0.713      0.821      0.743

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/29      30.1G     0.5014     0.4521     0.9261         57        608: 100% 36/36 [00:07<00:00,  4.76it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.26it/s]
                   all        150       2589       0.86      0.738      0.825      0.754

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/29      30.1G     0.5037     0.4429      0.924         55        608: 100% 36/36 [00:07<00:00,  5.00it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:01<00:00,  3.14it/s]
                   all        150       2589      0.879      0.728      0.826      0.749

30 epochs completed in 0.103 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, saved as runs/train/exp/weights/last_striped.pt, 51.5MB
Optimizer stripped from runs/train/exp/weights/best.pt, saved as runs/train/exp/weights/best_striped.pt, 51.5MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
gelan-c summary: 467 layers, 25417128 parameters, 0 gradients, 102.5 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:13<00:00,  2.78s/it]
                   all        150       2589      0.859      0.738      0.824      0.754
              biowaste        150        169      0.901      0.753      0.857      0.773
             cardboard        150        391      0.866      0.744      0.819      0.731
     combustible waste        150         25      0.823       0.68      0.837      0.791
                 glass        150        594      0.923      0.791      0.883      0.803
                 metal        150         29      0.851       0.69      0.778      0.723
                 paper        150        679      0.862      0.754      0.822      0.745
               plastic        150        678      0.891      0.785      0.856      0.782
                 trash        150         24      0.755      0.708      0.743      0.686
Results saved to runs/train/exp


```




## Problemas encontrados y cómo fueron solucionados por el equipo
- Cambio del entorno de ejecución del Colab: El equipo migró el proyecto a un entorno local para superar las limitaciones de recursos en Colab y optimizar el entrenamiento de YOLOv9.
- Baja velocidad y rendimiento del Google Colab: Se trasladó el proyecto a un entorno local con hardware más potente, como una GPU dedicada, para mejorar la velocidad y rendimiento del modelo YOLOv9 durante el entrenamiento.




