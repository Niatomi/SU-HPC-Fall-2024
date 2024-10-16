# Перемножение матриц

Задача: реализовать алгоритм перемножения матриц

Язык: C++

Входные данные: 2 матрицы размером от 100х100 до 2000х2000 каждая.

Выходные данные: проверка корректности перемножения + время вычисления

Реализация должна содержать 2 функции перемножения матриц: на CPU и на GPU с применением CUDA.

## Описание проделанной работы

Реализовано три алгоритма:
1. Native C - прямое переменожение матриц без распаралеливания
2. CUDA CPU
3. CUDA GPU

## Метод тестирования

Для разных рамерностей от 100х100 до 2000х2000 с шагом 100 генерируются две случайные матрицы.
Эти матрицы передаются для каждого алгоритма

Первоочерёдно считается Native C, так как результат перемножения этой функции, используется
для верификации данных с алгоритмов, использующих технологию CUDA.

Для каждого алгоритма считается дельта при помощи функции time

## Тестовое окружение

```bash
       _,met$$$$$gg.          nia@nia-pc 
    ,g$$$$$$$$$$$$$$$P.       ---------- 
  ,g$$P"     """Y$$.".        OS: Debian GNU/Linux 12 (bookworm) x86_64 
 ,$$P'              `$$$.     Host: OMEN Laptop 15-en1xxx 
',$$P       ,ggs.     `$$b:   Kernel: 6.1.0-26-amd64 
`d$$'     ,$P"'   .    $$$    Uptime: 5 mins 
 $$P      d$'     ,    $$P    Packages: 2259 (dpkg), 6 (flatpak), 9 (snap) 
 $$:      $$.   -    ,d$$'    Shell: zsh 5.9 
 $$;      Y$b._   _,d$P'      Resolution: 1920x1080 
 Y$$.    `.`"Y$$$$P"'         WM: i3 
 `$$b      "-.__              Theme: Raleigh [GTK2/3] 
  `Y$$                        Icons: hicolor [GTK2/3] 
   `Y$$.                      Terminal: kitty 
     `$$b.                    CPU: AMD Ryzen 7 5800H with Radeon Graphics (16) @ 3.200GHz 
       `Y$$b.                 GPU: NVIDIA GeForce RTX 3070 Mobile / Max-Q 
          `"Y$b._             GPU: AMD ATI Radeon Vega Series / Radeon Vega Mobile Series 
              `"""            Memory: 3346MiB / 15329MiB 

--- 

./deviceQuery 
./deviceQuery Starting...
 CUDA Device Query (Runtime API) version (CUDART static linking)
Detected 1 CUDA Capable device(s)
Device 0: "NVIDIA GeForce RTX 3070 Laptop GPU"
  CUDA Driver Version / Runtime Version          12.4 / 12.6
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 7967 MBytes (8353677312 bytes)
  (040) Multiprocessors, (128) CUDA Cores/MP:    5120 CUDA Cores
  GPU Max Clock rate:                            1290 MHz (1.29 GHz)
  Memory Clock rate:                             6001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.4, CUDA Runtime Version = 12.6, NumDevs = 1
Result = PASS
```

## Результаты тестирования

...