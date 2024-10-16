# Перемножение матриц

## Как запусктить?

```
make build - комплирует исходники в бинарник
make run - компилирует и запускает сразу же получившийся бинарник
```

Задача: реализовать алгоритм перемножения матриц

Язык: C++

Входные данные: 2 матрицы размером от 100х100 до 2000х2000 каждая.

Выходные данные: проверка корректности перемножения + время вычисления

Реализация должна содержать 2 функции перемножения матриц: на CPU и на GPU с применением CUDA.

## Описание проделанной работы

Реализованы три алгоритма:
1. Native C - прямое переменожение матриц без распаралеливания
2. CUDA CPU
3. CUDA GPU

Каждый протестирован в соотетствии с методологией тестирования

## Что распаралелено? Как работает?

Каждая нить расчитывает сумму для каждого элемента массива на основе переданных
в неё исходных массивов m1 и m2

## Метод тестирования

Для разных рамерностей от 100х100 до 2000х2000 с шагом 100 генерируются две случайные матрицы.
Эти матрицы передаются для каждого алгоритма

Первоочерёдно считается Native C, так как результат перемножения этой функции, используется
для верификации данных с алгоритмов, использующих технологию CUDA.

Для каждого алгоритма считается дельта при помощи функции time

## Тестовое окружение

```bash
           .-/+oossssoo+/-.
        `:+ssssssssssssssssss+:`
      -+ssssssssssssssssssyyssss+-
    .ossssssssssssssssssdMMMNysssso.
   /ssssssssssshdmmNNmmyNMMMMhssssss/      root@Nia 
  +ssssssssshmydMMMMMMMNddddyssssssss+     -------- 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    OS: Ubuntu 22.04.4 LTS on Windows 10 x86_64 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Kernel: 5.15.153.1-microsoft-standard-WSL2 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Uptime: 57 secs 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   Packages: 916 (dpkg) 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   Shell: bash 5.1.16 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Theme: Adwaita [GTK3] 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Icons: Adwaita [GTK3] 
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/    Terminal: Relay(10) 
  +sssssssssdmydMMMMMMMMddddyssssssss+     CPU: AMD Ryzen 7 5800H with Radeon Graphics (16) @ 3.193GHz 
   /ssssssssssshdmNNNNmyNMMMMhssssss/      GPU: 867a:00:00.0 Microsoft Corporation Device 008e 
    .ossssssssssssssssssdMMMNysssso.       Memory: 369MiB / 7612MiB 
      -+sssssssssssssssssyyyssss+-
        `:+ssssssssssssssssss+:`                                   
            .-/+oossssoo+/-.                                       
--- 
```

## Результаты тестирования

```bash                            
                                           sec
+-----------+-------------+-------------+--------+-----------+
|Matrix Size|Native C     |CUDA CPU     |CUDA GPU|CPU/GPU    |
+-----------+-------------+-------------+--------+-----------+
|100x100    |0.003864 sec |0.003006 sec |0.000099|30.363636  |
+-----------+-------------+-------------+--------+-----------+
|200x200    |0.032310 sec |0.025438 sec |0.000203|125.310345 |
+-----------+-------------+-------------+--------+-----------+
|300x300    |0.108075 sec |0.081811 sec |0.000219|373.566210 |
+-----------+-------------+-------------+--------+-----------+
|400x400    |0.252360 sec |0.194191 sec |0.000333|583.156156 |
+-----------+-------------+-------------+--------+-----------+
|500x500    |0.516216 sec |0.385043 sec |0.000555|693.771171 |
+-----------+-------------+-------------+--------+-----------+
|600x600    |0.879765 sec |0.674331 sec |0.000660|1021.713636|
+-----------+-------------+-------------+--------+-----------+
|700x700    |1.511329 sec |1.052961 sec |0.000787|1337.942821|
+-----------+-------------+-------------+--------+-----------+
|800x800    |2.144588 sec |1.580599 sec |0.000771|2050.063554|
+-----------+-------------+-------------+--------+-----------+
|900x900    |3.045145 sec |2.325119 sec |0.005592|415.793813 |
+-----------+-------------+-------------+--------+-----------+
|1000x1000  |4.409852 sec |3.086831 sec |0.005792|532.947341 |
+-----------+-------------+-------------+--------+-----------+
|1100x1100  |5.461150 sec |4.119951 sec |0.006133|671.767650 |
+-----------+-------------+-------------+--------+-----------+
|1200x1200  |7.646381 sec |5.306543 sec |0.006501|816.265651 |
+-----------+-------------+-------------+--------+-----------+
|1300x1300  |9.393658 sec |7.338033 sec |0.006888|1065.335801|
+-----------+-------------+-------------+--------+-----------+
|1400x1400  |11.985317 sec|8.998521 sec |0.007291|1234.195721|
+-----------+-------------+-------------+--------+-----------+
|1500x1500  |14.095731 sec|12.216470 sec|0.007768|1572.666066|
+-----------+-------------+-------------+--------+-----------+
|1600x1600  |18.732633 sec|13.802823 sec|0.008380|1647.114916|
+-----------+-------------+-------------+--------+-----------+
|1700x1700  |21.672809 sec|19.020062 sec|0.008370|2272.408841|
+-----------+-------------+-------------+--------+-----------+
|1800x1800  |26.461464 sec|20.776487 sec|0.009140|2273.138621|
+-----------+-------------+-------------+--------+-----------+
|1900x1900  |29.023007 sec|27.578086 sec|0.009647|2858.721468|
+-----------+-------------+-------------+--------+-----------+
|2000x2000  |37.437791 sec|26.880905 sec|0.009854|2727.918104|
+-----------+-------------+-------------+--------+-----------+
```