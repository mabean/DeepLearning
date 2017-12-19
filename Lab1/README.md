# Лабораторная работа №1

Выполнил:

 - Беспалов Максим
 - ННГУ, и-т ИТММ, каф. МОСТ, группа 381603-м4

 # Сборка

 Сборка осуществляется с помощью CMake. Для ускорения некоторых стадий используется OpenMP.

```
cd Lab1
mkdir build
cmake -H. -Bbuild
```

В случае msvc
```
cmake --build ./build --config Release
```

В случае всего остального (не тестировал, нужно, чтобы бинарник оказался в папке ./build/<любая директория>/ для корректности относительных путей)
```
cmake ./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build
```

 # Использование

 BespalovLab1 [HIDDENLAYERSIZE] [EPOCHCOUNT] [STOPCRITERIA] [LEARNRATE]