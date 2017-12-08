# Развертывание датасета

## Инструкция для Anaconda на Windows:

Сначала ставим tensorflow, как environment

```
conda create -n tensorflow python=3.6
activate tensorflow
conda install -c conda-forge tensorflow
```

Добавляем в PATH папки из Anaconda .../envs/tensorflow и .../envs/tensorflow/Scripts

Ставим pillow

```
conda install -n tensoflow pillow
```

Запускаем процесс создания датасета
```
python unzip_dataset.py data.zip
```

