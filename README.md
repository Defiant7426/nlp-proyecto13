# Proyecto 13: Resumen de Texto con Seq2Seq y Atención (nlp-proyecto13)

Autor: De la Cruz Valdiviezo, Pedro Luis David

Este repositorio contiene el desarrollo del Proyecto 13, enfocado en la construcción y evaluación de un modelo de **resumen de texto automático** utilizando una arquitectura **Sequence-to-Sequence (Seq2Seq)** con un mecanismo de **atención (Bahdanau)**. El proyecto se realiza en un plazo intensivo de 14 días.

## Descripción del Proyecto

El objetivo principal es implementar un modelo capaz de generar resúmenes concisos y coherentes a partir de textos más largos (artículos de noticias). Se explorarán técnicas de preprocesamiento de texto, la implementación del modelo Seq2Seq con atención en PyTorch, y la evaluación del rendimiento utilizando métricas estándar como ROUGE.

## Dataset Utilizado

Este proyecto utilizará el dataset **CNN / Daily Mail**. Este es un corpus ampliamente utilizado para tareas de resumen de text.

* **Fuente:** Hugging Face Datasets
* **Identificador:** `abisee/cnn_dailymail`
* **Enlace:** [https://huggingface.co/datasets/abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
* **Descripción:** Consiste en artículos de noticias de CNN y Daily Mail (campo `article`) junto con varios puntos destacados o resúmenes escritos por humanos (campo `highlights`).

La exploración inicial de datos (longitudes, ejemplos, estructura) se encuentra en el notebook `notebooks/01_data_exploration.ipynb`.

## Estructura del Repositorio