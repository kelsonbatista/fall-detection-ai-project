# Project: Fall Detection in Video (YOLO + LSTM) - AI

> Capstone Project for the **MBA in Artificial Intelligence and Big Data (University of São Paulo - USP/ICMC)**
> Projeto do **MBA em Inteligência Artificial e Big Data (Universidade de São Paulo - USP/ICMC)** 

**Keywords:** Fall Detection, Human Activity Recognition, Pattern Recognition, Temporal Classification, Convolutional Neural Networks, Domestic Accidents

**Palavras-chave:** Detecção de quedas, Reconhecimento de Atividade Humana, Classificação Temporal, Reconhecimento de padrões, Redes Neurais Convolucionais, Acidentes Domésticos 

## About the Project / Sobre o Projeto

In recent years, the number of people living alone has increased significantly, making them more vulnerable to emergency situations such as falls caused by fainting, tripping, or illness. In many cases, the individual may be unable to get up or call for help, and may even lose consciousness. Delays in assistance can lead to serious complications, permanent injuries, or even death. Systems based on wearable or environmental sensors have proven to be inefficient, invasive, and unreliable. In this context, this project aims to explore the use of artificial intelligence through computer vision to detect falls in videos and in real time in home environments, contributing to the safety and quality of life of millions of people who live or find themselves alone.

---

Nos últimos anos, o número de pessoas que moram sozinhas tem crescido de forma expressiva e estão sujeitas a situações de emergências nos casos de quedas, seja por desmaios, tropeços ou enfermidades. Em muitos casos, a pessoa fica impossibilitada de se levantar ou pedir ajuda, podendo até perder a consciência, e a demora no socorro pode levar a complicações graves, sequelas permanentes ou até mesmo ao óbito. Sistemas com sensores vestíveis ou ambientais se mostram ineficientes, invasivos e não confiáveis. Diante disso, esse projeto tem o objetivo explorar o uso da inteligência artificial através da visão computacional para detectar quedas em tempo real nos ambientes domésticos, contribuindo para a segurança e a qualidade de vida de milhões de pessoas que vivem ou se encontram sozinhas.


## Overview / Visão Geral

The system was implemented with a two-stage pipeline:

- The first stage consisted of training with **YOLOv11-pose, a Convolutional Neural Network** for detection and pose estimation, and extracting features from each frame as 128-dimensional vectors containing static (position) and dynamic (velocity) data.

- The second stage involved using these vectors, arranged in 30-frame sequences (windows), to train a **temporal classification model using a Recurrent Neural Network, LSTM**, capable of analyzing and classifying the action contained within that time window.

---

O sistema foi implementado com um pipeline em duas etapas principais:

A primeira etapa consistiu no treinamento com o **YOLOv11-pose, uma Rede Neural Convolucional** para detecção e estimativa de pose, e extração de características de cada frame, vetores de 128 dimensões contendo dados estáticos (posição) e dinâmicos (velocidade).

A segunda etapa envolveu a utilização dos vetores em formato de sequências (janelas) de 30 frames para treinar um **modelo de classificação temporal em uma Rede Neural Recorrente, LSTM**, capaz de analisar e classificar a ação contida nessa janela de tempo.










Base de Dados


Arquitetura do Pipeline


Tecnologias


Como reproduzir


Resultados


Cite




Sistema de visão computacional para **detecção automática de quedas** em ambientes domésticos, combinando **YOLO (detecção/pose)** e **LSTM (classificação temporal)**.

> 🎓 Projeto integrante do **TCC - MBA em Inteligência Artificial e Big Data (USP - ICMC)**  

---

## 📖 Sumário
- [Visão Geral](#-visão-geral)
- [Arquitetura do Pipeline](#-arquitetura-do-pipeline)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como Reproduzir](#-como-reproduzir)
  - [1) Preparação e Organização do Dataset](#1-preparação-e-organização-do-dataset)
  - [2) Treino do YOLO (detecção/pose)](#2-treino-do-yolo-detecçãopose)
  - [3) Extração de janelas e Treino do LSTM](#3-extração-de-janelas-e-treino-do-lstm)
  - [4) Avaliação e Testes em Vídeos](#4-avaliação-e-testes-em-vídeos)
  - [5) Teste em Tempo Real (Câmera)](#5-teste-em-tempo-real-câmera)
- [Resultados Principais](#-resultados-principais)
- [Limitações e Próximos Passos](#-limitações-e-próximos-passos)
- [Agradecimentos](#-agradecimentos)
- [Licença](#-licença)

---

## 📌 Visão Geral
Este projeto aborda o problema de **Reconhecimento de Atividade Humana (HAR)** aplicado à **detecção de quedas**.  

A solução opera em duas etapas principais:  
1. **YOLOv11n-pose** → detecção de pessoas + keypoints (18 pontos de pose).  
2. **LSTM** → análise temporal de janelas de frames para classificar ações:  
   - `0 - no_fall`  
   - `1 - attention`  
   - `2 - fall`  

---

## 🧩 Arquitetura do Pipeline
```mermaid
flowchart TD
    A[Vídeos Originais] --> B[Pré-processamento 01-09 + 13]
    B --> C[YOLO - Treino e Predições (10-12)]
    C --> D[Extração de Features e Janelas (14)]
    D --> E[LSTM - Treinamento e Avaliação (15-16)]
    E --> F[Testes em Vídeos (17)]
    E --> G[Teste em Câmera (18)]

