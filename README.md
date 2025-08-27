# Project: Fall Detection in Video (YOLO + LSTM) - AI Project


## About the Project / Sobre o Projeto

In recent years, the number of people living alone has increased significantly, making them more vulnerable to emergency situations such as falls caused by fainting, tripping, or illness. In many cases, the individual may be unable to get up or call for help, and may even lose consciousness. Delays in assistance can lead to serious complications, permanent injuries, or even death. Systems based on wearable or environmental sensors have proven to be inefficient, invasive, and unreliable. In this context, this project aims to explore the use of artificial intelligence through computer vision to detect falls in videos and in real time in home environments, contributing to the safety and quality of life of millions of people who live or find themselves alone.

---

Nos últimos anos, o número de pessoas que moram sozinhas tem crescido de forma expressiva e estão sujeitas a situações de emergências nos casos de quedas, seja por desmaios, tropeços ou enfermidades. Em muitos casos, a pessoa fica impossibilitada de se levantar ou pedir ajuda, podendo até perder a consciência, e a demora no socorro pode levar a complicações graves, sequelas permanentes ou até mesmo ao óbito. Sistemas com sensores vestíveis ou ambientais se mostram ineficientes, invasivos e não confiáveis. Diante disso, esse projeto tem o objetivo explorar o uso da inteligência artificial através da visão computacional para detectar quedas em vídeos e em tempo real nos ambientes domésticos, contribuindo para a segurança e a qualidade de vida de milhões de pessoas que vivem ou se encontram sozinhas.

---





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

