---
layout: default
title: "Home"
---

# Projects

## RAG Microservice: Research-mate chatbot

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Nightshade14/research-mate)

<div style="text-align: justify">

<em>Technologies: Python, FastAPI, PyTorch, Transformers, GCP, Pinecone, Prompt Engineering, Llamma 3.2, chat UI frontend</em><br><br>

LLMs introduced AI models with good general reasoning and intelligence. They are trained on public data. So, they do not have knowledge about one's private data. What if we can teach the LLM about private data? Augmenting LLMs with private data introduces large amount of possibilities. For instance, this project enables LLMs to learn about our collection of research papers and then answer queries about that research paper. We can also use it to find the relevant research papers.<br><br>

The project is a <b>context-aware RAG-based chatbot</b> leveraging Pinecone vector database, enabling <b>semantic search</b> across 2,700 research papers with <b> 95% query relevance</b> and reducing response time to seconds. It leverages <b>Anthropic AIs</b> latest research on RAG which is <b>Contextual Retrieval</b>. It is optimized for model performance with <b>Binary Quantization</b>, achieving <b>7x speedup in inference time</b> and <b>85% reduction in memory</b>. As the data size increases, the the embeddings would out-grow the memory size and disk IO for matching is orders of magnitude slower than in-memory operations. So, we decrease the storage amount of embeddings while keeping them relevant to make them store in memory for fast matching and low inference latency. The project plans to improve upon <b>ElasticSearch</b>'s beta feature: <b>Better Binary Quantization (BBQ)</b> which provides better savings but at a more harsh <b>memory-recall trade-off</b>.</div><br>

## End to End WebApp: Automated Essay Evaluation with fine-tuned transformers and fine-tuned open-source LLMs like GPT-2.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Nightshade14/Fined-tuned-LLM-like-GPT2-and-Transformers-based-Automated-Essay-Evaluation)

<div style="text-align: justify">

<em>Technologies: Python, PyTorch, ONNX, TensorRT, FastAPI, AWS, MLflow, Evidently, Frontend</em><br><br>

Essays represent a person's critical thinking. It provides a glimpse into a person's mind. This serves as an approach to cultivate thoughts and ideas, and also a way to evaluate a person. This method is used by academia and educational institutions. Essay evaluation has been performed manually by humans; which is subject to many factors and could lead to inconsistency.<br><br>

So, the project explores ways to automate essay evaluation with Transformers like BERT, DeBERTa-v3 and Large Language Models (LLMs) like GPT-2. By automating this task, we can reduce many hours of human labour, mental fatigue, evaluation time and errors in evaluation. We leveraged multiple <b>Parameter Efficient Fine Tuning (PEFT)</b> techniques to fine-tune LLMs on small amount of data. We used many other Transfer Learning techniques to find a good set of parameters fine-tuned on our dataset. Along with these techniques, the project also used <b>dynamic learning rate</b> with <b>cosine annealing</b> and <b>warm-up</b>. The project uses <b>Cohen's Kappa score</b> as a metric to evaluate the model. With all these techniques, the model achieved an impressive Kappa score of 81.7.</div><br>

## Open Source Project: mAIgic

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Nightshade14/mAIgic-nyu/tree/conv2_v2)

<div style="text-align: justify">

<em>Technologies: Python, OpenAI Function Calling, CircleCI, Pytest, MyPy, Ruff, uv</em><br><br>

LLMs introduced AI models with good general reasoning and intelligence. They are trained on public data. So, they do not have knowledge about one's private data. What if we can teach the LLM about private data? <b>Augmenting LLMs with private data</b> introduces large amount of possibilities. For instance, this project enables LLMs to learn about emails and extract meaningful tasks from them and add them to a Trello board. This also creates a private and query-able knowledge base. The project plans to extend functionalities by adding more tools and functionalities.<br><br>

mAIgic is a smart AI assistant with generalized knowledge and reasoning capabilities. The project leverages <b>OpenAI’s function calling</b>, achieving <b>95% accuracy in task extraction</b> and automated Trello board updates, <b>reducing manual email processing time by 70%</b>. This is a AI-software project, engineered with a <b>production-grade Python API</b> with <b>100% test coverage</b> through <b>CircleCI pipeline</b>, implementing comprehensive <b>static type checking with mypy</b>, and leveraged <b>SQLite</b>-based conversation tracking system. The project uses a modern Rust-based python package manager called uv for efficient dependency and package management.</div><br>

## Open Source Project: Migrating NYU Vida Lab's Wildlife Trafficking Prevention Project Pipeline onto Apache PySpark.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Nightshade14/NYU-VIDA-Wildlife-Pipeline)

<div style="text-align: justify">

<em>Technologies: Python, PySpark, AWS, MinIO, Databricks.Koalas, AWS</em><br><br>

The global wildlife trafficking industry is valued between $7 billion and $23 billion annually. Wildlife crime is estimated to cause a loss of $1 trillion to the global economy annually, considering the environmental damage, loss of biodiversity, and impact on local economies. In order to prevent the selling of illegally trafficed animal parts and products, NYU's Vida Lab is working on a project to identify these animal products, selling on the internet. This can be used to take necessary actions to identify the seller of such products. <br><br>

The project uses machine learning based and rules based crawling and extraction of the internet in-order to find relevant data. Then it uses a zero-shot, multi-modal, AI model to identify the products. It also uses AI to generated clean data. This whole data ETL pipeline was not scalable. So, we scaled the complete data pipeline by migrating it onto Apache Spark with PySpark. This made the pipeline more robust and faster. With our efforts, we achieved <b>160% speedup</b>.</div><br>

## End to End web-app: Personal Health Assistant for Diabetics hosted with auto-deployment on Heroku.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Nightshade14/Personal-Health-Assistant-for-Diabetics)

<div style="text-align: justify">

<em>Technologies: Python, TensorFlow, Flask, Heroku, Frontend, HTML, CSS, JavaScript, Bootstrap</em><br><br>

As of 2021, approximately 537 million adults (20-79 years) worldwide have diabetes. This number is expected to rise to 643 million by 2030 and 783 million by 2045. Type 2 Diabetes accounts for about 90-95% of all diabetes cases. It is primarily related to lifestyle factors and genetic predisposition. The global economic cost of diabetes in 2021 was estimated at $966 billion, a 316% increase over the past 15 years. Diabetes is a leading cause of death, responsible for approximately 6.7 million deaths in 2021, equating to 1 death every 5 seconds.<br><br>

Type-2 Diabetes is preventable with lifestyle changes. So, the project <b>estimates</b> a person's chances of <b>developing type-2 diabetes</b> on the basis of multiple factors. The project is an end-to-end webapp. The model was trained on Behavioral Survey dataset (BRFSS). Additionally, the webapp calculates a person's <b>BMI</b>, and informs them about their <b>lifestyle</b>. It also provides <b>MD Physician recommended food and lifestyle tips.</b> Alongwith this, it also provides a list of good quality medicines for diabetic people.<br>

The project uses ensemble of multiple machine learning and deep learning models. And the congregation of the decision of multiple models leads to a final and conclusive estimate. For applications of medical domain, accuracy is not a good metric to test the model. The <b>ideal metric</b> is the <b>combination of high precision, recall and specificity</b>. And the model achieves <b>95.8% precision and recall, and 99.4% specificity</b>.</div><br>

## AI based classification of New York City Open-Source Noise data into 10 categories.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Nightshade14/AI-based-identification-of-New-York-City-Open-Source-Noise-data)

<div style="text-align: justify">

<em>Technologies: Python, PyTorch, librosa, Convolutional Neural Network (CNN)</em><br><br>

New York City is a metropolitan city with numerous sources for noise pollution. These noise levels are harmful to hearing health of people. For instance, it has been estimated that 9 out of 10 adults in New York City (NYC) are exposed to excessive noise levels, i.e., beyond the limit of what the EPA considers to be harmful. When applied to U.S. cities of more than 4 million inhabitants, such estimates extend to over 72 million urban residents. So, it is important to identify noise sources to mitigate it. The noise data was sourced by NYU’s Music and Audio Research Lab; funded by NYU’s Centre for Urban Science.<br><br>

The project leverages multiple Machine Learning and Deep Learning techniques to identify noise categories with 86% accuracy. The project uses Deep Convolutional Network to understand and classify audio data. By leveraging this project, one can identify the noise sources in the city. This project can be extended to real-time identification by placing noise sensing devices across the city and that data can be leveraged to re-position appropriate officials to minimize noise pollution.</div><br>

## Leveraging custom Deep Residual Convolutional Networks under 5M parameters to classify images into 10 categories.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Nightshade14/Deep-Residual-Convolutional-Network)

<div style="text-align: justify">

<em>Technologies: Python, PyTorch, Convolutional Neural Network (CNN), ResNet50, Fine-tuning</em><br><br>

Conventional Deep Neural Networks have multiple layers. After a certain depth, the model starts suffering from issues like vanishing / exploding gradients, overfitting, etc. By leveraging Residual Connections / skip Connections, we can create deeper networks without overfitting. Additionally, the <b>improved gradient flow and regularization</b> also aids with the cause.<br><br>

The project uses a Deep Residual Convolutional Neural Network for identiying 10 types of RGB images. The images were RGB and of low resolution of 32x32 pixels. And on top of it, we did not use any pre-trained weights like of the imagenet dataset. We employed multiple data augmentation and feature enhancing transformations on the image. The model has around 4.7M parameters. The model is good balance between deep and shallow networks. So, it performs better on new images while still remembering the training images. <b>Dropouts and L2 regularization</b> also helps with the overfitting issue. The model has an accuracy of 81%. </div><br>

## Training Generative Adversarial Network (GAN) to generate images of clothes.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Nightshade14/Training-GAN-to-generate-clothes)

<div style="text-align: justify">

<em>Technologies: Python, PyTorch, Convolutional Generative Adversarial Networks</em><br><br>

Generative Adversarial Networks is a result of two different models trying to outperform each other. GANs create images from random noise provided to them. The Generator tries to outperform the Discriminator and the Discriminator tries to outperform Generator. To train GANs, it is imperative that GANs reach a Saddle point. The Saddle point provides an equilibrium between Discriminator and Generator.<br><br>

The project uses a deep convolutional generative adversarial network. It has 2 convolutional layers and uses activation functions like <b>LeakyRelu and Tanh</b>. By observing the training progress of the GAN, the losses of Generator and Discriminator <b>converges smoothly</b>. This indicates that the model has found the <b>saddle point</b>.</div><br>
