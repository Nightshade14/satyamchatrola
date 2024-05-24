# Projects
---
## Automated Essay Evaluation with fine-tuned transformers and fine-tuned open-source LLMs like GPT-2.
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/CS224n-NLP-Solutions/tree/master/assignments/)

<div style="text-align: justify">Essays represent a person's critical thinking. It provides a glimpse into a person's mind. This serves as an approach to cultivate thoughts and ideas, and also a way to evaluate a person. This method is used by academia and educational institutions. Essay evaluation has been performed manually by humans; which is subject to many factors and could lead to inconsistency.<br><br>

So, the project explores ways to automate essay evaluation with Transformers like BERT, DeBERTa-v3 and Large Language Models (LLMs) like GPT-2. By automating this task, we can reduce many hours of human labour, mental fatigue, evaluation time and errors in evaluation. We leveraged multiple <b>Parameter Efficient Fine Tuning (PEFT)</b> techniques to fine-tune LLMs on small amount of data. We used many other Transfer Learning techniques to find a good set of parameters fine-tuned on our dataset. Along with these techniques, the project also used <b>dynamic learning rate</b> with <b>cosine annealing</b> and <b>warm-up</b>. The project uses <b>Cohen's Kappa score</b> as a metric to evaluate the model. With all these techniques, the model achieved an impressive Kappa score of 81.7.</div>

<!-- <center><img src="images/nlp.png"/></center> -->

---
## AI based classification of New York City Open-Source Noise data into 10 categories. 
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/CS224n-NLP-Solutions/tree/master/assignments/)

<div style="text-align: justify">New York City is a metropolitan city with numerous sources for noise pollution. These noise levels are harmful to hearing health of people. For instance, it has been estimated that 9 out of 10 adults in New York City (NYC) are exposed to excessive noise levels, i.e., beyond the limit of what the EPA considers to be harmful. When applied to U.S. cities of more than 4 million inhabitants, such estimates extend to over 72 million urban residents. So, it is important to identify noise sources to mitigate it. The noise data was sourced by NYU’s Music and Audio Research Lab; funded by NYU’s Centre for Urban Science.<br><br>

The project leverages multiple Machine Learning and Deep Learning techniques to identify noise categories with 86% accuracy. The project uses Deep Convolutional Network to understand and classify audio data. By leveraging this project, one can identify the noise sources in the city. This project can be extended to real-time identification by placing noise sensing devices across the city and that data can be leveraged to re-position appropriate officials to minimize noise pollution.</div>

<!-- <center><img src="images/BERT-classification.png"/></center> -->

---
## Migrating NYU Vida Lab's Wildlife Trafficking Prevention Project Pipeline onto Apache PySpark
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/facebook-detect-food-trends)

<div style="text-align: justify">The global wildlife trafficking industry is valued between $7 billion and $23 billion annually. Wildlife crime is estimated to cause a loss of $1 trillion to the global economy annually, considering the environmental damage, loss of biodiversity, and impact on local economies. In order to prevent the selling of illegally trafficed animal parts and products, NYU's Vida Lab is working on a project to identify these animal products, selling on the internet. This can be used to take necessary actions to identify the seller of such products. <br><br>

The project uses machine learning based and rules based crawling and extraction of the internet in-order to find relevant data. Then it uses a zero-shot, multi-modal, AI model to identify the products. It also uses AI to generated clean data. This whole data ETL pipeline was not scalable. So, we scaled the complete data pipeline by migrating it onto Apache Spark with PySpark. This made the pipeline more robust and faster. With our efforts, we achieved <b>160% speedup</b>.</div>

<!-- <center><img src="images/fb-food-trends.png"></center> -->

---
## End to End web-app: Personal Health Assistant for Diabetics
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/detect-spam-messages-nlp/blob/master/detect-spam-nlp.ipynb)

<div style="text-align: justify">As of 2021, approximately 537 million adults (20-79 years) worldwide have diabetes. This number is expected to rise to 643 million by 2030 and 783 million by 2045. Type 2 Diabetes accounts for about 90-95% of all diabetes cases. It is primarily related to lifestyle factors and genetic predisposition. The global economic cost of diabetes in 2021 was estimated at $966 billion, a 316% increase over the past 15 years. Diabetes is a leading cause of death, responsible for approximately 6.7 million deaths in 2021, equating to 1 death every 5 seconds.<br><br>

Type-2 Diabetes is preventable with lifestyle changes. So, the project <b>estimates</b> a person's chances of <b>developing type-2 diabetes</b> on the basis of multiple factors. The project is an end-to-end webapp. The model was trained on Behavioral Survey dataset (BRFSS). Additionally, the webapp calculates a person's <b>BMI</b>, and informs them about their <b>lifestyle</b>. It also provides <b>MD Physician recommended food and lifestyle tips.</b> Alongwith this, it also provides a list of good quality medicines for diabetic people.<br><br>

The project uses ensemble of multiple machine learning and deep learning models. And the congregation of the decision of multiple models leads to a final and conclusive estimate. For applications of medical domain, accuracy is not a good metric to test the model. The <b>ideal metric</b> is the <b>combination of high precision, recall and specificity</b>. And the model achieves <b>95.8% precision and recall, and 99.4% specificity</b>.</div>
<!-- <center><img src="images/detect-spam-nlp.png"/></center> -->

