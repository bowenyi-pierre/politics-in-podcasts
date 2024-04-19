# Politics in Podcasts
**This folder contains code, main datasets, and plots for 486 final project on political leaning detection.**

Under Code directory, all code files are named by the time they are implemented. For example, data_extraction was done at the beginning of the semester, so its file name begins with 01. Notably, our final model's code is implemented in 03_chunk_and_train. Files begin with 04 are our distilBERT code, which we didn't decide to use. The rest of analysis are performed in files beginning with 05 and 06.

## Here's a high-level overview for the functionality of each file under Code:
<br> *01_data_extraction.ipynb:* Extract political transcripts from a large raw dataset. We explored three strategies to label political transcripts and chose the last one, which yieled the most data to work on.
<br> *02_preprocessing.ipynb:* Detailed preprocessing done on transcripts, including removing non-English and mistranscribed data. 
<br> *03_chunk_and_train.ipynb:* <br>1. Chunk all transcripts into text blocks, including the annotated transcripts, that are easier to train and analyse; <br>2. Finetuned a MiniLM model solely on annotated transcript chunks; <br>3. Finetuned a RoBERTa-base on annotated samples as well as labeled Reddit data.
<br> *04_model_training.py, 04_podcast_classifier_distilBERT.ipynb:* Code to finetune a distilBERT model. After seeing that the finetuned model didn't perform considerably well even on a clean Reddit dataset, we didn't plan to move on and turned to larger models such as RoBERTa. Please notice that 04_model_training should have been named as RedditDataset because it's more of a dataset class. 
<br> *05_political_analysis_in_wild.ipynb:* Answered our research question: what's the prevalence of left/right wing political discussion across podcast categories.
<br> *06_tfidf.ipynb:* Answered our research question: what terms receive the highest attention in right/left wing political discussion. 


## Note:
1. The GitHub commit history may not reflect the task distribution in the group. For example, Divya and Cullen tested their code locally and sent to Alan, who later pushed to GitHub. 
2. Due to the size of the raw podcast dataset, we are not able to upload it through GitHub. However, everyone is welcome to contact Benjamin Litterer (blitt@umich.edu) or Bowen Yi (bowenyi@umich.edu) for more details regarding the dataset.
As a result, we uploaded the dataset of political podcasts as well as the Reddit dataset on Kaggle, which are considerably smaller. 
3. Because the majority of this project is run on GPU at UMSI, the coding environment might not be compatible to local environment. When testing on local machine, please make sure to change the environment and CUDA variable.
4. Our finetuned RoBERTa model are available on Huggingface at https://huggingface.co/bowenyi/political-learning-RoBERTa. Due to the size of model files, they are not uploaded to GitHub, but can be easily retrived on HuggingFace: https://huggingface.co/bowenyi/political-learning-RoBERTa/tree/main.


## Core libraries used:
1. PYCLD2 for language detection: https://pypi.org/project/pycld2/
2. Sklearn for metric calculation and datset split: https://scikit-learn.org/
3. HuggingFace Transformers: https://huggingface.co/transformers/v3.0.2/index.html
4. Pandas: https://pandas.pydata.org/
5. Numpy: https://numpy.org/
6. Multiprocessing: https://docs.python.org/3/library/multiprocessing.html


## Credits:
Alan Salazar, Bowen Yi, Cullen Ye, Divya Reddy, Jacob Harwood
