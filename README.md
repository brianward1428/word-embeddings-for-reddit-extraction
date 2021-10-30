


# Word Embeddings for Identifying Company Names and Stock Tickers from Social Media Posts 

## Introduction
This project aims to create a tool which can be used to extract possible market insights on companies which are discussed on social media platforms such as Reddit. The main goal is to identify company names/tickers from raw text. Assuming that these company names would be used in similar context, word embeddings which are trained based on context were used to identify these targets. Using this approach I was able to extract company names from nearly 300k of the ~1.16 million reddit posts consisting of over 800 unique values. Testing on a 1000 record ground truth this model was able to achieve 87% accuracy. The second part of this project aimed to classify reddit posts into one of the following three sentiments: negative, neutral, or positive. I was able to achieve ~70% accuracy for this task by use of naive bayes classification.

For an abridged walk through of the code please see my [Medium post on the project here.](https://brianward1428.medium.com/using-word-embeddings-to-identify-company-names-and-stock-tickers-f194e3648a66?source=friends_link&sk=6ae0387caed7b7272163b0cbf674da9e)
For a more detailed paper on word embeddings and the project as whole [please see the full paper here.](https://github.com/brianward1428/word-embeddings-for-reddit-extraction/blob/master/project_writeup.pdf) 

## Instructions

1. Make sure you have setup your environment using the requirements.txt file. 

2. Find and download the data files from Kaggle: 

- Gabriel Preda, “Reddit WallStreetBets Posts.” Kaggle, 2021, https://www.kaggle.com/gpreda/reddit-wallstreetsbets-posts/ metadata
- Raphael Fontes, “Reddit - r/wallstreetbets”. Kaggle, 2021. https://www.kaggle.com/unanimad/reddit-rwallstreetbets

3. Add CSV's to new folder `data/` or adjust code accordingly.
4. Follow Jupiter notebooks in the following suggested order:
	
	- [Train_Model.ipynb](https://github.com/brianward1428/word-embeddings-for-reddit-extraction/blob/master/Train_Model.ipynb "Train_Model.ipynb")
	- [Model_Testing.ipynb](https://github.com/brianward1428/word-embeddings-for-reddit-extraction/blob/master/Model_Testing.ipynb "Model_Testing.ipynb")
	- [ParamAnalysisV2.ipynb](https://github.com/brianward1428/word-embeddings-for-reddit-extraction/blob/master/ParamAnalysisV2.ipynb "ParamAnalysisV2.ipynb")
	- [Extract_Companies.ipynb](https://github.com/brianward1428/word-embeddings-for-reddit-extraction/blob/master/Extract_Companies.ipynb "Extract_Companies.ipynb")
	- [sentiment_analysis.ipynb](https://github.com/brianward1428/word-embeddings-for-reddit-extraction/blob/master/sentiment_analysis.ipynb "sentiment_analysis.ipynb")
