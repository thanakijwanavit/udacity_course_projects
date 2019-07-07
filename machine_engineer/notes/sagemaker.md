# Sagemaker

## Set up a notebook

### First example, XG_boost model

#### Prepare the data

example of a read function
```python
import os
import glob

def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}

    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)

            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)

    return data, labels

```

```python
#shuffle the data
def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""

    #Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']

    #Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)

    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test
```

#### Process the data
> in this example, we are using a description sentence (human language) so we need to process them into tokens
```
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
from bs4 import BeautifulSoup

def review_to_words(review):
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words


import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]

        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])

    return words_train, words_test, labels_train, labels_test




```
### Classify with XTBoost
1. split train and val
``` python
train_X=pd.DataFrame(train_x[1000:])
val_X = pd.DataFrame(train_x[:1000])
```
same for x and y

* save to csv
``` python
pd.concat([train_y,train_x,axis=1).to_csv('path')
```
#### upload to s3

``` python
import sagemaker
session = sagemaker.Session()
## upload data
test_location=session.upload_data(path, key_prefix="folder_name")
```
#### Create XGBoost model
* create execution role
* get container uri
* create estimator
* set hyperparameters

``` python
## create execution role
role= sagemaker_get_execution_role()
## get container uri
container=sagemaker.amazon.amazon_estimator.get_image_uri(session.boto_region_name,'container_name (in this case is xgboost)')
## get container uri
container=sagemaker.amazon.amazon_estimator.get_image_uri(session.boto_region_name,'container_name (in this case is xgboost)')
## output location?
output_location='s3://{}/{}'.format(session.default_bucket(), prefix) #this is the folder name
## create estimator

xgb=sagemaker.estimator.Estimator(container,
	role,
	train_instance_count=1,
	train_instance_type='ml.m4.xlarge',
	sagemaker_session=session #created earlier
	)

## set hyperparameters
xgb.set_hyperparameters(max_depth = 5,
                              eta = .2,
                              gamma = 4,
                              min_child_weight = 6,
                              silent = 0,
                              objective = "multi:softmax",
                              num_class = 10,
                              num_round = 10)
```
* Finallyy get the inputs and fit the model
``` python
## get inputs
3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')
## train model
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```
#### Test the model with batch transform
* create a transformer
* transform
* wait
* copy result to directory
* readcsv and count the right result

``` python
## create transformer
transformer=xgb.transformer(instance_count=1,
	instance_type='ml.m4.xlarge'
	)
## transform
transformer.transform(test_location,content_type='text/csv, split_type='Line)
## wait
transformer.wait()

## copy
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
## read_data
predictions = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]
## compare data
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)

