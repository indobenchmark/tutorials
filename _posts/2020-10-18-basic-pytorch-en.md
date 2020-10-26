---
layout: post
title: Pre-Trained Models for NLP Tasks Using PyTorch
sub_tile: "PyTorch and IndoNLU tutorial for Beginner"
categories:
  - PyTorch
  - Deep Learning
  - NLP
excerpt_separator: "<!--more-->"
---
<div class="message" style="background:#eee">
<img width=50px src="/tutorials/assets/img/lamp.png" style="float:left;margin-right:20px;margin-top:15px"/>
<div style="color:#555">
In this tutorial, you will learn how to implement deep learning models for natural language processing with step-by-step examples that guide you to be a good machine learning engineer or scientist.
</div>
</div>

<b>Welcome</b>  to the first tutorial on Natural Language Processing (NLP) in the world of deep neural networks. Our goal is to enable everyone to grasp the knowledge in applying deep learning to NLP, no matter if you are a scientist, an engineer, a student, or anything else. We will guide you to go step-by-step with easy-to-follow examples using real data.

<!--more-->

<div style="background:#fafafa;padding:10px;padding-top:0.5px;margin-top:0px;padding-bottom:1px;margin-bottom:12px;padding-left:17px;border-color:#dedede;border-style: solid;border-width: 2px;">
<h3 style="padding:top:1px;margin-top:13px">Outline</h3>
<ul>
<li><a href="#background">A Brief Background on NLP</a>
    <ul>
        <li href="#introduction">From Rules to Deep Learning Models</li>
        <li href="#tasks">Category of NLP Tasks</li>
    </ul>
</li>
<li><a href="#nlp-pytorch">Building a Model Using PyTorch</a>
  <ul>
    <li><a href="#pytorch">PyTorch Framework</a></li>
    <li><a href="#data-prep">Data Preparation</a></li>
    <li><a href="#model">Modeling</a></li>
    <li><a href="#training-phase">Training Phase</a></li>
    <li><a href="#evaluation-phase">Evaluation Phase</a></li>
  </ul>
</li>
<li><a href="#resources">Additional Resources</a></li>
</ul>
</div>

Let’s start with a simple example of predicting emotion. Consider the following two sentences:

<div class="message" style="padding-bottom:1px;background:#eee">
<img width=50px src="/tutorials/assets/img/example.png" style="float:left;margin-right:40px;margin-top:8px"/>
<div style="color:#555">
<ul>
<li>We went to Bali for a holiday. It was fantastic!</li>
<li>My neighbour won the jackpot! and it's not me.</li>
</ul>
</div>
</div>

Can you tell which one expresses `sadness`? Yes, the `second` one. This task is trivial for us, but how can we teach a machine to predict like humans?

<div id="background"></div>
### A Brief Background on NLP

<div id="introduction"></div>
#### From Rules to Deep Learning Models

For long decades, practitioners in NLP focus on building hand-crafted rules and grammars for each language that are very tedious and laborious until statistical models are applied to NLP. Basically, those models are used to learn a function (or in layman terms, we call it mapping) between input and targets. Recently, deep learning models show significant progress in NLP, especially when open source deep learning frameworks, such as PyTorch, are available for academia and industry.

A simple naive solution for an NLP application is a keyword matching using rules. For example, in emotion classification tasks, we can collect words that represent happiness, and for sentences with those words, we can classify them as happy. But, is it the best we can do? Instead of checking word by word, we can train a model that accepts a sentence as input and predicts a label according to the semantic meaning of the input.

To show the difference between those methods, we will show you back the previous example!

<div class="message" style="padding-bottom:1px;background:#eee">
<img width=50px src="/tutorials/assets/img/example.png" style="float:left;margin-right:40px;margin-top:8px"/>
<div style="color:#555">
<ul>
<li>We went to Bali for a holiday. It was fantastic!</li>
<li>My neighbour won the jackpot! and it's not me.</li>
</ul>
</div>
</div>

By checking the lexical terms, we can easily fool our model to classify the `second` sentence as `happy` because it has `won the jackpot` phrase. If the model is able to understand the second sentence completely, then it is easy to notice the change of meaning after the second clause that makes that person feel sad because they didn’t win the jackpot.

Before we go to deep learning modeling with PyTorch, we will first explain briefly about the task categories in NLP.

<div id="tasks"></div>
#### Category of NLP Tasks
There are different tasks in Natural Language Processing. We can outline main categories of the NLP tasks as follow
- Text Classification <br/>
    We aim to classify a text or document with a label class. Our example above is one of the examples. We predict an emotion label corresponding to the text.
- Sequence Labeling <br/>
    This is a task to predict a label for every token in the input. The most common task is Named Entity Recognition, the task to predict named entities in a given text input.
- Language Generation <br/>
    The task is to generate a sequence given a sequence as input. One of the examples is question answering task. Our model accepts a question and a context as input and generates an answer accordingly.
    
In this tutorial, we will show you an example of applying deep learning techniques on text classification. If you are new to deep learning, this will be a quickstart for you to start learning deep learning models using PyTorch.

### Building a Model Using PyTorch

We’ll start simple. Let’s use the available pretrained model, and then fine-tune (train) the model again, to accommodate our example above. 
In this tutorial, we will use example in Indonesian language and we will show examples of using PyTorch for training a model based on the [IndoNLU](https://github.com/indobenchmark/indonlu) project.

<div id="pytorch"></div>
#### PyTorch Framework
PyTorch is the best open source framework using Python and CUDA for deep learning based on the Torch library commonly used in research and production in natural language processing, computer vision, and speech processing. PyTorch is one of the most common deep learning frameworks used by researchers and industries. It is very intuitive for Numpy users as its core data structure, `torch.Tensor`, is pretty similar to numpy array. What difference is that `torch.Tensor` has the capability to be calculated in both CPU and CUDA, while in numpy it is not possible. It makes PyTorch a better tool for training deep learning models compared to Numpy. Another great thing is that  PyTorch supports dynamic computation graphs and the network can be debugged and modified on the fly, unlike the static computation graph in Tensorflow. It makes PyTorch much more convenient to use for debugging because we can easily check the tensor during the execution of the code.

PyTorch can be installed and used in most operating systems via Anaconda or pip.

If you use `Anaconda`, PyTorch can be installed by executing the following command
```bash
conda install pytorch torchvision -c pytorch
```

If you use `pip`, PyTorch can be installed by executing the following command
```bash
pip install torch torchvision
```

To use the PyTorch library, simply import the PyTorch library
```python
import torch
```

<div id="dataset"></div>
#### Sentiment Analysis Dataset
To add the benefit to your fun, let’s start with sentiment analysis, one of the most popular use cases yet easy to implement. Sentiment analysis is a natural language processing task to understand a sentiment within a body of text. In business, it is beneficial to automatically analyze customer review that is written i.e. in Twitter, Zomato, TripAdvisor, Facebook, Instagram, Qraved, and to understand the polarity of their review, whether it is a positive, negative, or a neutral review. As this kind of review dataset is included in one of the downstream tasks provided in the IndoNLU called SmSA ((https://github.com/indobenchmark/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa) along with all the needed resources, let’s begin our interesting project.

<div id="data-prep"></div>
#### Data Preparation
Data preparation is one of the fundamental parts in modeling, it is even commonly said to take 60% of the time from the whole modeling pipeline. Fortunately, the tons of utilities provided by PyTorch and IndoNLU can simplify this process.
 
PyTorch provides a standardized way to prepare data for the model. It provides advanced features for data processing and to be able to utilize those features, we need to utilize 2 classes from `torch.utils.data` package, which are `Dataset` and `DataLoader`. `Dataset` is an abstract class that we need to extend in PyTorch, we will pass the dataset object into `DataLoader` class for further processing of the batch data. `DataLoader` is the heart of PyTorch data loading utility. It provides many functionalities for preparing batch data including different sampling methods, data parallelization, and even for distributed processing. To show how to implement `Dataset` and `DataLoader` in PyTorch, we are going to dig deeper into `DocumentSentimentDataset` and `DocumentSentimentDataLoader` classes from IndoNLU that can be found in https://github.com/indobenchmark/indonlu/blob/master/utils/data_utils.py.

Before we begin with implementation, we need to know the format of our sentiment dataset. Our data is stored in `tsv` format and has two columns `text` and `sentiment`. Here are some examples of the dataset

<img src="/tutorials/assets/img/sample.png"/>

Now, let's start to prepare the pipeline. First, let's import the required components
```python
from torch.utils.data import Dataset, DataLoader
```
 
Next, we will implement the `DocumentSentimentDataset` class for loading our dataset. To make a fully functional `DocumentSentimentDataset` class, we need to at least define 3 different functions: `__init__(self, ...)`, `__getitem__(self, index)`, and `__len__(self)`. 

First, let’s define class and the` __init__(self, ...)` function
```python
class DocumentSentimentDataset(Dataset):
	# Static constant variable (We need to have this part to comply with IndoNLU standard)
	LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2} # Label string to index
	INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'} # Index to label string
	NUM_LABELS = 3 # Number of label
   
	def load_dataset(self, path):
    	df = pd.read_csv(path, sep=’\t’, header=None) # Read tsv file with pandas
    	df.columns = ['text','sentiment'] # Rename the columns
    	df['sentiment'] = df['sentiment'].apply(lambda lab: self.LABEL2INDEX[lab]) # Convert string label into index
    	return df
   
	def __init__(self, dataset_path, tokenizer, *args, **kwargs):
    	self.data = self.load_dataset(dataset_path) # Load the tsv file

        # Assign the tokenizer for tokenization
        # here we use subword tokenizer from HuggingFace
    	self.tokenizer = tokenizer 
```

Now, we already have the data and the tokenizer defined in the `__init__(self, ...)` function. Next, let’s use it to `__getitem__(self, index)` and `__len__(self)` functions.
```python
	def __getitem__(self, index):
    	data = self.data.loc[index,:] # Taking data from a specific row from Pandas
    	text, sentiment = data['text'], data['sentiment'] # Take text and sentiment from the row
    	subwords = self.tokenizer.encode(text) # Tokenize the text with tokenizer
	
	# Return numpy array of subwords and label
    	return np.array(subwords), np.array(sentiment), data['text']
   
	def __len__(self):
    	return len(self.data)  # Return the length of the dataset
```

So, that’s it for the `DocumentSentimentDataset`. The full class definition is as follow:
```python
class DocumentSentimentDataset(Dataset):
	# Static constant variable (We need to have this part to comply with IndoNLU standard)
	LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2} # Label string to index
	INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'} # Index to label string
	NUM_LABELS = 3 # Number of label
   
	def load_dataset(self, path):
    	df = pd.read_csv(path, sep=’\t’, header=None) # Read tsv file with pandas
    	df.columns = ['text','sentiment'] # Rename the columns
    	df['sentiment'] = df['sentiment'].apply(lambda lab: self.LABEL2INDEX[lab]) # Convert string label into index
    	return df
   
	def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
    	self.data = self.load_dataset(dataset_path) # Load the tsv file

        # Assign the tokenizer for tokenization
        # here we use subword tokenizer to convert text into subword
    	self.tokenizer = tokenizer 

	def __getitem__(self, index):
    	data = self.data.loc[index,:] # Taking data from a specific row from Pandas
    	text, sentiment = data['text'], data['sentiment'] # Take text and sentiment from the row
    	subwords = self.tokenizer.encode(text) # Tokenize the text with tokenizer

	# Return numpy array of subwords and label
    	return np.array(subwords), np.array(sentiment)
   
	def __len__(self):
    	return len(self.data)  # Specify the length of the dataset
```
 
Notice that the dataset class returns `subwords` that can have different length for each index. In order to be fed to the model in batch, we need to standardize the length of the sequence by truncating the length and adding padding tokens. In this case we are going to implement the `DocumentSentimentDataLoader` class extending the PyTorch `DataLoader`.

In order to have the specified functionality, we need to override the `collate_fn(self, batch)` function from the `DataLoader` class. `collate_fn()` is a function that will be called after the dataloader collects a batch of data from the dataset. The argument `batch` consists of a list of data returned from the `Dataset.__getitem__()`. Our `collate_fn(self, batch)` function will receive list of `subword` and `sentiment` and spit out tuples of `padded_subword`, `mask`, and `sentiment`. `mask` is a variable that we use to prevent the model from considering the padding token as part of the input. To simplify, the visualization below shows the process of our `collate_fn(self, batch)` transform input subword into the padded subword and mask.

In the above visualization, 0 value means padding token. `mask` only consists of two values 0 and 1, where 0 means this token should be ignored by model and 1 means this token should be considered by model. OK, let’s now define our `DocumentSentimentDataLoader`
```python
class DocumentSentimentDataLoader(DataLoader):
	def __init__(self, max_seq_len=512, *args, **kwargs):
    	super(DocumentSentimentDataLoader, self).__init__(*args, **kwargs)
self.max_seq_len = max_seq_len # Assign max limit of the sequence length
    	self.collate_fn = self._collate_fn # Assign the collate_fn function with our function
       
	def _collate_fn(self, batch):
    	batch_size = len(batch) # Take the batch size
    	max_seq_len = max(map(lambda x: len(x[0]), batch)) # Find maximum sequence length from the batch 
    	max_seq_len = min(self.max_seq_len, max_seq_len) # Compare with our defined limit
       
	# Create buffer for subword, mask, and sentiment labels, initialize all with 0
    	subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    	mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
    	sentiment_batch = np.zeros((batch_size, 1), dtype=np.int64)
       
	# Fill all of the buffer
    	for i, (subwords, sentiment, raw_seq) in enumerate(batch):
        	subwords = subwords[:max_seq_len]
        	subword_batch[i,:len(subwords)] = subwords
        	mask_batch[i,:len(subwords)] = 1
        	sentiment_batch[i,0] = sentiment
           
	# Return the subword, mask, and sentiment data
    	return subword_batch, mask_batch, sentiment_batch
```

Hooray!! We have implemented our `DocumentSentimentDataLoader`. Now let’s try to integrate this `DocumentSentimentDataset` and `DocumentSentimentDataLoader`. We can initialize our `DocumentSentimentDataset` and `DocumentSentimentDataLoader` in the following way:

```python
dataset = DocumentSentimentDataset(‘./sentiment_analysis.csv’, tokenizer)
data_loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32 num_workers=16, shuffle=True)  
```

and then we can iterate our data loader by iterating it as follow
```python
for (subword, mask, label) in data_loader:
    …
```

Pretty simple right? As you can see, in the `DocumentSentimentDataLoader` parameters, there are some additional parameters other than `max_seq_len`. The `dataset` parameter is a required parameter for `DataLoader` class, which is the data source used to fetch the data from. `batch_size` defines the number of data we will take per batch, `num_workers` defines the number of workers we want to use to fetch the data in parallel, and `shuffle` defines whether we want to take shuffle batch data or just fetch it in sequential order.

Ok, for the next section we are going to use our `DocumentSentimentDataset` and `DocumentSentimentDataLoader` for modeling purposes.

#### Model

Let's import the available pretrained model, from the [IndoNLU project](https://github.com/indobenchmark/indonlu), using Hugging-Face library. Thanks IndoNLU, thanks Hugging-Face! (suruh install requirements dulu ga? Iya) (suruh siapin jupyter notebook ato kita siapin jupyter notebook ga? Ga usah lah)

```python
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

args = {}
args["model_checkpoint"] = "indobenchmark/indobert-base-p1"

tokenizer = BertTokenizer.from_pretrained(args['model_checkpoint'])
config = BertConfig.from_pretrained(args['model_checkpoint'])
model = BertForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
```

Done! With this, we have now the indobert-base-p1 model ready to be trained (fine-tuned).

#### Training Step

Here we are going to train (fine-tune) the indobert-base-p1 model for XXXX task.

```python
def train(model, train_loader, valid_loader, optimizer, forward_fn, metrics_fn, valid_criterion, i2w, n_epochs, evaluate_every=1, early_stop=3, step_size=1, gamma=0.5, model_dir="", exp_id=None):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_metric = -100
    count_stop = 0

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        list_hyp, list_label = [], []
        
        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            loss, batch_hyp, batch_label = forward_fn(model, batch_data[:-1], i2w=i2w, device=args['device'])

            optimizer.zero_grad()
            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_norm'])
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_norm'])
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label
            
            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(args, optimizer)))
                        
        metrics = metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(args, optimizer)))
        
        # Decay Learning Rate
        scheduler.step()

        # evaluate
        if ((epoch+1) % evaluate_every) == 0:
            val_loss, val_metrics = evaluate(model, valid_loader, forward_fn, metrics_fn, i2w, is_test=False)

            # Early stopping
            val_metric = val_metrics[valid_criterion]
            if best_val_metric < val_metric:
                best_val_metric = val_metric
                # save model
                if exp_id is not None:
                    torch.save(model.state_dict(), model_dir + "/best_model_" + str(exp_id) + ".th")
                else:
                    torch.save(model.state_dict(), model_dir + "/best_model.th")
                count_stop = 0
            else:
                count_stop += 1
                print("count stop:", count_stop)
                if count_stop == early_stop:
                    break
```

#### Evaluation Step
```python

```


Before we dive into deep learning for NLP, we need to know what deep learning is. So, in short, deep learning is a term to cover multi-layers neural-network-based machine learning algorithms where the model is updated iteratively by applying gradient descent. Each layer on a deep neural network consists of two kind functions that we need to implement before we can use a deep learning model: forward function and backward function. If we have these two functions for all kinds of layers we can then build a deep neural network model, BUT the model will not be able to learn until we define one more function, the loss function. So far, we have identified the three fundamental functions that need to be implemented to build a deep learning model. So, can we build a deep neural network model now? Not yet, there are two problems here. First, there are so many variations for each of these three functions, it will be very time consuming for us to implement it by ourselves, and second, unfortunately... there is a great deal of MATHEMATICAL understanding behind each of them >o< >o< >o< !!!! 

### How to build the model? PyTorch is the answer!

I am not good at math, how can I use deep learning then? :’( :’( :’(

Don’t worry people, as mentioned earlier, our goal is to enable everyone to be able to use deep learning regardless of your background and fortunately we have PyTorch, a powerful deep learning framework that turns the complexity of deep learning into a game of LEGO. PyTorch has already implemented 99.5% of all the possible variations of those three functions mentioned earlier. With PyTorch, we can stack one layer over another without touching any single detail of MATHEMATICAL formula required for deep learning. 

YEAY!!! 

Ok, now let’s go building a deep neural network with PyTorch. First, we need to know what kind of layers are provided in PyTorch. 


Pytorch < HF < IndoNLU

PYTORCH ITU TUMPUKAN IMBA

AUTOGRAD

MODELNYA NUMPANG LEWAT
 

PASARIN FEATURES2 NYA PYTORCH: PROFILING, AMP, 

Layers di Pytorch


Transformers

### Keep It Simple

In keeping with the original Hyde theme, Hydeout aims to keep the overall
design lightweight and plugin-free. JavaScript is currently limited only
to Disqus and Google Analytics (and is only loaded if you provide configuration
variables).

Hydeout makes heavy use of Flexbox in its CSS. If Flexbox is not available,
the CSS degrades into a single column layout.

### Customization

Hydeout replaces Hyde's class-based theming with the use
of the following SASS variables:

```scss
$sidebar-bg-color: #202020 !default;
$sidebar-fg-color: white !default;
$sidebar-sticky: true !default;
$layout-reverse: false !default;
$link-color: #268bd2 !default;
```

To override these variables, create your own `assets/css/main.scss` file.
Define your own variables, then import in Hydeout's SCSS, like so:

```
---
# Jekyll needs front matter for SCSS files
---

$sidebar-bg-color: #ac4142;
$link-color: #ac4142;
$sidebar-sticky: false;
@import "hydeout";
```

See the [_variables](https://github.com/fongandrew/hydeout/blob/master/_sass/hydeout/_variables.scss) file for other variables
you can override.

You can also insert custom head tags (e.g. to load your own stylesheets) by
defining your own `_includes/custom-head.html` or insert tags at the end
of the body (e.g. for custom JS) by defining your own
`_includes/custom-foot.html`.

### New Features

* Hydeout also adds a new tags page (accessible in the sidebar) and a new
  "category" layout for dedicated category pages.

* Category pages are automatically added to the sidebar. All other pages
  must have `sidebar_link: true` in their front matter to show up in
  the sidebar.

* A simple redirect-to-Google search is available. If you want to use
  Google Custom Search or Algolia or something with more involved,
  override the `search.html`.

* Disqus integration is ready out of the box. Just add the following to
  your config file:

  ```yaml
  disqus:
    shortname: my-disqus-shortname
  ```

  If you don't want Disqus or want to use something else, override
  `comments.html`.

* For Google Analytics support, define a `google_analytics` variable with
  your property ID in your config file.

There's also a bunch of minor tweaks and adjustments throughout the
theme. Hope this works for you!