---
layout: post
title: Pre-Trained Models for NLP Tasks Using PyTorch
sub_tile: "Beginner"
categories:
  - PyTorch
  - Deep Learning
  - NLP
excerpt_separator: "<!--more-->"
---
<div class="message" style="background:#eee">
<img width=50px src="/tutorials/assets/img/lamp.png" style="float:left;margin-right:20px;margin-top:15px"/>
<div style="color:#555">
In this tutorial, you will learn on how to implement deep learning models for natural language processing with step-by-step examples that guide you to be a good machine learning engineer or scientist. 
</div>
</div>

<b>Welcome</b> to the first tutorial on Natural Language Processing (NLP) in the world of deep neural networks. Our goal is to enable everyone to grasp the knowledge in applying deep learning to NLP, no matter if you are a scientist, an engineer, a student, or anything else. We will guide you to go step-by-step with easy-to-follow examples using real data.

<!--more-->

<div style="background:#fafafa;padding:10px;padding-top:0.5px;margin-top:0px;padding-bottom:1px;margin-bottom:12px;padding-left:17px;border-color:#dedede;border-style: solid;border-width: 2px;">
<h3 style="padding:top:1px;margin-top:13px">Outline</h3>
<ul>
<li><a href="#sec1">A Brief Background on NLP</a></li>
<li><a href="#sec2">Building a Model Using PyTorch</a>
  <ul>
    <li><a href="#sec2-1">Writing Model</a></li>
    <li><a href="#basic-concepts">Basic Concepts</a></li>
    <li>Basic Concepts</li>
  </ul>
</li>
</ul>
</div>

Let's start with a simple example of predicting emotion. Imagine that we have two sentences:

<div class="message" style="padding-bottom:1px;background:#eee">
<img width=50px src="/tutorials/assets/img/example.png" style="float:left;margin-right:40px;margin-top:8px"/>
<div style="color:#555">
<ul>
<li>We went to Bali for holiday. It was fantastic!</li>
<li>My neighbour won the jackpot! and it's not me.</li>
</ul>
</div>
</div>

Can you tell which one expresses ```sadness```? Yes, the ```second``` one. This task is trivial for us, but how can we teach a machine to predict as like humans?

<div id="sec1"></div>
### A Brief Background on NLP
#### From Rules to Deep Learning Models

For long decades, practitioners in NLP focus on building hand-crafted rules and grammars for each language that are very tedious and labourous until statistical models are applied to NLP. Basically, those models are used to learn a function (or in layman terms, we call it ```mapping```) between input and targets. Then, just recently, deep learning models show a significant progress in NLP, especially when open source deep learning frameworks, such as [PyTorch](https://pytorch.org/) is available for academia and industry.

A simple naive solution for an NLP application is <b>a keyword matching using rules</b>. For example, in emotion classification task, we can collect words that represent happiness, and for sentences with those words, we can classify them as ```happy```. But, is it the best we can do? Instead of checking word by word, we can train a model that accepts a sentence as input and predicts a label according to the semantic meaning of the input.

To show the difference between those methods, we will show you back the previous example!

<div class="message" style="padding-bottom:1px;background:#eee">
<img width=50px src="/tutorials/assets/img/example.png" style="float:left;margin-right:40px;margin-top:8px"/>
<div style="color:#555">
<ul>
<li>We went to Bali for <b>holiday</b>. It was <b>fantastic!</b></li>
<li>My neighbour <b>won the jackpot!</b> and it's <b>not me.</b></li>
</ul>
</div>
</div>

By checking the lexical terms, we can easily fool our model to classify the ```second``` sentence as ```happy``` because it has the ```won the jackpot``` phrase. If the model is able to understand the second sentence completely, then it is easy to notice the change of meaning after the second clause that that person feel sad because they didn't win the jackpot.

Let's start learning how to build a deep learning model using PyTorch.

<div id="sec2"></div>
### Building a Model Using PyTorch

We show examples to use PyTorch for training a model based on the [IndoNLU project](https://github.com/indobenchmark/indonlu). 

<div id="sec2-1"></div>
#### Model
```python
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

args = {}
args["model_checkpoint"] = "indobenchmark/indobert-base-p1"

tokenizer = BertTokenizer.from_pretrained(args['model_checkpoint'])
config = BertConfig.from_pretrained(args['model_checkpoint'])
model = BertForSequenceClassification.from_pretrained(args['model_checkpoint'], config=config)
```

#### Training Step
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