---
layout: post
title: Tutorial penggunaan Pre-Trained Model untuk NLP dengan menggunakan PyTorch
sub_tile: "Tutorial penggunaan PyTorch danIndoNLU untk Pemula"
categories:
  - PyTorch
  - Deep Learning
  - NLP
excerpt_separator: "<!--more-->"
---
<div class="message" style="background:#eee">
<img width=50px src="/tutorials/assets/img/lamp.png" style="float:left;margin-right:20px;margin-top:15px"/>
<div style="color:#555">
Dalam tutorial ini kamu akan belajar bagaimana implementasi model deep learning untuk natural language processing (NLP) dengan contoh bertahap yang akan mengantarkanmu untuk menjadi machine learning engineer / data scientist / AI researcher yang baik.
</div>
</div>

<b>Selamat datang</b> di tutorial pertama Natural Language Processing (NLP) dengan menggunakan deep learning. Tujuan dari tutorial ini adalah untuk memberikan bahan pembelajaran yang mudah dipahami agar siapa saja dapat mengaplikasikan deep learning dalam kasus NLP. Tutorial ini dapat diikuti siapa saja yang sudah memiliki pengetahuan terkait bahasa pemrograman Python. Disini kita akan mempelajari NLP dengan menggunakan contoh data riil yang mudah diikuti. Hal-hal yang akan kita bahas di dalam tutorial ini adalah sebagai berikut:

<!--more-->

<div style="background:#fafafa;padding:10px;padding-top:0.5px;margin-top:0px;padding-bottom:1px;margin-bottom:12px;padding-left:17px;border-color:#dedede;border-style: solid;border-width: 2px;">
<h3 style="padding:top:1px;margin-top:13px">Outline</h3>
<ul>
<li><a href="#background">Deep Learning di NLP</a>
    <ul>
        <li href="#introduction">Transisi dari sistem berbasis aturan ke deep learning</li>
        <li href="#tasks">Kategori task dibidang NLP</li>
    </ul>
</li>
<li><a href="#nlp-pytorch">Membangun model NLP dengan PyTorch</a>
  <ul>
    <li><a href="#pytorch">Apa itu PyTorch?</a></li>
    <li><a href="#data-prep">Data Preparation</a></li>
    <li><a href="#model">Modeling</a></li>
    <li><a href="#training-phase">Training Phase</a></li>
    <li><a href="#evaluation-phase">Evaluation Phase</a></li>
  </ul>
</li>
<li><a href="#resources">Additional Resources</a></li>
</ul>
</div>

Pertama-tama mari mulai dengan contoh sederhana dalam pendeteksian emosi. Mari lihat dua contoh dibawah ini:

<div class="message" style="padding-bottom:1px;background:#eee">
<img width=50px src="/tutorials/assets/img/example.png" style="float:left;margin-right:40px;margin-top:8px"/>
<div style="color:#555">
<ul>
<li>Kita pergi ke Bali untuk berlibur. Sungguh menakjubkan!</li>
<li>Tetanggaku memenangkan undian berhadiah! Tapi sayangnya aku belum beruntung.</li>
</ul>
</div>
</div>

Manakah dari kedua contoh tersebut yang lebih mengutarakan `kekecewaan`? Ya, tentu kalimat yang `kedua`. Contoh ini nampak mudah bagi kita sebagai manusia, namun bagaimanakah kita dapat mengajarkan cara berpikir seperti ini pada komputer?

<div id="background"></div>
### Deep Learning di NLP

<div id="introduction"></div>
#### Transisi dari sistem berbasis aturan ke deep learning

Selama beberapa dekade ke belakang, praktisi di bidang NLP fokus membangun aturan-aturan untuk menangkap gramatika dari setiap bahasa. Langkah ini sangatlah sulit dan memakan banyak waktu sampai suatu momen dimana model-model statistik mulai digunakan di bidang NLP. Model-model tersebut dibangun untuk mempelajari suatu fungsi yang memetakan suatu masukan ke suatu nilai keluaran. Dalam beberapa tahun terakhir, model-model deep learning menunjukan progres yang signifikan di bidang NLP, terutama sejak keberadaan framework-framework deep learning yang open source seperti PyTorch.

Solusi sederhana untuk aplikasi NLP adalah dengan aturan keyword matching. Dalam contoh kasus pendeteksian emosi, kita dapat mengumpulkan kata-kata yang merepresentasikan senang, sedih, dan lainnya. Untuk setiap kalimat yang memiliki kata-kata paling banyak terkait suatu emosi, kita dapat mengklasifikasikannya ke dalam kategori emosi tersebut. Apakah itu cara yang terbaik yang dapat kita lakukan? Dibandingkan melakukan pengecekan kemunculan kata, dengan menggunakan deep learning kita dapat melatih model yang dapat menerima masukan kalimat dan memprediksi emosi bergantung arti semantik dari kalimat tersebut.

Untuk memperlihatkan perbedaan kedua metode tersebut, mari lihat kembali contoh sebelumnya!

<div class="message" style="padding-bottom:1px;background:#eee">
<img width=50px src="/tutorials/assets/img/example.png" style="float:left;margin-right:40px;margin-top:8px"/>
<div style="color:#555">
<ul>
<li>Kita pergi ke Bali untuk berlibur. Sungguh menakjubkan!</li>
<li>Tetanggaku memenangkan undian berhadiah! Tapi sayangnya aku belum beruntung.</li>
</ul>
</div>
</div>

Dengan melakukan pengecekan kata, kita dapat dengan mudah mengelabui model untuk mengklasifikasi emosi pada kalimat `kedua` sebagai `senang` karena terdapat frasa `memenangkan undian`. Jika model tersebut dapat memahami arti dari kalimat kedua, maka model dapat dengan mudah mengetahui perubahan makna setelah klausa pertama dimana orang yang mengatakan kalimat tersebut sedih karena ia tidak memenangkan undian.

Sebelum masuk ke pemodelan deep learning model dengan PyTorch, kita akan membahas sedikit lebih jauh mengenai jenis-jenis task di NLP.

<div id="tasks"></div>
#### Kategori task dibidang NLP
Dibidang NLP, terdapat banyak task yang dapat dicapai. Kita dapat membagi task-task tersebut kedalam beberapa kategori sebagai berikut
- Text Classification <br/>
    Klasifikasi dokumen dengan sebuah label. Contoh pendeteksian emosi sebelumnya adalah salah satu contoh dari kategori ini. Kita memprediksi label emosi bergantung dari kalimat yang diberikan.
- Sequence Labeling <br/>
    Kategori task ini melakukan prediksi label untuk setiap token dalam teks masukan. Task yang paling umum dalam kategori ini adalah Named Entity Recognition yang bertujuan untuk mendeteksi token pada kalimat masukan mana yang merupakan nama dari suatu entitas named entities.
- Language Generation <br/>
    Dalam kategori task ini, model akan menghasilkan serangkaian kalimat baru berdasarkan kalimat masukan yang diberikan. Salah satu contoh dari kategori ini adalah mesin penjawab pertanyaan, dimana model menerima pertanyaan masukan dan konteks pertanyaan sebagai masukan dan menghasilkan kalimat jawaban dari pertanyaan tersebut.

Dalam tutorial ini, kita akan melihat contoh aplikasi deep learning pada kategori text classification. Jika kamu baru pertama kali belajar deep learning, ini akan menjadi jalan pintas untuk mempelajari model deep learning pada kasus NLP dengan menggunakan PyTorch.

### Membangun model NLP dengan PyTorch

Kita akan mulai dari contoh sederhana. Disini kita akan menggunakan model yang telah dilatih sebelumnya (pre-trained model), dan kemudian melakukan penyesuaian (fine-tuning) pada pre-trained model tersebut. Pada contoh ini kita akan menggunakan contoh langsung berbahasa Indonesia dengan menggunakan micro-framework [IndoNLU](https://github.com/indobenchmark/indonlu) yang dibangun diatas framework deep learning, PyTorch.

<div id="pytorch"></div>
#### Apa itu Pytorch?
PyTorch adalah salah satu open-source framework deep learning terbaik yang dibangun diatas Python dan CUDA berdasarkan pada library Torch. PyTorch umum digunakan untuk riset dan juga aplikasi-aplikasi di bidang NLP, computer-vision, pemrosesan audio, dan lain-lain. PyTorch adalah salah satu framework deep learning yang paling sering digunakan oleh di riset-riset akademik dan juga di industri. PyTorch akan sangat mudah dipelajari bagi para pengguna Numpy dikarenakan struktur data dasar dari PyTorch, `torch.Tensor`, sangat mirip dengan numpy array pada library Numpy. Hal utama yang membedakan `torch.Tensor` dengan numpy array adalah kemampuan dari `torch.Tensor` untuk melakukan operasi pada CPU dan CUDA. Hal ini membuat PyTorch menjadi sebuah framework yang jauh lebih baik untuk deep learning dibandingkan dengan Numpy. Selain itu, PyTorch juga memiliki kapabilitas untuk membangun graf komputasi yang dinamis sehingga kita dapat melakukan debugging model yang dibangun dengan mudah dan struktur model dapat diubah kapan saja. Hal ini menjadi kelebihan besar dari PyTorch dibandingkan dengan framework deep learning seperti TensorFlow yang memiliki graf komputasi yang statis.

PyTorch dapat diinstall dan digunakan hampir diseluruh sistem operasi dengan menggunakan `Anaconda` atau `pip`. 

Bila menggunakan `Anaconda`, PyTorch dapat diinstall dengan cara mengeksekusi perintah berikut
```bash
conda install pytorch torchvision -c pytorch
```

Bila menggunakan `pip`, PyTorch dapat diinstall dengan cara mengeksekusi perintah berikut
```bash
pip install torch torchvision
```

Untuk menggunakan PyTorch pada kode Python kita dapat lakukan dengan melakukan import modul PyTorch
```python
import torch
```

<div id="dataset"></div>
#### Dataset Sentiment Analysis
Sebagai contoh kasus, dalam tutorial ini kita akan menggunakan dataset sentiment analysis. Sentiment analysis merupakan salah satu task yang paling populer di NLP. Dalam sentiment analysis kita akan membangun model yang dapat mendeteksi sentimen dalam sebuah kalimat. Umumnya, sentimen memiliki tiga nilai polaritas yaitu positif, negatif, dan netral. Di berbagai industri, sangatlah menguntungkan jika dapat menganalisa review dari pelanggan secara otomatis. Review-review tersebut dapat berasal dari berbagai macam platform sosial seperti Twitter, Zomato, TripAdvisor, Facebook, Instagram, dan lain-lain. Dataset sentiment analysis yang akan kita gunakan berasal dari salah satu task di IndoNLU yang bernama SmSA (https://github.com/indobenchmark/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa). Mari kita mulai proyek sentiment analysis ini.

<div id="data-prep"></div>
#### Data Preparation
Kita mulai proyek ini dengan persiapan data di PyTorch. Persiapan data adalah salah satu tahap yang paling penting dalam pemodelan. Secara umum, persiapan data memakan kurang lebih 60% dari total waktu untuk persiapan pemodelan. Untungnya, berbagai fungsi untuk melakukan data preprocessing, telah disediakan oleh PyTorch dan IndoNLU.
 
PyTorch menyediakan cara terstandarisasi untuk menyiapkan data untuk melakukan pemodelan. PyTorch menyediakan fitur-fitur canggih untuk memproses data dan untuk dapat menggunakan fitur-fitur tersebut, kita perlu menggunakan 2 kelas yang disediakan di PyTorch dalam modul `torch.utils.data` yaitu `Dataset` dan `DataLoader`. `Dataset` is sebuah abstract class yang perlu kita extend di PyTorch, kita akan mengoper objek dari kelas `Dataset` ke dalam objek dari kelas `DataLoader`. `DataLoader` adalah inti dari perangkat pemrosesan data di PyTorch. `DataLoader` menyediakan banyak fungsionalitas untuk mempersiapkan data termasuk berbagai metode sampling, komputasi paralel, dan pemrosesan terdistribusi. Untuk menunjukan bagaimana cara mengimplementasi `Dataset` dan `DataLoader` di PyTorch, kita akan melihat lebih dalam pada kelas `DocumentSentimentDataset` and `DocumentSentimentDataLoader` yang disediakan oleh IndoNLU dan dapat diakses di https://github.com/indobenchmark/indonlu/blob/master/utils/data_utils.py.
 
Sebelum masuk ke implementasi, kita perlu mengetahui format dari dataset sentiment analysis yang kita gunakan. Dataset SmSA yang akan kita gunakan tersimpan dalam format `tsv` dan terdiri dari 2 kolom, yaitu: `text` dan `sentiment`. Berikut adalah contoh dari data pada dataset SmSA

<img src="/tutorials/assets/img/sample.png"/>

Mari kita mulai mempersiapkan pipeline pemrosesan data. Pertama-tama, mari kita import modul-modul yang diperlukan
```python
from torch.utils.data import Dataset, DataLoader
```
 
Berikutnya, kita akan mengimplementasi kelas `DocumentSentimentDataset` untuk data loading. Untuk membuat kelas `DocumentSentimentDataset` yang fungsional, kita perlu mengimplementasikan 3 fungsi, yaitu `__init__(self, ...)`, `__getitem__(self, index)`, dan `__len__(self)`. 

Pertama-tama, mari kita implementasikan fungsi ` __init__(self, ...)` 
```python
class DocumentSentimentDataset(Dataset):
	# Static constant variable
	LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2} # Map dari label string ke index
	INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'} # Map dari Index ke label string
	NUM_LABELS = 3 # Jumlah label
   
	def load_dataset(self, path):
    	df = pd.read_csv(path, sep=’\t’, header=None) # Baca tsv file dengan pandas
    	df.columns = ['text','sentiment'] # Berikan nama pada kolom tabel
    	df['sentiment'] = df['sentiment'].apply(lambda lab: self.LABEL2INDEX[lab]) # Konversi string label ke index
    	return df
   
	def __init__(self, dataset_path, tokenizer, *args, **kwargs):
    	self.data = self.load_dataset(dataset_path) # Load tsv file

# Assign tokenizer, disini kita menggunakan tokenizer subword dari HuggingFace
    	self.tokenizer = tokenizer 
```

Sekarang kita sudah memiliki data dan tokenizer yang didefinisikan difungsi `__init__(self, ...)`. Selanjutnya kita akan mengimplementasikan fungsi `__getitem__(self, index)` dan `__len__(self)`.
```python
	def __getitem__(self, index):
    	data = self.data.loc[index,:] # Ambil data pada baris tertentu dari tabel
    	text, sentiment = data['text'], data['sentiment'] # Ambil nilai text dan sentiment
    	subwords = self.tokenizer.encode(text) # Tokenisasi text menjadi subword
	
	# Return numpy array dari subwords dan label
    	return np.array(subwords), np.array(sentiment), data['text']
   
	def __len__(self):
    	return len(self.data)  # Return panjang dari dataset
```

Ya, kita sudah mengimplementasikan seluruh fungsi pada `DocumentSentimentDataset`. Definisi lengkap dari `DocumentSentimentDataset` adalah seperti berikut:
```python
class DocumentSentimentDataset(Dataset):
	# Static constant variable
	LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2} # Map dari label string ke index
	INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'} # Map dari Index ke label string
	NUM_LABELS = 3 # Jumlah label
   
	def load_dataset(self, path):
    	df = pd.read_csv(path, sep=’\t’, header=None) # Baca tsv file dengan pandas
    	df.columns = ['text','sentiment'] # Berikan nama pada kolom tabel
    	df['sentiment'] = df['sentiment'].apply(lambda lab: self.LABEL2INDEX[lab]) # Konversi string label ke index
    	return df
   
	def __init__(self, dataset_path, tokenizer, *args, **kwargs):
    	self.data = self.load_dataset(dataset_path) # Load tsv file

        # Assign tokenizer, disini kita menggunakan tokenizer subword dari HuggingFace
    	self.tokenizer = tokenizer 

	def __getitem__(self, index):
    	data = self.data.loc[index,:] # Ambil data pada baris tertentu dari tabel
    	text, sentiment = data['text'], data['sentiment'] # Ambil nilai text dan sentiment
    	subwords = self.tokenizer.encode(text) # Tokenisasi text menjadi subword
	
	# Return numpy array dari subwords dan label
    	return np.array(subwords), np.array(sentiment), data['text']
   
	def __len__(self):
    	return len(self.data)  # Return panjang dari dataset
```
 
Perhatikan bahwa `subwords` yang dikembalikan oleh dataset dapat memiliki panjang yang berbeda-beda untuk setiap index. Untuk dapat diproses secara paralel oleh model, kita perlu menstandarisasi panjang dari `subwords` dengan memotong beberapa `subwords` atau menambahkan padding token. Untuk itu kita perlu mengimplementasikan `DocumentSentimentDataLoader`.

Untuk mendapatkan fungsi yang sesuai dengan apa yang kita butuhkan, kita dapat meng-override fungsi `collate_fn(self, batch)` dari kelas `DataLoader`. `collate_fn()` adalah fungsi yang akan dipanggil setelah dataloader mengumpulkan satu batch data dari dataset. Argumen `batch` pada fungsi `collate_fn` berisi list data yang dikembalikan dari `Dataset.__getitem__()`. Fungsi `collate_fn(self, batch)` kita memproses list `subword` dan `sentiment` dan mengeluarkan `padded_subword`, `mask`, dan `sentiment`. `mask` adalah suatu variabel yang digunakan untuk mencegah model memproses padding token sebagai bagian dari masukan. Untuk memudahkan, visualisasi berikut menunjukan proses transformasi subword masukan menjadi subword  yang telah di padding beserta `mask`nya.

<img src="/tutorials/assets/img/padding_mask.png"/>

Dari gambar diatas, angka 0 pada subword keluaran menandakan padding token, sedangkan nilai lainnya adalah nilai dari subword masukan. Variable `mask` terdiri dari dua buah nilai, 0 and 1, dimana 0 berarti token tidak perlu masuk dalah perhitungan model dan 1 berarti token yang perlu diperhitungkan. OK, sekarang mari kita implementasikan kelas  `DocumentSentimentDataLoader`.
```python
class DocumentSentimentDataLoader(DataLoader):
	def __init__(self, max_seq_len=512, *args, **kwargs):
    	super(DocumentSentimentDataLoader, self).__init__(*args, **kwargs)
self.max_seq_len = max_seq_len # Assign batas maksimum subword
    	self.collate_fn = self._collate_fn # Assign fungsi collate_fn dengan fungsi yang kita definisikan
       
	def _collate_fn(self, batch):
    	batch_size = len(batch) # Ambil batch size
    	max_seq_len = max(map(lambda x: len(x[0]), batch)) # Cari panjang subword maksimal dari batch 
    	max_seq_len = min(self.max_seq_len, max_seq_len) # Bandingkan dengan batas yang kita tentukan sebelumnya
       
	# Buat buffer untuk subword, mask, dan sentiment labels, inisialisasikan semuanya dengan 0
    	subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    	mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
    	sentiment_batch = np.zeros((batch_size, 1), dtype=np.int64)
       
	# Isi semua buffer
    	for i, (subwords, sentiment, raw_seq) in enumerate(batch):
        	subwords = subwords[:max_seq_len]
        	subword_batch[i,:len(subwords)] = subwords
        	mask_batch[i,:len(subwords)] = 1
        	sentiment_batch[i,0] = sentiment
           
	# Return subword, mask, dan sentiment data
    	return subword_batch, mask_batch, sentiment_batch
```

Horeee!! Kita sudah berhasil mengimplementasikan kelas `DocumentSentimentDataLoader`. Sekarang mari kita coba integrasikan kelas `DocumentSentimentDataset` and `DocumentSentimentDataLoader`. Kita dapat menginisialisasi kelas `DocumentSentimentDataset` and `DocumentSentimentDataLoader` dengan cara berikut ini:

```python
dataset = DocumentSentimentDataset(‘./sentiment_analysis.csv’, tokenizer)
data_loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32 num_workers=16, shuffle=True)  
```

Kemudian kita bisa mengambil data dari dataloader dengan cara berikut:
```python
for (subword, mask, label) in data_loader:
    …
```

Sangat mudah bukan? Seperti yang kita lihat, pada parameter instansiasi kelas `DocumentSentimentDataLoader`, ada beberapa parameter lain selain `max_seq_len`. Parameter `dataset` adalah parameter wajib untuk kelas `DataLoader` dan akan digunakan sebagai sumber data. `batch_size` menandakan jumlah data yang akan diambil pada setiap batch, `num_workers` menandakan jumlah proses yang akan kita gunakan untuk mengambil data secara paralel, and `shuffle` menandakan jika kita ingin mengambil data secara terurut atau acak.

Pada sesi selanjutnya, kita akan menggunakan `DocumentSentimentDataset` dan `DocumentSentimentDataLoader` dalam proses pemodelan

#### Model

For our modeling purpose, we are going to use a very popular model in NLP called BERT. BERT is a very popular pre-trained contextualized language model that stands for Bidirectional Encoder Representations from Transformers. Just like what it says in its name, BERT makes use of transformers, the attention mechanism that takes contextual relations between words in a text into account. Before BERT, the most popular techniques were recurrent based models, which read the text input sequentially. What makes BERT better is that it removes the first order markov assumption and provides a self-attention mechanism. 




There is an option to do modeling but not from scratch, that is to only tell the model to learn a little bit more from what it already knows. This kind of training is called fine-tuning. So here, we’re not doing the training from scratch, but rather, we will download the pretrained model from IndoNLU, specifically the [`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1) model.

Now, let’s import the available pretrained model from the IndoNLU project that is hosted in the Hugging-Face platform. Thanks IndoNLU and Hugging-Face! For our sentiment analysis task, we will perform fine-tuning using the `BertForSequenceClassification` model class from HuggingFace `transformers` package.

```python
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
config = BertConfig.from_pretrained("indobenchmark/indobert-base-p1")
Model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", config=config)
```
Done! We now have the `indobert-base-p1` tokenizer and model ready to be fine-tuned.
Training Phase
Here we are going to fine-tune the `indobert-base-p1` model with our sentiment analysis dataset.

Another essential feature that PyTorch provides is the autograd package. So, before we jump into training the model, we first briefly welcome an autograd to join the team. Basically, what autograd does is to provide automatic differentiation for all operations that happened on Tensors. This framework automatically generates a computational graph that will be used in the backpropagation. 

The finetuning training is done in these steps:
1. Define an optimizer `Adam` with a small learning rate, usually below `1e-3`
2. Enable the dropout regularization layer in the model by calling `model.train()`
3. Enable the autograd computation by calling `torch.set_grad_enabled(True)`
4. Iterate our data loader `train_loader` to get `batch_data` and pass it to the forward function `forward_sequence_classification` in the model.
5. Calculate the gradient by calling `loss.backward()` to compute all the gradients automatically. This function is inherited by the autograd package.
6. Call `optimizer.step()` to apply gradient update operation.’

We can see that the points above are coded in this below training script that we will use later for our finetuning.

```python
optimizer = optim.Adam(model.parameters(), lr=3e-6)
model = model.cuda()

n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    torch.set_grad_enabled(True)
 
    total_train_loss = 0
    list_hyp, list_label = [], []

    train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
    for i, batch_data in enumerate(train_pbar):
        # Forward model
        loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss = loss.item()
        total_train_loss = total_train_loss + tr_loss

        # Calculate metrics
        list_hyp += batch_hyp
        list_label += batch_label

        train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), get_lr(optimizer)))

    # Calculate train metric
    metrics = document_sentiment_metrics_fn(list_hyp, list_label)
    print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
        total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))
```

The training script above uses the `forward_sequence_classification` function, as the forward function, and following is the snippet of the important parts of  `forward_sequence_classification` function.

```python
def forward_sequence_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    …
    if device == "cuda":
    	subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
    	label_batch = label_batch.cuda()
    
    # Forward model
    outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids = token_type_batch, labels=label_batch)
        loss, logits = outputs[:2]
    …
    return loss, list_hyp, list_label
```

In both the training script and forward function above we leverage some of the pytorch capabilities, such as the easiness of switching the computation to CPU or GPU, the flexibilities of defining the loss function and computing the loss, and also the hassle-free gradient update by leveraging the autograd package to do the optimization and back propagation. Let’s add some more detail into the capabilities we are leveraging on.

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

##### CPU vs CUDA
```
model = model.cuda()
```

Training a million parameterized models like BERT on the CPU will take a vast amount of time. The linear algebra operations are done in parallel on the GPU and therefore you can achieve around 100x faster in training time. To cater for this need, PyTorch has prepared an easy way for us to move our tensor data typed variable to be computed either in CPU or GPU. This is done by accessing the method `.cuda()` or `.cpu()` in every tensor instantiated class and also the model, to move an object either to GPU memory from CPU memory, or vice versa.

##### Loss Function

```
    loss_fct = CrossEntropyLoss()
    total_loss = 0
    for i, (logit, num_label) in enumerate(zip(logits, self.num_labels)):
        label = labels[:,i]
        loss = loss_fct(logit.view(-1, num_label), label.view(-1))
        total_loss += loss
```

In PyTorch, we can build our own loss function or use loss function provided by the pytorch package. Building custom loss functions in Pytorch is not that hard actually, we just need to define a function that compares the output logits tensor with the label tensor and with that our loss function can have the same properties as the provided loss functions (automatically computed gradients, etc.). In our example here, we are using a provided loss function called `CrossEntropyLoss()`. Cross entropy loss is calculated by comparing how well the probability distribution output by Softmax matches the one-hot-encoded ground truth label of the data. We use this loss function in our sentiment analysis case because this loss fits perfectly to our needs as this is stating the likelihood of the model's capability to output perfect separability of the possible sentiment category in the label.

##### Optimizer and Back Propagation

```
optimizer = optim.Adam(model.parameters(), lr=5e-6)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Initializing the optimizer needs us to explicitly tell it what parameters (tensors) of the model it should be updating. To do a back propagation, we only need to call `.backward()` on the loss Variable as this will start a process of backpropagation at the end loss and goes through all of its parents all the way to model inputs and output the gradient needed for backpropagation. Behind the curtain, while you’re called `loss.backward()`, gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes). This is why, to do the backpropagation, optimizers don’t need to know anything about your loss. To update the gradient for all tensors in the model, we need to zero out all the previous grad from previous training by calling `optimizer.zero_grad()`, and then we need to call `optimizer.step()` as this will make the optimizer iterate over all parameters (tensors) it is supposed to update (requires_grad =True) and use their internally stored grad to update their values.

#### Evaluation Step

At each of the epoch training, we will evaluate the trained model performance. In the above finetuning training tutorial, we set the `set_grad_enabled` parameter as True, allowing all the gradients computed automatically when we call `loss.backward()`. In this evaluation phase, we don’t need that capability to be activated, and we don’t want to update any gradient parameters in any of the tensors. thus we set the `set_grad_enabled` parameter as False.

The fine tuning validation is done in these steps:
Enable the autograd computation by calling `torch.set_grad_enabled(True)`.
Iterate our data loader `valid_loader` to get `batch_data`.
Pass it to the same forward function `forward_sequence_classification` in the model to output the model prediction.
Evaluate the predictions using sklearn functions that are provided in the `document_sentiment_metrics_fn` script, to output the accuracy, F1 score, recall, and precision.

We can see that the points above are coded in this below evaluation script that we will use for our fine tuning evaluation.

```python
    # Evaluate on validation
    model.eval()
    torch.set_grad_enabled(False)
    
    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]        
        loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')
        
        # Calculate total loss
        valid_loss = loss.item()
        total_loss = total_loss + valid_loss

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        metrics = document_sentiment_metrics_fn(list_hyp, list_label)

        pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
        
    metrics = document_sentiment_metrics_fn(list_hyp, list_label)
    print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
        total_loss/(i+1), metrics_to_string(metrics)))
```

The evaluation script above uses the `document_sentiment_metrics_fn` function to do the mentioned accuracy, F1 score, recall, and precision metrics calculations, and the following is the snippet of it.

```
def document_sentiment_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics
```

This evaluation then concludes the whole modelling process. We hope you get a good result on your experiment and we hope you enjoyed our short but fun tutorial. If you have finished this tutorial, post your experiences and results in your facebook story. Inspire others to do the same by sharing this tutorial to other deep learning and NLP enthusiasts around the world!
