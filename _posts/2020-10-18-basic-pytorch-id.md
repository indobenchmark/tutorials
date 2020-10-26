---
layout: post
title: Tutorial penggunaan Pre-Trained Model untuk NLP dengan menggunakan PyTorch
sub_tile: "Tutorial penggunaan IndoNLU untk Pemula"
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

To show the difference between those methods, we will show you back the previous example!

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

Sebelum maju lebih jauh pada pemodelan deep learning model dengan PyTorch, kita akan membahas sedikit lebih jauh mengenai jenis-jenis task di NLP.

<div id="tasks"></div>
#### Kategori task dibidang NLP
Dibidang NLP, terdapat banyak task yang dapat dicapai. Kita dapat membagi task-task tersebut kedalam beberapa kategori sebagai berikut
- Text Classification
    Klasifikasi dokumen dengan sebuah label. Contoh pendeteksian emosi sebelumnya adalah salah satu contoh dari kategori ini. Kita memprediksi label emosi bergantung dari kalimat yang diberikan.
- Sequence Labeling
    Kategori task ini melakukan prediksi label untuk setiap token dalam teks masukan. Task yang paling umum dalam kategori ini adalah Named Entity Recognition yang bertujuan untuk mendeteksi token pada kalimat masukan mana yang merupakan nama dari suatu entitas named entities.
- Language Generation
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