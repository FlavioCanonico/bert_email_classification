# bert_email_classification
Email's classification with google's BERT model to automatize ticket opening

# Intro

In questo repository viene presentato un progetto di natural language processing. Nello specifico lo scopo è automatizzare la classificazione delle email ricevute dalla rete vendita per evitare la lettura delle singole email da parte degli operatori. In un secondo step del progetto si automatizzerà l'apertura dei ticket e la loro associazione ai singoli id cliente sul CRM aziendale. In questo modo si potrà ottimizzare l'utilizzo delle risorse umane utilizzate per questa operazione e minimizzare gli errori di apertura dei ticket dovute alla necessaria disuniformità nell'interpretazione dei messaggi da parte degli operatori.
L'azienda in cui è stato svolto il progetto opera principalmente nel mercato energetico e delle telecomunicazioni.

Il principale strumento utilizzato è BERT, una rete neurale pre-addestrata e rilasciata gratuitamente per la prima volta alla fine del 2018 da Google a questo [indirizzo](https://github.com/google-research/bert). Verrà utilizzata la versione di BERT scritta in Tensorflow.

Per adattare la rete neurale ai nostri scopi è stata necessaria una fase di fine-tuning. Per prima cosa è strato costruito un dataset manualmente composto dai testi delle email classificati con una delle triplette utilizzate per nel sistema di ticketing. Ad esempio la mail si riferisce a un problema nella spedizione del router verrà classificata come un'Informazione;Commerciale;Spedizione, se fa riferimento alla richiesta di informazioni sulla fattura verrà classificata come un'Informazione;Amministrativo;Fattura, e così via. Sono state individuate un totale di 17 classi.
Una volta costruito il dataset sono state utilizzate delle metodologie di text augmentation per rendere il dataset più ampio, poiché in linea generale le performance della rete aumentano al numero di esempi utilizzati nella fase di fine-tuning. Si è ottenuto un dataset di circa 3000 email pre-classificate che per ragioni di privacy non può essere caricato su questo repository github. Il condice utilizzato invece viene rilasciato in forma semplificata nei vari file del repository.

Per il data pre-processing è stato utilizzato python3 e in particolar modo si è utilizzato Google Colab nella sua forma gratuita, addestrando la rete con una singola GPU.

Per prima cosa sono stati importati i dati e clonato il repository github di BERT

```ruby
from google.colab import files
files.upload()

!git clone https://github.com/google-research/bert.git
```

Lo script python del repository per la text-classification è run_classifier.py, che è stato modificato per i nostri scopi. La modifica è stata fatta alla line 354 dello script per modificare la funzione get_labels per adattarla al numero di classi di output utilizzate in questa applicazione. Per evitare di modificare di volta in volta lo script viene definita una funzione generica che permette di ridefinire come parametro modificabile il numero delle classi di output dall'esterno quando verrà richiamato lo script.

```ruby
  def get_labels(self):
    """See base class."""
    return [str(x) for x in range(int(sys.argv[1]))]
```

Una volta sostituito il nuovo run_classifier.py al vecchio caricato in colab, bisogna scaricare BERT model preaddestrato. Nello specifico è stata utilizzata la versione multilingual che è utilizzabile anche per l'italiano nella sua versione più aggiornata, vale a dire la Cased.

```ruby
!wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
!unzip the file
!unzip multi_cased_L-12_H-768_A-12.zip
```

Conviene lavorare su un virtual environment, che verrà chiamato bertenv

```ruby
!pip3 install virtualenv
!virtualenv bertenv
!source ./bertenv/bin/activate
```

# Data pre-processing

A questo punto si passa alla fase di pre-processing. I dati devono avere una forma adatta per essere inseriti in BERT. Nello specifico il dataset di train e di validation (dev) devono avere quattro colonne:
* codice alfanumerico identificativo della riga
* etichetta che identifica la classe di appartenenza
* testo che viene utilizzato solo nel caso di task di next sentence prediction (riempita simbolicamente con una "a" in questo caso perché non utilizzata)
* testo vero e proprio utilizzato per l'addestramento

Il dataset di test avrà solo la prima e la quarta colonna per ovvie ragioni e a differenza del train e validation set avrà anche l'intestazione delle colonne.

```ruby
from pandas import DataFrame
import numpy as np
import pandas as pd

df=df.dropna()

df_bert = pd.DataFrame({'user_id': range(1, len(df) + 1),
  'label': df['title'],
  'alpha': ['a']*df.shape[0],
  'text': df['messaggio_all'].replace(r'\n',' ',regex=True)})
```

Si è proceduto poi con l'encoding numerico delle etichette e con lo split in train e test set. Si è optato per la creazione di un dataframe di test con tutte e quattro le colonne da utilizzare poi per valutare le performance del modello sul test set tramite il confronto tra risultati previsti e reali. Dal train set è stato poi estratto il dataset di validation da usare durante il fine-tuning di BERT. Lo split è stato effettuato mediante una procedura casuale ma mantenendo una proporzionalità numerica tra le classi del set di dati originario e i sottoinsiemi di dati generati.

```ruby
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
df_bert['one_hot_label'] = labelencoder_X.fit_transform(df_bert['label'])

df_bert_train, df_bert_test_completo = train_test_split(df_bert, test_size=0.2, random_state=1234,stratify=df_bert["label"])

df_bert_train =  pd.DataFrame({'user_id': df_bert_train['user_id'],
  'label': df_bert_train['one_hot_label'],
  'alpha': df_bert_train['alpha'],
  'text':df_bert_train['text']})

df_bert_train, df_bert_dev = train_test_split(df_bert_train, test_size=0.10, random_state=1234
                                              ,stratify=df_bert["label"]
                                              )


df_bert_test =  pd.DataFrame({'user_id': df_bert_test_completo['user_id'],
  'text': df_bert_test_completo['text']})
```

I dati sono stati salvati in formato tsv (con colonne separate da tab), per rispettare il modo in cui vengono letti nello script run_classifier.py


# Fine Tuning
Rispetto a quanto riportato nell'esempio originario di Google per la classificazione binaria dei testi con BERT, si è deciso di effettuare le modifiche al run_classifier e mandare direttamente lo script da Shell in Linux, ottenendo così un codice più snello e facilmente modificabile. Il codice va scritto senza spazi aggiuntivi.

```ruby
!python /content/bert/run_classifier.py 17\
 --task_name=cola\
 --do_train=true\
 --do_eval=true\
 --data_dir=/content/data/\
 --vocab_file=/content/multi_cased_L-12_H-768_A-12/vocab.txt\
 --bert_config_file=/content/multi_cased_L-12_H-768_A-12/bert_config.json\
 --init_checkpoint=/content/multi_cased_L-12_H-768_A-12/bert_model.ckpt\
 --max_seq_length=128\
 --train_batch_size=32\
 --learning_rate=2e-5\
 --num_train_epochs=30.0\
 --output_dir=/content/bert_output/\
 --do_lower_case=False
```

Per un approfondimento dei vari parametri settati si veda la guida ufficiale di BERT.
In colab, utilizzando la GPU disponibile gratuitamente e con i dati a disposizione, in questa applicazione non è stato possibile superare una max_seq_length pari a 194. In qeusto caso si è optato per impostarla pari a 128. Questo paramentro indica la lunghezza massima del vettore di input generato tramite il word embedding a partire dai testi inseriti. Il limite massimo in BERT è pari a 512, tuttavia la maggior parte dei testi inseriti rientrano in questo limite di 128 per cui non è necessario aumentare il costo computazionale e di conseguenza la durata del training del modello.

L'accuracy ottenuta sul validation set è di circa il 90%. Si è proceduto allora all'inserimento del test set per valutare l'accuracy su un ulteriore set di dati indipendente da quello di train. Per farlo bisognerà settare il parametro ```do_predict``` uguale a ```True```
