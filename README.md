# Email classification with BERT
Email's text classification with google's BERT model to automatize ticket opening

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

L'accuracy ottenuta sul validation set è di circa il 90%. Si è proceduto allora all'inserimento del test set per valutare l'accuracy su un ulteriore set di dati indipendente da quello di train. Per farlo bisognerà settare il parametro ```do_predict``` uguale a ```True```. Inoltre sarà necessario specificare il path del modello addestrato per il parametro ```init_checkpoint```, in modo tale che vengano utilizzati i pesi del modello aggiornato. La ```max_seq_length``` deve essere la stessa utilizzata per la fase di training.

```ruby
!python /content/bert/run_classifier.py 17 \
--task_name=cola \
--do_predict=true \
--data_dir=/content/data/ \
--vocab_file=/content/multi_cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=/content/multi_cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=/content/bert_output/model.ckpt-1907 \
--max_seq_length=128 \
--output_dir=/content/bert_output/ \
--do_lower_case=False
```

Per il calcolo dell'accuracy bisogna caricare il dataframe generato dalle predizioni e confrontarlo con il test set originario

```ruby
df_result = pd.read_csv('./bert_output/test_results.tsv', delimiter = '\t',encoding="utf-8",header=None)
df_test_completo = pd.read_csv('./data/test_completo.tsv', delimiter = '\t',encoding="utf-8")
```

Per il confronto si crea un nuovo dataframe e, dato che il risultato è espresso in termini di probabilità del testo doi appartenere a una certa classe, si è optato per la creazione di una colonna che dia come output il valore che identifica la classe a cui con maggiore probabilità va associato il testo. viene conservata anche una colonna che riporta il valore di probabilità, nel caso si voglia utilizzare come criterio di valutazione dell'affidabilità della predizione. Chiaramente se la probabilità che un testo vada associato a una certa classe è "spalmato" in più di una classe è probabile che ci sia maggiore ambiguita nella classificazione del testo.

```ruby
risultati_test = pd.DataFrame({'guid': df_test_completo['user_id'],
'text': df_test_completo['text'],
'label':df_test_completo['label'],
'one_hot_label':df_test_completo['one_hot_label'],
'predict_label': df_result.idxmax(axis=1),
'Prob' : df_result.max(axis=1)})
```
Il calcolo dell'accuracy è stato fatto costruendo una tabella di contingenza e valutando la percentuale di testi ben classificati sul totale dei testi.

```ruby
x = pd.crosstab(risultati_test['one_hot_label'],risultati_test['predict_label'])
np.diag(x).sum()/risultati_test.shape[0]
```

L'accuracy si conferma al 90% circa.

Si è proceduto poi allo sviluppo di un apposito strumento per inserire nuove email in modo rapido all'interno del modello addestrato. In questo moso si può testare velocemente lo strumento, mostrarne i risultati in modo semplice a terzi e, in caso di necessità, inserire lo strumento (che si vedrà, altro non è che uno script richiamabile da diversi ambienti) all'interno di un flusso di elaborazione dei dati per automatizzare la classificazione dei testi near real time.

Qui verrà presentata la versione utilizzabile in Colab. Chiaramente il modello può essere scaricato in locale e riutilizzato (con le necessarie modifiche e integrazioni) su una macchina differente per permetterne, ad esempio, l'utilizzo in produzione sui sistemi aziendali.

È stato creato un file shell (si veda il file Script1.sh nel presente repocitory) in cui si definisce un codice che, quando viene richiamato e viene specificato un testo come parametro, crea un file tsv che può essere dato in pasto al modello pre-addestrato.

```
#!/bin/bash

cat > /content/Shell/test.tsv << EOF
guid	text
1	$1
EOF

python /content/bert/run_classifier.py 17 \
--task_name=cola \
--do_predict=true \
--data_dir=/content/Shell/ \
--vocab_file=/content/multi_cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=/content/multi_cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=/content/bert_output/model.ckpt-1907 \
--max_seq_length=192 \
--output_dir=/content/Shell/ \
--do_lower_case=False
```

È stato poi scritto uno script python (si veda il file Script2.py nel presente repocitory) in cui alle colonne vengono riasssegnate le etichette delle classi originarie

```ruby
import pandas as pd;  
import os;
df_result = pd.read_csv('/content/Shell/test_results.tsv', sep = '\t', header=None, encoding='utf-8');
df_result.columns = [
'Informazione;Amministrativo;Fattura',
'Informazione;Amministrativo;Richiesta dilazione di pagamento',
'Informazione;Amministrativo;Situazione contabile',
'Informazione;Amministrativo;Subentro',
'Informazione;Commerciale;Attivazione nuovi Prodotti',
'Informazione;Commerciale;Spedizione',
'Reclamo;Amministrativo;Doppia fatturazione',
'Reclamo;Amministrativo;Fatturazione',
'Reclamo;Commerciale;Attivazione',
'Reclamo;Tecnico;Guasto',
'Variazione;Amministrativo;Anagrafica',
'Variazione;Amministrativo;Fatturazione',
'Variazione;Amministrativo;Modalita di Pagamento',
'Variazione;Commerciale;Rimodulazione Offerta',
'Variazione;Tecnico;Autolettura',
'Variazione;Tecnico;MNP IN Post Attivazione',
'Variazione;Tecnico;Tensione e potenza']
```

Sempre nello stesso script si è optato per il print della classe a cui con maggiore probabilità il testo inserito appartiene, la probabilità che il testo appartenga a quella classe e in coda la probabilità che appartenga alle altre classi

```ruby
print(df_result.idxmax(1));
print(df_result.max(1));
print(df_result);
```

A questo punto è sufficiente creare la directory "Shell" in Colab

```ruby
import os
os.mkdir('./Shell')
```

e richiamare all'interno dello Script1.sh lo script python

```ruby
python /content/pythonscript_percolab.py
```

E direttamente da bash in linux eseguire lo Script1.sh specificando quale frase classificare

```ruby
!sh ./percolab.sh "Salve ragazzi, il cliente 123456 dice di aver pagato più di quanto ha consumato, in allegato vi comunico l'autolettura. Grazie mille e buon lavoro. Soggetto: Comunicazione Autolettura. Da: Mario Rossi, Sales area Milano"
```

Il risultato è il seguente

```ruby
[...]
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:prediction_loop marked as finished
INFO:tensorflow:prediction_loop marked as finished
0    Variazione;Tecnico;Autolettura
dtype: object
0    0.995345
dtype: float64
   Informazione;Amministrativo;Fattura  ...  Variazione;Tecnico;Tensione e potenza
0                             0.000309  ...                               0.000159

[1 rows x 17 columns]
```

Il modello ha previsto correttamente l'apertura di un ticket di "Variazione;Tecnico;Autolettura".


# Conclusioni

Da questo esempio emergono chiaramente le potenzialità di BERT come strumento di Intelligenza Artificiale per il Natural Language Processing. Le applicazioni possono essere numerose e vanno dalla text classification alla costruzione di chatbot alla next sentence prediction alla sentiment analysis e così via. Si tratta quindi di uno strumento che può servire sia all'automatizzazione di processi di lettura dei dati (come in questo esempio), che all'analisi automatica dei testi per la costruzione, ad esempio, di strumetni per valutare la voice of customers.
