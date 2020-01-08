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
'Variazione;Tecnico;Tensione e potenza'
]

print(df_result.idxmax(1));
print(df_result.max(1));
print(df_result);
