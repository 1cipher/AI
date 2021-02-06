ELABORATO PER ESAME FINALE DI INTELLIGENZA ARTIFICIALE
AUTORE:Alessio Bonacchi

Tutta la parte del codice è stata prodotta in autonomia,tuttavia ho preso spunto da alcuni siti per la realizzazione di due passaggi secondari:
1)https://stackoverflow.com/questions/57144762/cope-with-different-slicing-behaviour-in-scipy-sparse-and-numpy : per far fronte alla conversione di righe di una csr sparse matrix ad una array.

2)LemmaTokenizer() function: presentata nell'ultima videolezione del corso.Ho tuttavia reimplementato, per interesse personale, tutto il meccanismo di text preprocessing senza usare le matrici sparsi(anche se ovviamente per motivi di efficienza non viene usato)



Per riprodurre i risultati sarà necessario scaricare il dataset https://ai.stanford.edu/~amaas//data/sentiment/ e rimpiazzare all'inizio del main.py (modulo di load_files train e test set) la directory corrispondente a dove è stato salvato il file.
