# ELABORATO PER ESAME FINALE DI INTELLIGENZA ARTIFICIALE

## Obiettivo
L'obiettivo del progetto è stato quello di implementare in python gli algoritmi per l’inferenza e per l’apprendimento con Naive Bayes come descritto in [*McCallum & Nigam 1998*](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf),sia in versione Bernoulli che Multinomiale.Questo al fine di classificare il dataset di documenti [*Large Movie Review Dataset*](https://ai.stanford.edu/~amaas//data/sentiment/) e riportarne le matrici di confusione.

## Come riprodurre i risultati
Ho salvato tramite la funzione *savez_compressed* di *NumPy* tutte le matrici e vettori sparsi che rappresentano il dataset di train e test già processati con il Vectorizer di *sklearn* al fine di velocizzare il processo. 

Per riprodurre i risultati basterà infatti eseguire gli script *BernoulliClassification* e *MultinomialClassification* per verificare il corretto funzionamento della classificazione Naive Bayes in versione Bernoulli e Multinomiale,rispettivamente,sul dataset considerato


