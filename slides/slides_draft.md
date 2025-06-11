In generale sappiamo che le Bayesian Network ci danno informazioni solo sulle associazioni statistiche fra le variabili ma non associazioni di tipo causali.

Per evidenziare le relazioni di causalità si utilizza nella causal inference un tipo nuovo di grafo detto Causal Graphical Model, in cui ogni *edge* rappresenta una relazione di causalità (per definizione).

La relazione di causalità può essere studiata soltanto nel contesto in cui si agisce direttamente su una variabile, forzandone un valore e misrandone gli effetti --come si fa in qualsiasi esperimento scientifico. 
La nozione di intervento è stata formalizzata da Pearl (fonte???) e caratterizzata tramite l'operatore $do()$. 

Per intervento si intende forzare un trattamento su TUTTI gli individui e misurarne i risultati, espressi dalle variabili aleatorie: $Y|do(T=1)$ e $Y|do(T=0)$.

Una misura dell'intensità della relazione diretta di causalità tra $T$ e $Y$ può essere definita allora da:

$$
\mathbb{E}[Y|do(T=1)]-\mathbb{E}[Y|do(T=0)]
$$

Il formalismo dell'operatore do implica l'esistenza di scenari ipotetici in cui studiamo contemporaneamente gli effetti di due trattamenti diversi su tutta la popolazione.

<!-- Il problema è che non abbiamo un modo diretto di calcolare espressioni che contengono il $do()$ operator, ma possiamo calcolarle soltanto sotto particolari assunzioni. -->

Per ottenere una stima di $\mathbb{E}[Y|do(T=1)]$  dovremmo condurre un RCT, tale per cui siamo sicuri che l'unica associazione esistente tra trattamento ed outcome è di tipo causale. Varrebbe dunque:

$$
P(Y|do(T=t)) = P(Y|T=t) 
$$

In generale però vogliamo stimare la stessa quantità a partire da dati osservazionali, cioè senza aver effettuato un RCT. In generale esisteranno delle variabili dette confounders tali che il loro effetto influenza sia il tarttamento che l'outcome. Queste introducono dunque delle associazioni statistiche spurie, diverse dal meccanismo di causalità.

L'idea è allora quella di immaginare, sulla base dei dati osservati, un ipotetico mondo nel quale abbiamo effettuato il trattamento. La stima corretta è allora data dalla formula di Backdoor Adjustment 

$$
P(Y|do(T))=\sum_z P(Y|T,X=x)P(X=x)
$$

Intuitivamente l'idea è che condizionando su $X$, blocchiamo il **backdoor parth** che crea la relazione spuria tra T e Y. Ciò che rimane tra le due è soltanto l'associazione causale.

Problema: nella realtà anche se avessimo una lista molto grande di possibili confondenti (scenario tipico oggi nel regime dei big data), anche se condizionassimo su tutti non avremmo mai la certezza di aver bloccato tutti i backdoor path. Potrebbe esserci qualche **variabile latente confondente** che non abbiamo preso in considerazione, tale da creare un nuovo backdoor path.

**Idea dei CEVAE:** assumere che i confondenti osservati sono in realtà proxy (osservate) $X$ di un confondente non osservato $Z$. Allora usare i VAE per stimare il confondente $Z$ (se è vero che esiste ed influenza X,T ed Y, dovrei avere abbastanza informazioni per farne una stima) ed una volta stimato, fissarlo per bloccare il backdoor path assunto ed avere una stima più ragionevole per il tretment effect.

Dunque cerchiamo di stimare tutta la correlazione che troviamo tra le tre in modo da condizionarla via in seguito. Inoltre il vantaggio dei CEVAE è che forniscono un modello generativo (del tipo SCM di Pearl?) tale da essere in grado di generare controfattuali.



## Struttura delle slides
1. dai PGM ai CGM
2. intervento (martello) e do-operator