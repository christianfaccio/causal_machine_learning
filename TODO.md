# Possibile struttura:

1. Introdurre il problema dei confounders in causal inference 
2. Introdurre il modello CEVAE e spiegarne l'idea principale --  Cfr. [https://arxiv.org/abs/1705.08821](Causal Effect Inference with Deep Latent-Variable Models)
3. Confronto CEVAE con modelli di causal inference antecedenti (e.g. doWhy ? quali altri?) -- sul senso delle variabili latenti e delle proxies
    - su dati sintetici -- inventiamo un problema simpatico?
    - su dati reali -- quali? Cancro  [https://portal.gdc.cancer.gov/](GDC Data Portal) ? spotify (se bohemian rapsody fosse stata più ballabile sarebbe stata più popolare) ?
4. Analisi dei risultati
    - generazione di controfattuali (sia nel caso di dati sintetici che reali) -- può essere interessante per la presentazione
    - studio dello spazio latente: è interpretabile? com'è fatto (studiando il vettore medio della normale multivariata)?
    - analisi di robustezza -- Cfr. [https://arxiv.org/abs/2102.06648](A Critical Look at the Consistency of Causal Estimation With Deep Latent Variable Models)
        - **nota:** tutti i parametri da fissare possono essere messi in discussione, in particolare la dimensione dello spazio latente. Mostrare ad esempio come varia l'errore al variare della dimensione dello spazio latente (immagino ci sia un plateau ad un certo punto)
5. Conclusioni e futuri sviluppi

# TODO

- [ ] (cuda optimization)
- [ ] Add a method to sample from the latent space and decode it
- [ ] Add more methods for explainability and analysis
- [ ] Interpretation of the latent space on syntetic and real data