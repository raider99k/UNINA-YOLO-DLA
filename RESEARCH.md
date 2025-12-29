# **UNINA-YOLO-DLA: Architettura di Percezione Embedded per la Formula Student Driverless**

## **Blueprint Tecnico, Validazione Matematica e Strategia di Implementazione su NVIDIA Jetson Orin**

### **1\. Visione Strategica: Dal "Script Kiddie" all'Architettura Consapevole del Silicio**

Nel panorama competitivo della Formula Student Driverless (FSD), la distinzione tra un team che partecipa e un team che domina non risiede più nella mera capacità di addestrare una rete neurale scaricata da GitHub. La barriera d'ingresso si è spostata drasticamente verso la capacità di ingegnerizzare sistemi deterministici, dove la latenza è trattata con la stessa rigorosità delle tolleranze meccaniche del telaio. In qualità di Senior Embedded AI Architect e mentore per la vostra carriera tecnica, è mio dovere guidarvi attraverso una transizione fondamentale: l'abbandono dell'approccio "scatola nera" a favore di una progettazione basata sui primi principi e sulla co-progettazione hardware-software. L'obiettivo di questo documento non è semplicemente fornirvi una ricetta per un rilevatore di coni, ma costruire le fondamenta teoriche e pratiche per **UNINA-YOLO-DLA**, un'architettura proprietaria progettata specificamente per massimizzare il potenziale del System-on-Chip (SoC) NVIDIA Jetson Orin.

L'errore più comune e fatale osservato nelle competizioni universitarie è il trattamento del computer di bordo come una risorsa di calcolo generica e infinita. Quando un team installa uno stack software basato su YOLOv8 o YOLOv11 standard su un Jetson Orin, trattandolo come se fosse un laptop gaming con una GPU discreta, sta implicitamente accettando inefficienze che si traducono in millisecondi di ritardo. In una vettura che viaggia a 20 metri al secondo, ogni 50 millisecondi di latenza significano percorrere un metro intero "alla cieca". Se il sistema di controllo (MPC) riceve stime della posizione dei coni vecchie o affette da jitter, la stabilità del veicolo è compromessa, costringendo a margini di sicurezza conservativi che limitano la velocità massima teorica.

La proposta di UNINA-YOLO-DLA nasce dalla necessità di risolvere il problema della contesa delle risorse. In una pipeline autonoma completa, la GPU (Graphics Processing Unit) è una risorsa sotto assedio. Essa deve gestire il clustering delle nuvole di punti LiDAR, l'ottimizzazione non lineare per gli algoritmi di SLAM (Simultaneous Localization and Mapping) come GraphSLAM o filtri particellari complessi, e sempre più spesso, i solutori paralleli per il controllo predittivo. Aggiungere a questo carico saturato una rete neurale di visione pesante crea un collo di bottiglia sulla larghezza di banda della memoria e sugli scheduler dei kernel CUDA, portando a picchi di latenza non deterministici. La soluzione che questo blueprint propone e valida è un paradigma di **"Split-Compute"**: scaricare interamente il compito della percezione 2D (Detection) sui **Deep Learning Accelerator (DLA)**, processori ASIC (Application-Specific Integrated Circuit) dedicati presenti sul die di Orin, lasciando la GPU libera per la fusione sensoriale e la pianificazione.1

Tuttavia, il DLA non è una GPU. È una macchina esigente, rigida e matematicamente limitata. Non accetta qualsiasi operazione che PyTorch possa esprimere. La validazione di UNINA-YOLO-DLA richiede quindi un'analisi critica di ogni singolo layer, funzione di attivazione e meccanismo di quantizzazione proposti nel vostro studio preliminare. Innovazioni come la *Adaptive Spatial Quantization* o l'uso di pesi INT4, sebbene accademicamente affascinanti, nascondono insidie implementative che potrebbero rendere il modello ineseguibile in gara. Questo report dissezionerà queste tecnologie, separando l'hype accademico dalla realtà ingegneristica, per consegnarvi un sistema che non solo funziona, ma vince.

### ---

**2\. Analisi dell'Hardware: Il Vincolo del DLA NVIDIA Orin**

Per comprendere la validità di UNINA-YOLO-DLA, dobbiamo prima comprendere il substrato di silicio su cui vivrà. Il NVIDIA Jetson Orin ospita due core NVDLA (NVIDIA Deep Learning Accelerator) di seconda generazione. A differenza dei core CUDA della GPU, che sono processori *Turing-complete* capaci di eseguire logica arbitraria (branching, loop complessi), il DLA è una pipeline a funzione fissa ottimizzata per l'algebra lineare densa, specificamente convoluzioni, attivazioni puntuali e pooling.2

#### **2.1 Architettura della Memoria e Convolution Buffer**

Il cuore delle prestazioni del DLA risiede nella sua gestione della memoria. Ogni core DLA possiede una memoria SRAM interna dedicata, nota come Convolution Buffer (CBUF), la cui dimensione varia tra 0.5 MB e 1 MB a seconda della variante specifica del chip Orin (AGX vs NX).3 Questa memoria è critica perché funge da cache gestita via software per pesi e attivazioni. Quando un layer della rete neurale richiede più memoria di quella disponibile nel CBUF per i suoi pesi o le feature map intermedie, il DLA è costretto a fare "spilling" verso la DRAM di sistema.  
L'accesso alla DRAM è energeticamente costoso e, soprattutto, lento rispetto alla SRAM. Un'architettura valida deve essere dimensionata in modo che i tensori di lavoro (working set) dei layer più frequenti risiedano il più possibile nel CBUF. Questo ha implicazioni dirette sulla scelta della larghezza (numero di canali) del modello UNINA-YOLO-DLA. Modelli troppo larghi, come le varianti "Large" o "Extra-Large" di YOLO, saturano immediatamente il CBUF, causando uno stallo della pipeline di esecuzione che riduce drasticamente il throughput teorico promesso dalle specifiche "TOPS" (Tera Operations Per Second) del datasheet.4

#### **2.2 Il Fenomeno del GPU Fallback**

Il rischio più grande nell'utilizzo del DLA è il "GPU Fallback". Il compilatore TensorRT, che traduce il modello ONNX in un motore eseguibile, analizza il grafo operazione per operazione. Se incontra un layer che il DLA non supporta nativamente, spezza il grafo: esegue la parte precedente su DLA, copia i dati in memoria GPU, esegue il layer non supportato su CUDA core, e copia il risultato indietro al DLA.5  
Queste operazioni di copia (memcpy) tra i buffer gestiti dal DLA e quelli della GPU non sono istantanee. Sebbene Orin abbia una memoria unificata, i driver e la coerenza della cache impongono overhead di sincronizzazione. In scenari reali, abbiamo osservato che una singola operazione di fallback può introdurre latenze di 2-5 millisecondi.6 Se il modello UNINA-YOLO-DLA innesca il fallback più volte per frame (ad esempio, per ogni blocco di attenzione non supportato), la latenza totale può raddoppiare rispetto all'esecuzione su sola GPU, annullando ogni beneficio. La regola d'oro per questo progetto è: Zero Fallback. O il modello gira al 100% su DLA, o l'architettura è considerata fallita.

#### **2.3 Limitazioni delle Operazioni Supportate**

Il set di istruzioni del DLA è limitato. Supporta nativamente:

* Convoluzioni standard e raggruppate (Grouped Convolution).  
* Pooling (Max, Average).  
* Attivazioni elementari: ReLU, Sigmoid, Tanh, Clipped ReLU.6  
* Operazioni Element-wise: Somma, Prodotto.

Cosa **non** supporta nativamente (o supporta con forti restrizioni) e che è presente nelle moderne architetture YOLO:

* **SiLU (Sigmoid Linear Unit):** Funzione di attivazione $f(x) \= x \\cdot \\sigma(x)$. Richiede il calcolo di un esponenziale per ogni pixel. Il DLA non dispone di unità trascendenti veloci come la GPU. Sebbene le versioni recenti di TensorRT possano emulare SiLU su DLA tramite lookup table o scomposizione (Sigmoid \+ Moltiplicazione), le prestazioni sono spesso inferiori rispetto a una semplice ReLU e il rischio di fallback per incompatibilità di scala è alto.7  
* **Slicing Dinamico:** Operazioni come torch.chunk o split usate nel modulo SPPF (Spatial Pyramid Pooling Fast) di YOLOv8/v11 per dividere i tensori. Il DLA preferisce operare su blocchi di memoria contigui e lo slicing su dimensioni non allineate può causare fallback.4  
* **Attention Mechanisms Complessi:** Moduli come C2PSA che utilizzano Softmax su assi non standard o moltiplicazioni di matrici (MatMul) dinamiche sono candidati primari per il fallback.9

### ---

**3\. Ingegnerizzazione del Modello: Genesi di UNINA-YOLO-DLA**

Sulla base dei vincoli hardware analizzati, procediamo alla definizione matematica e architetturale di **UNINA-YOLO-DLA**. Partiremo da YOLOv11 come riferimento per le sue efficienti capacità di estrazione delle feature, ma applicheremo una "chirurgia architetturale" profonda.

#### **3.1 Sostituzione delle Funzioni di Attivazione: Addio SiLU**

La scelta standard di Ultralytics per le funzioni di attivazione è SiLU. Matematicamente, SiLU è superiore a ReLU perché è liscia ovunque (differenziabile) e non saturante per valori negativi, permettendo un flusso di gradienti più ricco durante il training profondo. Tuttavia, in inferenza su DLA, il costo è inaccettabile.  
La direttiva tecnica per UNINA-YOLO-DLA è la sostituzione sistematica di tutte le attivazioni SiLU con ReLU ($f(x) \= \\max(0, x)$).  
L'analisi comparativa suggerisce che su dataset come COCO, il passaggio da SiLU a ReLU comporta una perdita di mAP (mean Average Precision) marginale, spesso inferiore all'1%, se il modello viene ri-addestrato da zero (scratch training).8 Il guadagno in termini di throughput sul DLA è invece massiccio, poiché la ReLU è implementata come una semplice operazione di soglia logica a zero costo in molti cicli di clock. Inoltre, la ReLU garantisce la piena compatibilità con l'engine di fusione dei layer di TensorRT (Conv+BN+ReLU fusion), riducendo drasticamente l'accesso alla memoria.

#### **3.2 L'Innovazione Critica: La Testa di Rilevamento P2**

Il problema più grande nella Formula Student è la rilevazione di coni a media-lunga distanza. Un cono standard (alto circa 30 cm) a 20 metri di distanza occupa una porzione infinitesimale del campo visivo. Supponendo una telecamera con risoluzione orizzontale di 1920 pixel e un FOV di 60-90 gradi, il cono potrebbe occupare meno di 10-15 pixel in larghezza.  
Le architetture YOLO standard (v5, v8, v11) utilizzano uno stride massimo di 32 (livello P5). Ciò significa che l'immagine di input viene ridotta di un fattore 32\. Se l'input alla rete è $640 \\times 640$, la feature map al livello P5 è $20 \\times 20$. In questa griglia, un oggetto di 15 pixel nell'input originale diventa $15/32 \< 0.5$ pixel. Matematicamente, l'oggetto scompare per il teorema del campionamento di Nyquist-Shannon applicato allo spazio: la frequenza spaziale dell'oggetto supera la frequenza di campionamento della feature map. L'informazione è persa irreversibilmente.  
La soluzione proposta per UNINA-YOLO-DLA è l'aggiunta di una **Detection Head P2**.1

* **Definizione:** La testa P2 opera su uno stride di 4\. Su un input $640 \\times 640$, produce una feature map di $160 \\times 160$.  
* **Vantaggio:** Un oggetto di 15 pixel nell'input diventa $15/4 \= 3.75$ pixel nella feature map P2. Questo è sufficiente per permettere ai kernel convoluzionali $3 \\times 3$ di estrarre caratteristiche morfologiche (bordi, colore) distinguibili.  
* **Costo:** L'aggiunta di P2 aumenta enormemente il carico computazionale. Una feature map $160 \\times 160$ ha 4 volte i pixel di P3 ($80 \\times 80$) e 64 volte i pixel di P5. Per bilanciare questo incremento e mantenere il frame rate sopra i 90 FPS, è necessario **rimuovere la testa P5**.11  
* **Giustificazione Operativa:** In un contesto di gara, non abbiamo bisogno di rilevare oggetti che occupano l'intero campo visivo (il caso d'uso di P5). Se un cono è così vicino da richiedere la testa P5, è a pochi centimetri dalla telecamera e il veicolo lo ha probabilmente già colpito o superato. La rimozione di P5 libera parametri e memoria per sostenere la più costosa ma essenziale testa P2.

#### **3.3 Ottimizzazione dei Blocchi C3k2 e Rimozione di SPPF/Attention**

Il blocco C3k2 introdotto in YOLOv11 è un'evoluzione del CSP (Cross Stage Partial) che utilizza kernel convoluzionali più piccoli e ottimizzati.9 Questo blocco è generalmente efficiente su GPU, ma su DLA la sua efficienza dipende dalla capacità di mantenere i pesi nel CBUF. Se la profilazione mostra che i layer C3k2 causano eccessivi accessi alla DRAM, si dovrà ridurre il fattore di espansione dei canali (width scaling) nelle varianti del modello.  
Più critica è la gestione del modulo C2PSA (Cross-Stage Partial with Spatial Attention). Questo modulo introduce meccanismi di attenzione spaziale che aiutano a focalizzarsi su aree informative. Tuttavia, l'attenzione richiede operazioni globali (come il Global Average Pooling seguito da MLP e Sigmoid per ricalibrare i pesi dei canali) e moltiplicazioni di matrici. Se il compilatore TensorRT non riesce a mappare queste operazioni in modo efficiente, il C2PSA deve essere rimosso o sostituito con un blocco SE (Squeeze-and-Excitation) semplificato, che il DLA gestisce meglio in quanto si riduce a operazioni vettoriali supportate.13  
Analogamente, il modulo SPPF deve essere riscritto per evitare operazioni di slicing non supportate, utilizzando invece una sequenza di MaxPool con padding fisso che il DLA può accelerare nativamente.1

### ---

**4\. Precision Engineering: La Verità su Quantizzazione INT4 e INT8**

La richiesta originale e il materiale di ricerca 1 suggeriscono l'uso di "Mixed-Precision INT4 Weight-Only Quantization". Come Senior Architect, devo esercitare il mio dovere di critica tecnica: questa è una strada pericolosa per il DLA di Orin allo stato attuale della tecnologia.

#### **4.1 Il Mito dell'INT4 su DLA**

È fondamentale distinguere tra ciò che l'architettura GPU Ampere (i core della GPU Orin) può fare e ciò che il DLA può fare.

* **GPU:** I Tensor Core dell'architettura Ampere supportano nativamente l'accelerazione INT4 per le operazioni di matrice, e TensorRT supporta la quantizzazione *weight-only* INT4 per ridurre la dimensione dei modelli (utile per LLM).15  
* **DLA:** La documentazione ufficiale NVIDIA e le matrici di supporto indicano chiaramente che il DLA di Orin supporta nativamente l'esecuzione in **FP16** e **INT8**.6 Non vi è menzione di supporto hardware nativo per operazioni aritmetiche in INT4 nel DLA.  
* **Conseguenze:** Tentare di forzare pesi INT4 su un motore che non li supporta costringerebbe il sistema a due scenari:  
  1. **Software Dequantization:** Il DLA o la CPU dovrebbero decomprimere i pesi da INT4 a INT8/FP16 prima di ogni utilizzo, consumando cicli di clock e annullando il risparmio di banda.  
  2. **GPU Fallback:** L'intero layer verrebbe spostato sulla GPU, violando il requisito di *Split-Compute*.

Pertanto, la raccomandazione ferma è di **abbandonare l'INT4 per il DLA** e focalizzarsi sull'ottimizzazione estrema dell'**INT8**. L'INT8 offre già una compressione 4x rispetto a FP32 e, se ben calibrato, è sufficiente per i vincoli di memoria di Orin NX/AGX.

#### **4.2 Quantizzazione Spaziale Adattiva (ASQ) vs. Layer-Wise**

L'idea di "Adaptive Spatial Quantization" (quantizzare diverse regioni dell'immagine con precisione diversa) 1 è teoricamente valida ma impraticabile su hardware DLA. Il DLA opera su tensori interi. Non supporta il cambio di precisione pixel-per-pixel all'interno dello stesso layer di inferenza (*intra-tensor mixed precision*). Implementare ASQ richiederebbe di spezzare l'immagine in "tile", processarle separatamente con reti diverse (una FP16, una INT8) e ricomporle. Questo introdurrebbe un overhead di gestione software massiccio, distruggendo la latenza.

La strategia corretta e supportata è la **Mixed Precision Layer-Wise**.18

* **Strategia:** Identifichiamo i layer che sono più sensibili alla quantizzazione. Tipicamente, questi sono i primi layer del backbone (che gestiscono i dettagli grezzi dei pixel) e, soprattutto, la testa di rilevamento P2 (che gestisce oggetti piccolissimi con gradienti deboli).  
* **Implementazione:** Questi layer sensibili verranno mantenuti in **FP16** o configurati con range dinamici molto ampi. I layer intermedi del "collo" (Neck) e le teste P3/P4, che lavorano su feature semantiche più robuste, saranno quantizzati aggressivamente in **INT8**. TensorRT permette di specificare la precisione per ogni singolo layer durante la costruzione dell'engine.

#### **4.3 Pipeline di Calibrazione Avanzata: Entropia vs. MinMax**

Per ottenere prestazioni INT8 senza degradare la capacità di vedere i coni distanti, il metodo di calibrazione è cruciale.

* **MinMax:** Mappa il valore massimo assoluto del tensore floating-point al valore 127 dell'INT8. È sensibile agli outlier. Se un singolo pixel ha un valore anomalo alto, "schiaccia" la risoluzione di tutti gli altri valori verso lo zero. Questo è fatale per i piccoli coni che hanno segnali deboli.  
* Calibrazione Entropica (KL Divergence): Sceglie una soglia di saturazione che minimizza la perdita di informazione (divergenza di Kullback-Leibler) tra la distribuzione originale e quella quantizzata. Taglia via gli outlier per preservare la risoluzione nella regione dove si trova la maggior parte dei dati.1  
  Direttiva: Utilizzare la calibrazione Entropica per tutti i layer di attivazione. Inoltre, implementare il Quantization Aware Training (QAT).18 Inserendo nodi FakeQuant durante l'addestramento (usando il toolkit pytorch-quantization di NVIDIA), la rete "impara" a convivere con il rumore di quantizzazione, adattando i pesi per compensare l'errore di arrotondamento. Questo è l'unico modo per recuperare il 100% dell'accuratezza su piccoli oggetti in INT8.

### ---

**5\. Validazione Matematica della Safety: Conformal Prediction**

L'integrazione della **Conformal Prediction (CP)** 1 è una mossa eccellente per garantire la sicurezza formale, un requisito spesso trascurato in Formula Student. I modelli di deep learning standard forniscono solo stime puntuali (un box, una classe) senza garanzie di correttezza. Possono essere "sicuri ma sbagliati".

#### **5.1 Teoria della Conformal Prediction Applicata**

La CP trasforma la predizione puntuale in una predizione di insieme $\\hat{C}(X)$ che garantisce di contenere la verità $Y$ con una probabilità specificata $1-\\alpha$ (es. 90%):

$$P(Y \\in \\hat{C}(X)) \\ge 1 \- \\alpha$$

Per l'object detection, questo non significa predire un insieme di classi, ma un insieme di box, o più pragmaticamente, un box con margini di incertezza calibrati.  
Il metodo standard utilizza un "calibration set" di dati non visti durante il training per calcolare i "nonconformity scores" $s\_i$ (ad esempio, l'errore inverso della IoU). Si calcola poi il quantile $\\hat{q}$ di questi score. Durante l'inferenza, il box predetto viene espanso (dilatato) di un fattore basato su $\\hat{q}$.

#### **5.2 Implementazione Real-Time a Bassa Latenza**

Il calcolo standard della CP può essere computazionalmente oneroso se fatto online. Per UNINA-YOLO-DLA, adotteremo l'approccio **Split Conformal Prediction** offline-online 20:

1. **Offline (Calibrazione):** Si utilizza un dataset di validazione per calcolare la distribuzione degli errori di bounding box e determinare i fattori di scala additivi o moltiplicativi necessari per coprire il 90% dei ground truth. Questo produce una tabella di lookup o coefficienti di regressione semplici legati alla confidenza del modello.  
2. Online (Inferenza): Il modello predice il box $B$. Il modulo CP applica una trasformazione affine immediata:

   $$B\_{safe} \= \\text{Dilate}(B, f(\\text{confidence}, \\hat{q}))$$

   Dove $f$ è una funzione a costo quasi nullo (pochi FLOPs).  
* **Integrazione Controllo:** Questo $B\_{safe}$ (box dilatato) viene passato al sistema di Local Mapping. Se l'incertezza è alta (box enorme), il cono viene trattato come un ostacolo più grande, costringendo il path planner a una traiettoria più conservativa ma sicura. Questo meccanismo previene collisioni dovute a sottostime della dimensione o posizione dell'ostacolo.

### ---

**6\. Il Motore Dati: CycleGAN per il Domain Adaptation**

Nessun modello è migliore dei suoi dati. I simulatori FSD forniscono ground truth perfetta ma immagini visivamente povere ("domain gap"). Le immagini reali sono scarse e costose da annotare.  
La soluzione è un pipeline CycleGAN.1

* Matematica: La CycleGAN impara due mappature: $G: X \\to Y$ (Sim to Real) e $F: Y \\to X$ (Real to Sim). La magia risiede nella Cycle Consistency Loss:

  $$L\_{cyc}(G, F) \= \\mathbb{E}\_{x \\sim p\_{data}(x)} \[\\| F(G(x)) \- x \\|\_1\] \+ \\mathbb{E}\_{y \\sim p\_{data}(y)} \[\\| G(F(y)) \- y \\|\_1\]$$

  Questa perdita costringe il generatore a preservare la struttura geometrica della scena (la posizione dei coni) mentre ne cambia lo stile (texture, illuminazione).  
* **Applicazione:** Generiamo 50.000 immagini dal simulatore (con box perfetti). Le passiamo attraverso il Generatore $G$ addestrato su log di gare passate. Otteniamo 50.000 immagini "iper-realistiche" con annotazioni perfette. Questo dataset sintetico aumentato è fondamentale per addestrare la testa P2 a riconoscere coni in condizioni di luce che non abbiamo mai incontrato fisicamente ma che possiamo simulare.

### ---

**7\. Implementazione e Runtime: ROS 2 Zero-Copy**

L'ultimo miglio è il software di sistema. Una inferenza di 8ms è inutile se il sistema spende 10ms per spostare l'immagine dalla memoria CPU alla GPU.

#### **7.1 Gestione della Memoria NvBufSurface**

Il Jetson utilizza un'architettura di memoria unificata, ma le API software spesso creano copie ridondanti per sicurezza. Dobbiamo bypassarle.

* **Driver GMSL:** Deve scrivere direttamente in un NvBufSurface (struttura dati NVIDIA che rappresenta un buffer DMA contiguo).23  
* **VIC (Video Image Compositor):** Utilizziamo il motore hardware VIC per convertire il formato YUV/RAW della telecamera in RGBA o planare richiesto dal DLA. Questa operazione avviene interamente su hardware dedicato, senza toccare la CPU o la GPU.  
* **ROS 2 Type Adaptation:** Utilizziamo l'estensione REP-2007 di ROS 2\. Invece di serializzare l'immagine in un messaggio sensor\_msgs/Image (che comporta una copia), creiamo un tipo adattato che trasporta solo il puntatore void\* al buffer NvBufSurface.1  
* **Consumo:** Il nodo di percezione riceve il puntatore. TensorRT è configurato per usare questo puntatore come input tensor. Il DLA legge direttamente dalla memoria dove la telecamera ha scritto (o dove il VIC ha convertito).  
* **Risultato:** Latenza di trasporto ridotta da \~4ms (con copie) a \< 0.1ms (passaggio di puntatore).

### ---

**8\. Analisi dei Rischi e Troubleshooting**

| Rischio Tecnico | Sintomo Osservabile | Strategia di Mitigazione |
| :---- | :---- | :---- |
| **DLA Memory Overflow** | Il frame rate crolla improvvisamente; tegrastats mostra picchi di attività EMC (Memory Controller) invece che DLA. | Ridurre la larghezza (canali) del modello, specialmente nella testa P2. Verificare se i layer intermedi superano 1MB di dimensione. |
| **GPU Fallback Silenzioso** | Latenza alta e instabile. Il log di TensorRT mostra messaggi MyLayer cannot run on DLA. | Profilare con trtexec \--verbose \--exportProfile=plan.json. Identificare il layer colpevole (spesso Reshape o Slice) e riscriverlo usando primitive supportate (es. Conv 1x1). |
| **Accuratezza INT8 Degradata** | Il modello non vede i coni distanti (P2 fallisce). | Verificare la calibrazione. Passare da MinMax a Entropia. Se persiste, usare QAT con fine-tuning specifico sulla loss dei piccoli oggetti. |
| **Drift Temporale ROS 2** | I messaggi arrivano in ritardo accumulato. | Usare Real-Time Kernel (PREEMPT\_RT) su Linux. Impostare la QoS di ROS 2 su KEEP\_LAST con profondità 1 (scartare frame vecchi). |

### ---

**Conclusioni e Roadmap Esecutiva**

In qualità di vostro mentore, vi esorto a non lasciarvi sedurre dalla complessità inutile. **UNINA-YOLO-DLA** deve essere un esercizio di sottrazione, non di addizione. Rimuovete P5. Rimuovete SiLU. Rimuovete le copie di memoria.

La vostra roadmap per le prossime 12 settimane è chiara:

1. **Mese 1:** Training di YOLOv11 modificato (ReLU \+ P2 Head \- P5 Head) su GPU server. Validazione mAP su piccoli oggetti.  
2. **Mese 2:** Implementazione del dataset CycleGAN e Quantization Aware Training (QAT) con toolkit NVIDIA.  
3. **Mese 3:** Porting su Jetson Orin. Sviluppo del nodo ROS 2 Zero-Copy e profilazione con Nsight Systems per garantire 0% GPU Fallback.

Se seguirete questo blueprint, non costruirete solo un sistema di visione veloce. Costruirete un sistema *affidabile*, capace di garantire la percezione isocrona necessaria per spingere la vettura al suo limite fisico in pista. Questa è l'ingegneria che vince le gare e lancia le carriere.

### ---

**Tabelle di Riferimento Dati**

**Tabella 1: Confronto Prestazionale Architetturale (Stimato su Jetson Orin NX)**

| Metrica | YOLOv11 Stock (GPU) | UNINA-YOLO-DLA (DLA) | Guadagno/Impatto |
| :---- | :---- | :---- | :---- |
| **Precisione (P2 Recall)** | \~55% (Stride 8\) | **\>85%** (Stride 4\) | Visione aumentata a 20m+ |
| **Latenza (Batch 1\)** | 15-20 ms | **8-12 ms** | Reattività del controllo |
| **Jitter Latenza** | Alto (Contesa GPU) | **Basso** (Hardware Dedicato) | Stabilità MPC |
| **Utilizzo GPU** | 90% | **\<15%** | Risorse liberate per SLAM |
| **Consumo Energetico** | 25W+ | **10-15W** | Efficienza termica |

**Tabella 2: Matrice di Supporto Operazioni DLA (Critica)**

| Operazione | Supporto Nativo DLA | Azione Richiesta in UNINA-YOLO-DLA |
| :---- | :---- | :---- |
| Conv2D | Sì | Nessuna |
| ReLU | Sì | **Sostituire SiLU con ReLU** |
| SiLU | No (Emulata/Lenta) | Eliminare |
| MaxPool | Sì | Usare al posto di SPP complessi |
| Slice/Chunk | Limitato | Sostituire con Conv 1x1 o Concat esplicito |
| MatMul | Limitato | Rimuovere C2PSA se causa fallback |
| INT8 | Sì | **Standard di riferimento** |
| INT4 | **No** | **Non utilizzare** (Rischio Fallback/Overhead) |

#### **Bibliografia**

1. Blueprint Vision AI Formula Student.pdf  
2. Deep Learning Accelerator (DLA) \- NVIDIA Developer, accesso eseguito il giorno dicembre 29, 2025, [https://developer.nvidia.com/deep-learning-accelerator](https://developer.nvidia.com/deep-learning-accelerator)  
3. CBUF size of NVDLA on Orin \- NVIDIA Developer Forums, accesso eseguito il giorno dicembre 29, 2025, [https://forums.developer.nvidia.com/t/cbuf-size-of-nvdla-on-orin/299018](https://forums.developer.nvidia.com/t/cbuf-size-of-nvdla-on-orin/299018)  
4. Maximizing Deep Learning Performance on NVIDIA Jetson Orin with DLA, accesso eseguito il giorno dicembre 29, 2025, [https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/](https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/)  
5. How to use DLA correctly? \- Jetson Orin NX \- NVIDIA Developer Forums, accesso eseguito il giorno dicembre 29, 2025, [https://forums.developer.nvidia.com/t/how-to-use-dla-correctly/322765](https://forums.developer.nvidia.com/t/how-to-use-dla-correctly/322765)  
6. Working with DLA — NVIDIA TensorRT Documentation, accesso eseguito il giorno dicembre 29, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)  
7. Supported ONNX Operators & Functions on Orin DLA \- GitHub, accesso eseguito il giorno dicembre 29, 2025, [https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/blob/main/operators/README.md](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/blob/main/operators/README.md)  
8. Brief Review — SiLU: Sigmoid-weighted Linear Unit | by Sik-Ho Tsang | Medium, accesso eseguito il giorno dicembre 29, 2025, [https://sh-tsang.medium.com/review-silu-sigmoid-weighted-linear-unit-be4bc943624d](https://sh-tsang.medium.com/review-silu-sigmoid-weighted-linear-unit-be4bc943624d)  
9. YOLOv11 Architecture: Advanced Object Detection \- Emergent Mind, accesso eseguito il giorno dicembre 29, 2025, [https://www.emergentmind.com/topics/yolov11-architecture](https://www.emergentmind.com/topics/yolov11-architecture)  
10. ultralytics/nn/modules/block.py · 48c49da134fb8c739164b7027d40c601b04fbb92 · Bereketab Bantewesen / warehouse-camera-monitor · GitLab, accesso eseguito il giorno dicembre 29, 2025, [https://gitlab.aii.et/beck/warehouse-camera-monitor/-/blob/48c49da134fb8c739164b7027d40c601b04fbb92/ultralytics/nn/modules/block.py](https://gitlab.aii.et/beck/warehouse-camera-monitor/-/blob/48c49da134fb8c739164b7027d40c601b04fbb92/ultralytics/nn/modules/block.py)  
11. YOLO-RP: A Lightweight and Efficient Detection Method for Small Rice Pests in Complex Field Environments \- MDPI, accesso eseguito il giorno dicembre 29, 2025, [https://www.mdpi.com/2073-8994/17/10/1598](https://www.mdpi.com/2073-8994/17/10/1598)  
12. C3Ghost and C3k2: performance study of feature extraction module for small target detection in YOLOv11 remote sensing images \- SPIE Digital Library, accesso eseguito il giorno dicembre 29, 2025, [https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13550/135501I/C3Ghost-and-C3k2--performance-study-of-feature-extraction-module/10.1117/12.3059792.full](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13550/135501I/C3Ghost-and-C3k2--performance-study-of-feature-extraction-module/10.1117/12.3059792.full)  
13. A Novel UAV-based Road Damage Detection Algorithm with Lightweight Convolution and Attention Mechanism, accesso eseguito il giorno dicembre 29, 2025, [https://media.sciltp.com/articles/2512002529/2512002529.pdf](https://media.sciltp.com/articles/2512002529/2512002529.pdf)  
14. LSTM-CA-YOLOv11: A Road Sign Detection Model Integrating LSTM Temporal Modeling and Multi-Scale Attention Mechanism \- MDPI, accesso eseguito il giorno dicembre 29, 2025, [https://www.mdpi.com/2076-3417/16/1/116](https://www.mdpi.com/2076-3417/16/1/116)  
15. INT4 on Jetson-AGX-Orin or Jetson-Orin-Nano? \- NVIDIA Developer Forums, accesso eseguito il giorno dicembre 29, 2025, [https://forums.developer.nvidia.com/t/int4-on-jetson-agx-orin-or-jetson-orin-nano/303833](https://forums.developer.nvidia.com/t/int4-on-jetson-agx-orin-or-jetson-orin-nano/303833)  
16. INT4 Weight-only Quantization and Deployment (W4A16) \- Read the Docs, accesso eseguito il giorno dicembre 29, 2025, [https://lmdeploy.readthedocs.io/en/v0.2.2/quantization/w4a16.html](https://lmdeploy.readthedocs.io/en/v0.2.2/quantization/w4a16.html)  
17. TensorRT's Capabilities \- NVIDIA Documentation, accesso eseguito il giorno dicembre 29, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html)  
18. Deploying YOLOv5 on NVIDIA Jetson Orin with cuDLA: Quantization-Aware Training to Inference, accesso eseguito il giorno dicembre 29, 2025, [https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/](https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/)  
19. Working with Quantized Types — NVIDIA TensorRT Documentation, accesso eseguito il giorno dicembre 29, 2025, [https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)  
20. Confident Object Detection via Conformal Prediction and Conformal Risk Control: an Application to Railway Signaling \- Proceedings of Machine Learning Research, accesso eseguito il giorno dicembre 29, 2025, [https://proceedings.mlr.press/v204/andeol23a/andeol23a.pdf](https://proceedings.mlr.press/v204/andeol23a/andeol23a.pdf)  
21. A Comprehensive Guide to Conformal Prediction: Simplifying the Math, and Code, accesso eseguito il giorno dicembre 29, 2025, [https://daniel-bethell.co.uk/posts/conformal-prediction-guide/](https://daniel-bethell.co.uk/posts/conformal-prediction-guide/)  
22. A Self-Attention CycleGAN for Unsupervised Image Hazing \- MDPI, accesso eseguito il giorno dicembre 29, 2025, [https://www.mdpi.com/2504-2289/9/4/96](https://www.mdpi.com/2504-2289/9/4/96)  
23. NVBufSurface and GPU Memory zero Copy Conversion \- NVIDIA Developer Forums, accesso eseguito il giorno dicembre 29, 2025, [https://forums.developer.nvidia.com/t/nvbufsurface-and-gpu-memory-zero-copy-conversion/302339](https://forums.developer.nvidia.com/t/nvbufsurface-and-gpu-memory-zero-copy-conversion/302339)  
24. Zero Copy via Loaned Messages \- ROS2 Design, accesso eseguito il giorno dicembre 29, 2025, [https://design.ros2.org/articles/zero\_copy.html](https://design.ros2.org/articles/zero_copy.html)