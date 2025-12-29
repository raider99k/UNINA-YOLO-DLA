UNINA-YOLO-DLA: Master Prompts for AI Agent ExecutionAuthor: Senior Embedded AI ArchitectTarget: NVIDIA Jetson Orin (DLA Cores)Context: Formula Student Driverless - Perception StackIstruzioni per l'UsoCopia e incolla questi blocchi sequenzialmente nel tuo AI Coding Assistant (Cursor, Windsurf, GitHub Copilot Chat). Non modificare i vincoli hardware (scritti in MAIUSCOLO) poiché derivano direttamente dall'analisi del silicio.MODULO 1: Architettura Neurale (PyTorch & YAML)Obiettivo: Definire una topologia di rete che rispetti i limiti del DLA (no operazioni trascendenti, no slicing dinamico).Prompt:Agisci come Senior Deep Learning Engineer specializzato in NVIDIA Jetson Orin e TensorRT.
Devo definire l'architettura per "UNINA-YOLO-DLA", una variante di YOLOv11 ottimizzata per girare esclusivamente sui Deep Learning Accelerator (DLA).

Genera due file:
1. `unina-yolo-dla.yaml` (Configurazione Modello)
2. `dla_blocks.py` (Implementazioni Custom)

RISPETTA RIGOROSAMENTE I SEGUENTI VINCOLI HARDWARE (NON NEGOZIABILI):

1. **BANNARE SiLU**: Il DLA non gestisce efficientemente SiLU. Sostituisci sistematicamente TUTTE le funzioni di attivazione (nel backbone e nella head) con `nn.ReLU`.
2. **DETECTION HEAD P2 (High-Res)**:
   - Il target sono coni piccoli a lunga distanza (10px).
   - Abilita output stride: 4 (P2), 8 (P3), 16 (P4).
   - RIMUOVI COMPLETAMENTE la testa P5 (stride 32) e i relativi layer di downsampling nel backbone per risparmiare parametri.
3. **RI-INGEGNERIZZAZIONE SPPF**:
   - L'implementazione standard di SPPF usa `torch.chunk` o slicing. Questo causa GPU Fallback.
   - Scrivi una classe `SPPF_DLA` in `dla_blocks.py`.
   - Implementazione richiesta: Usa una sequenza seriale di `MaxPool2d(kernel=5, stride=1, padding=2)`. Output = Concat([x, m1(x), m2(m1(x)), ...]). NON USARE SLICING.
4. **ATTENTION BLOCK**:
   - Se il blocco C2PSA o l'attenzione usano MatMul complessi o Softmax su assi dinamici, sostituiscili con un blocco `C3k2` standard o `Identity`.
5. **SCALING**:
   - Parti da YOLOv11n (Nano).
   - Imposta depth_multiple: 0.50, width_multiple: 0.25.

Genera il codice Python e YAML ora.
MODULO 2: Export & Calibrazione (TensorRT)Obiettivo: Compilare il modello garantendo INT8 e Zero Fallback.Prompt:Agisci come NVIDIA AI Deployment Specialist.
Scrivi uno script Python robusto `export_dla_engine.py` per convertire il modello PyTorch/ONNX in un TensorRT Engine (.engine) specifico per Jetson Orin.

SPECIFICHE DI COMPILAZIONE:

1. **TARGET DEVICE**:
   - Default Device Type: `trt.DeviceType.DLA`
   - DLA Core: 0
   - Consenti `GPU_FALLBACK`, MA lo script deve analizzare il `layer_info` finale e stampare un ERRORE ROSSO BLOCCANTE se qualsiasi layer diverso dall'input/output risulta assegnato alla GPU. Obiettivo: 100% DLA.

2. **PRECISIONE & CALIBRAZIONE**:
   - Abilita Flag: `FP16` e `INT8`.
   - Calibratore: USA ESCLUSIVAMENTE `trt.IInt8EntropyCalibrator2` (Cruciale per piccoli oggetti/coni). NON usare MinMax.
   - Implementa una classe `ConeCalibrationStream` che legge un batch di 50 immagini da una cartella `calib_imgs/`.
   - Salva la cache in `calibration.cache`.

3. **INPUT SHAPE**:
   - Shape statica fissata a: `(1, 3, 640, 640)`.
   - NESSUNA dimensione dinamica permessa (DLA non le supporta).

Lo script deve includere logging dettagliato e gestione degli errori per i path.
MODULO 3: Runtime ROS 2 Zero-Copy (C++)Obiettivo: Eliminare la latenza di trasporto dati tra Camera e DLA.Prompt:Agisci come Senior Robotics Software Engineer esperto in CUDA e ROS 2 (Humble/Jazzy).
Genera un nodo ROS 2 Lifecycle in C++ moderno (`vision_dla_inference`) per la Formula Student.

ARCHITETTURA ZERO-COPY OBBLIGATORIA:

1. **INPUT INTERFACE**:
   - NON sottoscriverti a `sensor_msgs::msg::Image` standard (troppe copie).
   - Definisci/Usa un messaggio custom o `TypeAdaptation` che trasporta un puntatore alla memoria condivisa (es. `NvBufSurface` pointer o Shared Memory Handle).
   - Assumi che il driver della camera scriva direttamente in memoria GPU/Unified.

2. **TENSORRT EXECUTION**:
   - Carica il file `.engine` generato nel modulo precedente.
   - Crea un `IExecutionContext`.
   - **BINDING**: Non usare `cudaMemcpy` per l'input. Passa direttamente il puntatore ricevuto dal messaggio ROS al `context->setTensorAddress()`. Il DLA deve leggere esattamente dove la camera ha scritto.

3. **OUTPUT**:
   - Esegui inferenza asincrona (`enqueueV3`).
   - Post-processa i tensori di output (decodifica YOLO) su CPU (o CUDA se efficiente) per estrarre Bounding Box.
   - Pubblica un messaggio `fsd_interfaces::msg::ConeDetections` (array di coni con x, y, conf, color, class).

4. **CODICE**:
   - Usa `rclcpp_lifecycle`.
   - Gestisci correttamente la memoria CUDA (RAII o smart pointers custom).
MODULO 4 (Opzionale): Domain Adaptation (CycleGAN)Obiettivo: Generare dati di training fotorealistici dal simulatore.Prompt:Agisci come Computer Vision Researcher.
Scrivi uno script di training PyTorch per una **CycleGAN** finalizzata al Domain Adaptation (Sim-to-Real) per la Formula Student.

OBIETTIVO:
Trasformare immagini renderizzate dal simulatore (Dominio A) in immagini fotorealistiche (Dominio B) mantenendo intatta la posizione geometrica dei coni.

REQUISITI:
1. **Generator Architecture**: ResNet-based generator (9 blocchi).
2. **Loss Function**:
   - Adversarial Loss (LSGAN).
   - **Cycle Consistency Loss** (Cruciale: peso lambda=10.0) per garantire che A -> B -> A restituisca l'immagine originale.
   - **Identity Loss** (peso lambda=5.0) per preservare i colori dei coni (Giallo/Blu).
3. **Data Loading**:
   - Carica immagini non appaiate da due cartelle `data/sim` e `data/real`.
   - Applica trasformazioni base (Resize 640x640, Normalize).

Fornisci il loop di training completo e una funzione `inference_transform.py` per convertire un intero dataset sintetico.
