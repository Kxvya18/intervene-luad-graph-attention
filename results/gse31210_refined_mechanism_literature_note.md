# Literature Note: Refined GSE31210 Relapse Mechanism

## Computational lead from this project

The refined GSE31210 analysis suggests that the original broad proteostasis/stress signal is more specific than we first thought.

Main computational findings:

- **MRPL21** is the strongest single relapse-associated marker in GSE31210: C-index **0.734**, HR high/low **20.47**, logrank p **2.20e-9**.
- **MRPL21 remains informative after removing proliferation**: residual C-index **0.683**, logrank p **3.01e-6**.
- **Proteasome core**, defined as `PSMA1 + PSMB5`, is strong: C-index **0.713**, HR **8.82**, logrank p **3.40e-8**.
- **Proteasome core remains significant after proliferation residualization**: C-index **0.635**, logrank p **0.00463**.
- **High proteasome + high mito/redox** patients have the worst relapse burden: relapse fraction **44.2%** versus **9.1%** in low-both patients; early relapse within 24 months **21/77** versus **1/77**.
- Added-value tests show **MRPL21 adds information after proliferation** and also after proteasome-core scoring.

Working hypothesis:

> A relapse-prone subset of GSE31210 lung adenocarcinoma may be marked by a dual stress-adaptation state: proteasome-core burden plus mitochondrial translation/redox adaptation, with MRPL21 as a sharper mitochondrial-translation lead than the broad mitochondrial module.

## Literature support and gaps

### 1. Proteasome burden in lung cancer / NSCLC

Cron et al. showed that proteasome inhibitors can block DNA repair and radiosensitize non-small cell lung cancer. This supports the idea that a proteasome-high relapse state may reflect stress tolerance and DNA-damage repair capacity, not merely a passive expression marker.

- Cron KR et al. **Proteasome inhibitors block DNA repair and radiosensitize non-small cell lung cancer.** PLoS One, 2013. PMID: 24040035. DOI: 10.1371/journal.pone.0073710. PubMed: https://pubmed.ncbi.nlm.nih.gov/24040035/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3764058/

A LUAD-focused study also proposed a ubiquitin-proteasome gene signature for prognosis. This supports the broader direction that UPS/proteasome state can carry prognostic signal in lung adenocarcinoma, although our specific split into proteasome-core versus VCP/UPR axes is our own refinement.

- **A Ubiquitin-Proteasome Gene Signature for Predicting Prognosis in Patients With Lung Adenocarcinoma.** Frontiers in Genetics, 2022. PMID: 35711913. DOI: 10.3389/fgene.2022.893511. PubMed: https://pubmed.ncbi.nlm.nih.gov/35711913/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9194557/

### 2. PSMA1 / PSMB5 relevance

PSMA1 has direct cancer-prognosis literature. One study reports PSMA1 as a poor prognostic factor in lung squamous cell carcinoma, and a pan-cancer analysis examined proteasome alpha subunits as prognostic markers. This supports PSMA1 as a plausible aggressive-cancer/proteasome marker, but it does **not** by itself prove LUAD relapse causation.

- **PSMA1, a Poor Prognostic Factor, Promotes Tumor Growth in Lung Squamous Cell Carcinoma.** Disease Markers, 2023. PMID: 36776923. DOI: 10.1155/2023/5386635. PubMed: https://pubmed.ncbi.nlm.nih.gov/36776923/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9918360/
- **The transcription levels and prognostic values of seven proteasome alpha subunits in human cancers.** Oncotarget, 2017. PMID: 27966459. DOI: 10.18632/oncotarget.13885. PubMed: https://pubmed.ncbi.nlm.nih.gov/27966459/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5354849/

PSMB5-specific NSCLC literature appears thinner than PSMA1 literature, but PSMB5 is the catalytic beta-5 proteasome subunit targeted by proteasome inhibitors. In our data, `PSMA1 + PSMB5` together works better as a **proteasome core burden** axis than as isolated biomarkers.

### 3. TXNRD1 / redox buffering and lung cancer stress vulnerability

Galicia-Vázquez et al. reported KEAP1-dependent synthetic lethality induced by AKT and TXNRD1 inhibitors in lung cancer. This is highly relevant because our GSE31210 model links redox buffering (`TXNRD1`) with AKT/mTOR and stress survival.

- Galicia-Vázquez G et al. **KEAP1-dependent synthetic lethality induced by AKT and TXNRD1 inhibitors in lung cancer.** Cancer Research, 2013. PMID: 23824739. DOI: 10.1158/0008-5472.CAN-13-0712. PubMed: https://pubmed.ncbi.nlm.nih.gov/23824739/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3868367/

Another Cancer Research paper showed that inhibiting thioredoxin/thioredoxin reductase can induce synthetic lethality in lung cancers with compromised glutathione homeostasis. This supports a redox-buffering vulnerability model rather than a generic redox-expression marker.

- **Inhibition of Thioredoxin/Thioredoxin Reductase Induces Synthetic Lethality in Lung Cancers with Compromised Glutathione Homeostasis.** Cancer Research, 2019. PMID: 30401714. DOI: 10.1158/0008-5472.CAN-18-1938. PubMed: https://pubmed.ncbi.nlm.nih.gov/30401714/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7293542/

### 4. Mitochondrial ribosomal proteins / MRPL21

Direct MRPL21 relapse literature in lung adenocarcinoma appears sparse based on the PubMed scan. That is important: MRPL21 may be a **novel lead** rather than a well-known relapse marker.

Broader literature supports mitochondrial ribosomal proteins as cancer prognosis/therapy-relevant biology, and mitochondrial stress response as a cancer survival mechanism.

- **Mitochondrial ribosomal proteins: potential targets for cancer prognosis and therapy.** Frontiers in Oncology, 2025. PMID: 40371222. DOI: 10.3389/fonc.2025.1586137. PubMed: https://pubmed.ncbi.nlm.nih.gov/40371222/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12074914/
- **Mitochondrial Stress Response and Cancer.** Trends in Cancer, 2020. PMID: 32451306. DOI: 10.1016/j.trecan.2020.04.009. PubMed: https://pubmed.ncbi.nlm.nih.gov/32451306/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7392807/

Interpretation: our MRPL21 result should be framed as **candidate novel relapse biology requiring validation**, not as already-established NSCLC relapse biology.

### 5. Proteasome + mitochondrial/redox co-dependency

The strongest computational combination is not “proteasome only” or “redox only.” The worse group is high proteasome core plus high mito/redox. Literature supports each side separately: proteasome inhibition affects DNA repair/radiosensitivity in NSCLC, while TXNRD1/redox targeting creates synthetic lethality in lung cancer contexts. The exact **proteasome-core + MRPL21/mito-redox relapse state** appears underexplored.

Paper-safe claim:

> GraphSparse Relapse-INTERVENE nominates a proteasome-core/mitochondrial-translation/redox relapse state in GSE31210, with MRPL21 and PSMA1/PSMB5 as the sharpest computational drivers. Existing literature supports proteasome and TXNRD1/redox vulnerabilities in lung cancer, while the MRPL21-centered relapse link appears less characterized and should be treated as a novel hypothesis for experimental validation.

Avoid overclaiming:

- Do not say we discovered the cause of relapse.
- Do not say MRPL21 is proven as a LUAD relapse biomarker.
- Do not say proteasome inhibitors are guaranteed therapy for this group.
- Do say this is a **computationally prioritized, literature-supported hypothesis**.
