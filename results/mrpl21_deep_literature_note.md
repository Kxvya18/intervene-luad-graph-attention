# MRPL21-Focused Literature Note

## Why MRPL21 is interesting in our analysis

In GSE31210, `MRPL21` is the strongest single relapse-associated gene found after INTERVENE-guided module refinement.

Project result:

- MRPL21 C-index in GSE31210: **0.734**.
- MRPL21 high/low hazard ratio approximation: **20.47**.
- MRPL21 logrank p: **2.20e-9**.
- MRPL21 after proliferation residualization: C-index **0.683**, logrank p **3.01e-6**.
- MRPL21 adds value after proliferation and after proteasome-core scoring.

Interpretation:

> MRPL21 is not merely a generic proliferation marker in GSE31210. It behaves like a mitochondrial-translation relapse lead discovered downstream of INTERVENE attention-guided module refinement.

## PubMed scan summary

A direct PubMed query for `MRPL21` returned a small literature set. MRPL21 has limited direct lung cancer/relapse literature, which supports novelty but also means we must avoid overclaiming.

### Lung adenocarcinoma / lung cancer

A 2026 paper specifically studies MRPL21 in pan-cancer and lung adenocarcinoma progression. This is highly relevant for background, but it is not the same as our relapse-specific GSE31210 analysis.

- **Integrative Analysis of the Role of MRPL21 in Human Pan-Cancer and Its Relationship with the Progression of Lung Adenocarcinoma.** Molecular Biotechnology, 2026. PMID: 39775548. DOI: 10.1007/s12033-024-01348-7. PubMed: https://pubmed.ncbi.nlm.nih.gov/39775548/

A lung squamous cell carcinoma bioinformatics study also appears in the MRPL21/lung search space, but it concerns metastatic progression prediction in LUSC, not LUAD relapse.

- **Genes whose expressions in the primary lung squamous cell carcinoma are able to accurately predict the progression of metastasis through lymphatic system, inferred from a bioinformatics analyses.** Scientific Reports, 2023. PMID: 37185598. DOI: 10.1038/s41598-023-33897-3. PubMed: https://pubmed.ncbi.nlm.nih.gov/37185598/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10130036/

### Therapy resistance

A 2025 experimental cancer paper reports an MRPL21-PARP1 axis promoting cisplatin resistance in head and neck squamous cell carcinoma through PI3K/AKT/mTOR and autophagy-related biology. This supports the idea that MRPL21 can participate in therapy-resistance biology, but it is not NSCLC relapse evidence.

- **MRPL21-PARP1 axis promotes cisplatin resistance in head and neck squamous cell carcinoma by inhibiting autophagy through the PI3K/AKT/mTOR signaling pathway.** Journal of Experimental & Clinical Cancer Research, 2025. PMID: 40713706. DOI: 10.1186/s13046-025-03482-9. PubMed: https://pubmed.ncbi.nlm.nih.gov/40713706/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12297673/

### Broader mitochondrial ribosomal protein context

A review supports mitochondrial ribosomal proteins as cancer prognosis and therapy-relevant targets, giving biological plausibility to MRPL21 as part of a mitochondrial translation axis.

- **Mitochondrial ribosomal proteins: potential targets for cancer prognosis and therapy.** Frontiers in Oncology, 2025. PMID: 40371222. DOI: 10.3389/fonc.2025.1586137. PubMed: https://pubmed.ncbi.nlm.nih.gov/40371222/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12074914/

A mitochondrial stress-response review supports the broader idea that mitochondrial stress adaptation can shape cancer survival.

- **Mitochondrial Stress Response and Cancer.** Trends in Cancer, 2020. PMID: 32451306. DOI: 10.1016/j.trecan.2020.04.009. PubMed: https://pubmed.ncbi.nlm.nih.gov/32451306/ PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7392807/

## What we can claim

Strong claim:

> INTERVENE nominates MRPL21 as a relapse-associated mitochondrial-translation lead in GSE31210, and this signal remains partly independent of proliferation.

Moderate claim:

> External validation is partial: MRPL21 validates directionally and significantly in GSE30219 but not strongly in GSE37745, and GSE68465 cannot directly test MRPL21 with the current GPL96 mapping.

Novelty claim:

> MRPL21 has emerging cancer and LUAD progression literature, but its role in lung adenocarcinoma relapse appears under-characterized. This makes it a plausible novel lead rather than established relapse biology.

Do not claim:

- MRPL21 causes relapse.
- MRPL21 is a clinically validated biomarker.
- MRPL21 is universal across all LUAD cohorts.
- MRPL21-directed therapy is proven.

Best final phrasing:

> GraphSparse Relapse-INTERVENE identifies an MRPL21-linked mitochondrial-translation/proteasome-core relapse state in GSE31210. External cohorts provide partial support and important falsification boundaries, positioning MRPL21 as a candidate relapse-biology lead for experimental validation rather than a completed biomarker discovery.
