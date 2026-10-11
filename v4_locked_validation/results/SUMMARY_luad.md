# v4 locked results (luad)

## H1 nested internal (GSE31210, pool 52 module genes, 5x50)
mean C = 0.738 [0.701, 0.754]  -> PASS=True
selection frequency of S4 genes: MRPL21 1.00, PSMA1 0.99, PSMB5 0.97, TXNRD1 0.00
top-8 selected: MRPL21 1.00, PSMA1 0.99, PSMB5 0.97, SCD 0.79, MRPL10 0.12, BIRC2 0.06, VCP 0.02, MRPL45 0.02

## Per-cohort (S4 = MRPL21+PSMA1+PSMB5+TXNRD1; P13 = proliferation)
```
  cohort   n  events                  s4_genes  c_S4  c_S4_lo  c_S4_hi  c_P13  hr_S4  p_S4  loghr_S4_adj  p_S4_adj  p_P13_adj
GSE31210 204      54 MRPL21;PSMA1;PSMB5;TXNRD1 0.750    0.699    0.802  0.732  2.026 0.000         0.528     0.002      0.093
GSE30219  83      27 MRPL21;PSMA1;PSMB5;TXNRD1 0.665    0.561    0.755  0.773  1.562 0.016        -0.072     0.753      0.001
GSE37745  53      26 MRPL21;PSMA1;PSMB5;TXNRD1 0.530    0.400    0.644  0.617  1.054 0.793        -0.101     0.657      0.211
GSE50081 125      37 MRPL21;PSMA1;PSMB5;TXNRD1 0.512    0.405    0.612  0.601  0.969 0.859        -0.244     0.230      0.052
GSE68465 333     178        PSMA1;PSMB5;TXNRD1 0.562    0.514    0.608  0.601  1.110 0.158        -0.006     0.944      0.009
```
## H2 pooled external C(S4) (random effects) = 0.569 [0.511, 0.626], tau2=0.0015  -> PASS=True
## H3 pooled S4 log-HR given P13: fixed -0.050 (SE 0.071, p=0.4764); random -0.050 (SE 0.071)  -> PASS=False
## H4 random 4-gene signatures (n=1000 from 12499 genes in all cohorts)
S4 mean external C = 0.567: percentile 89.7 (random median 0.510, 95th 0.582)
S4 pooled adjusted log-HR = -0.050: percentile 20.3 (random median 0.024, 95th 0.150)
  -> PASS=False
## H5 stage-adjusted (secondary)
```
  cohort  n_stage  loghr_S4_given_stage  p_S4_given_stage  loghr_S4_given_stage_P13  p_S4_given_stage_P13
GSE31210      204                0.5956            0.0000                    0.3828                0.0307
GSE30219       83                0.2918            0.1635                   -0.1469                0.5516
GSE37745       53                0.0532            0.7955                   -0.0617                0.7927
GSE50081      125               -0.1096            0.5375                   -0.2740                0.1627
GSE68465      331                0.0830            0.2654                    0.0086                0.9194
```
