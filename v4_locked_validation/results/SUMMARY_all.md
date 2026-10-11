# v4 locked results (all)

## Per-cohort (S4 = MRPL21+PSMA1+PSMB5+TXNRD1; P13 = proliferation)
```
  cohort   n  events                  s4_genes  c_S4  c_S4_lo  c_S4_hi  c_P13  hr_S4  p_S4  loghr_S4_adj  p_S4_adj  p_P13_adj
GSE31210 204      54 MRPL21;PSMA1;PSMB5;TXNRD1 0.750    0.695    0.800  0.732  2.026 0.000         0.528     0.002      0.093
GSE30219 274     114 MRPL21;PSMA1;PSMB5;TXNRD1 0.627    0.570    0.681  0.679  1.484 0.000         0.057     0.612      0.000
GSE37745  96      49 MRPL21;PSMA1;PSMB5;TXNRD1 0.564    0.482    0.654  0.607  1.187 0.235         0.104     0.534      0.439
GSE50081 177      51 MRPL21;PSMA1;PSMB5;TXNRD1 0.512    0.433    0.595  0.547  1.001 0.992        -0.100     0.569      0.325
GSE68465 333     178        PSMA1;PSMB5;TXNRD1 0.562    0.513    0.609  0.601  1.110 0.158        -0.006     0.944      0.009
```
## H2 pooled external C(S4) (random effects) = 0.572 [0.526, 0.619], tau2=0.0011  -> PASS=True
## H3 pooled S4 log-HR given P13: fixed 0.014 (SE 0.059, p=0.8073); random 0.014 (SE 0.059)  -> PASS=False
## H4 random 4-gene signatures (n=1000 from 12499 genes in all cohorts)
S4 mean external C = 0.566: percentile 98.0 (random median 0.503, 95th 0.557)
S4 pooled adjusted log-HR = 0.014: percentile 51.0 (random median 0.013, 95th 0.123)
  -> PASS=False
## H5 stage-adjusted (secondary)
```
  cohort  n_stage  loghr_S4_given_stage  p_S4_given_stage  loghr_S4_given_stage_P13  p_S4_given_stage_P13
GSE31210      204                0.5956            0.0000                    0.3828                0.0307
GSE30219      269                0.3039            0.0017                    0.0686                0.5423
GSE37745       96                0.1650            0.2598                    0.1143                0.4974
GSE50081      177               -0.0560            0.7042                   -0.1323                0.4528
GSE68465      331                0.0830            0.2654                    0.0086                0.9194
```
