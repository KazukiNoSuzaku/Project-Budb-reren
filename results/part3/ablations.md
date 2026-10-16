| variant | val acc @ 95% | val acc @ 98% |
|---|---:|---:|
| fisher (default) | 0.9815 ± 0.0013 | 0.9806 ± 0.0006 |
| fisher + regrowth 0.3 | 0.9817 ± 0.0013 | 0.9799 ± 0.0001 |
| fisher, single-batch scores (beta=0) | 0.9811 ± 0.0024 | 0.9712 ± 0.0020 |
| fisher, layer-wise allocation | 0.9817 ± 0.0018 | 0.9766 ± 0.0011 |
| fisher, one-shot at end of ramp | 0.9668 ± 0.0023 | 0.9349 ± 0.0066 |
| magnitude | 0.9836 ± 0.0006 | 0.9790 ± 0.0008 |
| magnitude, layer-wise allocation | 0.9818 ± 0.0006 | 0.9706 ± 0.0006 |
| magnitude + regrowth 0.3 | 0.9837 ± 0.0007 | 0.9819 ± 0.0007 |

Mean ± std over seeds [100, 101], validation split.
