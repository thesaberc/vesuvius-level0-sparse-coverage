# Preserve level-0 sparse CT coverage under native crop translation

Villa’s native sampler can translate a training crop by up to 40 voxels along Y and/or X. The level-0 sparse planner collects chunks for the original crop. When a retained shift crosses a chunk boundary, Zarr can fill an omitted chunk with zeros and silently change the sampled CT image.

The planner patch expands each retained level-0 crop by the existing translation bound before clipping and chunk collection. Z is unchanged. The envelope also applies when augmentation is disabled, so the cost is predictable for a given crop and can be unnecessary for a particular sample.

## Evidence

| Check | Baseline | With the envelope |
| --- | --- | --- |
| PHerc. 139 w035, native 9.362 µm, frozen retained −40 Y crop | Two required chunks missing, 4,096 voxel mismatches | All required chunks available. Image, ink labels, and supervision exactly match the complete local reference. |
| Same-window zero control | Exact reference match | Exact reference match |
| Targeted window plan | Four chunks, 8 MiB | Six chunks, 12 MiB. The six keys exactly cover the union of both reads. |
| Separate three-window cohort | Six nonzero proposals all fell back to zero. The baseline covered every observed read. | 16 to 24 deduplicated planned chunks, or 32 to 48 MiB logical. No CT bodies were acquired. The accepted-translation omission fraction is undefined. |
| Synthetic M1 and M2 | M1 reproduced 163,840 mismatches and exact restoration | M2 cached and serialized regressions verified exact image, label, and supervision results for original and translated samples. |

The real-scroll result establishes a correctness benefit for one targeted crop. The cohort measures planning overhead under one fixed schedule. Neither establishes frequency across scrolls, training accuracy, reading improvement, or adoption. [Machine-readable results](evidence/results.json)

## Patch and reproducibility

- [Executed historical patch](patches/00-executed-m2.diff) contains the planner change, focused regression, and one debug-label rename. The evidence is from this patch.
- [Minimal planner patch](patches/01-level0-planner.diff) and [focused test patch](patches/02-focused-regression.diff) separate the functional fix and regression. Static comparison against the executed patch verified the debug-only difference.
- [Bicubic compatibility patch](patches/03-bicubic-compatibility.diff) is a separate reader interoperability adjustment. Both real-scroll arms used it. It is independent of the coverage fix.
- [Reproduction instructions](REPRODUCE.md) bind the runtime, source and offline input requirements. [Shareable inventory](SHARE-MANIFEST.json) enumerates the contents and hashes.

The historical run used Villa revision `3fbb3bbadb84fba3648a0c5c7cdbf7d767f9a3c0`. The three review patches also passed a file-level static application check on the planner and geometry preimages from Villa `c86cbd14bb34e0124684d3d97b03b310c947819a`. No scientific replay ran on that newer revision. [Villa](https://github.com/ScrollPrize/villa/tree/3fbb3bbadb84fba3648a0c5c7cdbf7d767f9a3c0) · [Static-check revision](https://github.com/ScrollPrize/villa/commit/c86cbd14bb34e0124684d3d97b03b310c947819a)

Data attribution: Giorgio Angelotti et al., *Vesuvius Challenge — CT Scans of Herculaneum Papyri*. PHerc. 139 w035 uses official volume `20250728140407`. The input release is CC BY-NC 4.0. This artifact contains code, aggregate evidence and object identities. It contains no raw CT, labels, masks, images or hidden text. The [Villa license](LICENSE) is retained. [PHerc. 139 release](https://scrollprize.org/data_browser/PHerc0139) · [Dataset citation and licensing](https://scrollprize.org/data#licenses)
