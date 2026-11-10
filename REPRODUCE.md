# Reproduction

The offline wrapper accepts an existing Linux checkout at Villa revision `3fbb3bbadb84fba3648a0c5c7cdbf7d767f9a3c0`. It checks source blobs and stages to a new scratch directory without changing the checkout or fetching dependencies. The exact executed patch is the default. `--minimal` selects the separately static-checked split patch. The public wrapper is a portable adaptation of the local workflow. Its staging is checked independently. Historical execution receipts come from the original controlled launchers.

Use an existing CPU Python 3.14 environment with pytest 9.2.0, NumPy 2.5.4, Zarr 2.18.7, numcodecs 0.15.2, Torch 2.12.0+cpu, tqdm 4.67.1, and vc-reader 0.1.0 built from the pinned Villa tree. The historical environment used Python 3.14.4 and an SSE2-compatible Blosc build. Pass a compatible local codec overlay if your environment needs one. No dependency or input download is part of the wrapper. The wrapper does not create a network namespace or enforce host resource quotas. Use it inside an isolated, read-only-source environment and an empty bounded scratch directory.

```bash
sha256sum -c SHA256SUMS
checkout=/path/to/pinned/villa
runtime=/path/to/qualified-venv/bin/python
mkdir -p scratch-m1 scratch-focused scratch-serialized
"$runtime" -I -B reproduce.py m1 --checkout "$checkout" --python "$runtime" --scratch "$PWD/scratch-m1"
"$runtime" -I -B reproduce.py focused --checkout "$checkout" --python "$runtime" --scratch "$PWD/scratch-focused"
"$runtime" -I -B reproduce.py serialized --checkout "$checkout" --python "$runtime" --scratch "$PWD/scratch-serialized"
```

Use `--stage-only` with a fresh empty scratch directory to verify source and patch staging without importing a test. Each executed case writes `junit.xml` under its scratch directory.

## Real targeted comparison

With permission to use the official data, independently stage the 12 exact auxiliary objects from [the input list](w035/w035-stage1-assets-draft.json) under one local directory preserving their full keys. Stage the six native CT chunks under a second directory using relative `Z/Y/X` paths from [the reference manifest](w035/reference-manifest.json). All expected lengths and SHA-256 identities are supplied. An absent CT file is an error. Keep both input roots read-only. The included [preflight handoff](w035/preflight-witness.json) is a projection of the actual receipt needed to enforce matching frozen boxes and plan/read keys. It has no machine-specific information. The comparison revalidates the same manifest against the pinned sampler. No raw assets are distributed here.

```bash
mkdir -p scratch-w035
"$runtime" -I -B reproduce.py w035 --checkout "$checkout" --python "$runtime" \
  --scratch "$PWD/scratch-w035" --auxiliary-root /path/to/pinned-aux \
  --ct-root /path/to/pinned-level0-chunks
```

The comparison should report a −40 Y realized shift, omissions at `0/28/17/25` and `0/28/17/26`, 4,096 baseline mismatches and zero patched mismatches. Both arms should match the zero control. The cohort used zero CT placeholders only. Its [aggregate receipt](evidence/results.json) records plan and read counts. It cannot be used to verify voxel equality.
