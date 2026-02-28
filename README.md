# spliceai_bins_min.py

Minimal SpliceAI ensemble wrapper for bin-counting splice-site signal from a single input sequence in forward and reverse.

**Interface**
- **stdin:** one nucleotide sequence (whitespace ignored)
- **stdout:** **two lines**, each **7 integers**
	- line 1: SpliceAI on the input (“sense”)
	- line 2: SpliceAI on the **reversed** sequence ("anti-sense", NOT reverse-complement)
- **stderr:** hard-silenced at the OS file-descriptor level

---

## Installation / environment expectations

- Python 3.x
- `TensorFlow` + `Keras` Python packages installed.
	- If Keras is unavailable, the line:
	  `from keras.saving import load_model`
	  can be replaced with:
	  `from tensorflow.keras.models import load_model`
- `spliceai` Python package installed.
	- This relies on the default bundling of the models being at: `${spliceai_package_dir}/models/spliceai1.h5` … `spliceai5.h5`
	- I personally prefer [this fork](https://github.com/bw2/SpliceAI) of the package, but the models this script uses are identical regardless.

---

## Invocation patterns

Any mechanism that writes the sequence to **stdin** works.

### POSIX shells (bash/zsh)

**Pipe**
```bash
printf 'GATTACA\n' | python spliceai_bins_min.py
````

**Here-string**

```bash
python spliceai_bins_min.py <<< 'GATTACA'
```

**Redirect a file**

```bash
python spliceai_bins_min.py < seq.txt
```

### Windows PowerShell

**Pipe**

```powershell
'GATTACA' | python .\spliceai_bins_min.py
```

**Redirect a file**

```powershell
python .\spliceai_bins_min.py < seq.txt
```

---

## Output semantics

The underlying SpliceAI model produces per-base probabilities for 3 classes:

* `neither`, `acceptor`, `donor`

This script:

1. ensembles 5 models by arithmetic mean,
2. extracts **acceptor** and **donor** probabilities for each base,
3. pools them into one vector of length `2L` (**acceptor + donor**),
4. bins each probability into 7 intervals, and prints counts.

### Bin edges

| Order | Range           | Label                                         |
| ----- | --------------- | --------------------------------------------- |
| 1.    | `[0, 0.001)`    | "Not a site"                                  |
| 2.    | `[0.001, 0.05)` | "Unlikely site"                               |
| 3.    | `[0.05, 0.2)`   | "Insignificant site"                          |
| 4.    | `[0.2, 0.35)`   | "Weak-cryptic or tissue-specific site"        |
| 5.    | `[0.35, 0.5)`   | "Significant-cryptic or tissue-specific site" |
| 6.    | `[0.5, 0.8)`    | "Significant site"                            |
| 7.    | `[0.8, 1]`      | "Very strong site"                            |


---

## Constraints

* **Input must be just sequence.** The script uppercases and strips whitespace; anything else is treated as “unknown base.”
* **RNA & DNA supported:** everything except for A/C/G/T/U (including N and IUPAC ambiguity) becomes an all-zero channel vector (effectively “N”)
* **Fixed SpliceAI context:** hard-coded 10k padding (5k each flank), matching the standard SpliceAI-10k setting.
* **Counts are pooled:** acceptor and donor are **not** reported separately, only their combined bin counts.
* **Stderr is suppressed:** if TensorFlow throws, you may get a silent failure. For debugging, remove the `dup2` stderr redirect line.
* **Sequence length scaling:** runtime/memory scale with sequence length; long cassettes can OOM (Out of Memory) depending on device.
* **Model determinism is not strictly enforced:** for the sake of simplicity and runtime, the exact predicted strengths of sites can vary based on hardware.
	* However, the chance of this altering the values enough to shift the bin counts between runs is effectively zero.

---

## How the code works

### 1) Silence stderr at the process level

```py
d=os.open(os.devnull,os.O_WRONLY);os.dup2(d,2);os.close(d)
```

This redirects the **OS file descriptor 2** (stderr) to `/dev/null` *before* TensorFlow loads. That’s important because verbose TF/absl logs frequently bypass Python’s `sys.stderr`.

### 2) Load the 5 SpliceAI models once

```py
K=[load_model(spliceai.__path__[0]+f"/models/spliceai{i}.h5",compile=False)for i in range(1,6)]
```

* Uses Keras’ HDF5 loader.
* `compile=False` skips optimizer/loss state (irrelevant for inference).
* Models are kept in memory for both strand passes.

### 3) Encode the sequence as padded one-hot

```py
x=tf.one_hot([4]*5000+[{65:0,67:1,71:2,84:3,85:3}.get(c,4)for c in s.encode()]+[4]*5000,5)[:,:4][None]
```

* Maps ASCII bytes to indices: `A→0, C→1, G→2, T→3, U→3`, default `4` (unknown).
* Adds 5k unknown bases on each side (SpliceAI flank context).
* `tf.one_hot(...,5)` produces a 5-channel one-hot; slicing `[:,:4]` drops the “unknown” channel.
	* Result: unknown bases become `[0,0,0,0]`, i.e., no base evidence.
* Adds batch dimension with `[None]`: `shape = (1, L+10000, 4)`.

### 4) Run the ensemble of the models and average

```py
y=0
for k in K:
  o=k(x,training=False);o=o[0]if isinstance(o,(list,tuple))else o
  y+=o[0]
y/=5
```

* Some Keras exports wrap the output in a list; the `isinstance(...,(list,tuple))` unpacks that.
* `o[0]` removes the batch dimension, yielding `(L,3)`.
* Sum across 5 models and divide by 5 yields ensemble mean `y` with shape `(L,3)`.

### 5) Convert probabilities to a 7-bin histogram

```py
b=tf.searchsorted((.001,.05,.2,.35,.5,.8),tf.concat((y[:,1],y[:,2]),0),side='right')
tf.math.bincount(b,None,7)
```

* `y[:,1]` and `y[:,2]` are acceptor and donor probabilities across bases.
* Concatenation pools them: vector length `2L`.
* `searchsorted(..., side='right')` maps each value to an integer bin index 0–6.
* `bincount(..., minlength=7)` yields 7 counts.

### 6) Do it twice: sense + reverse-only

```py
print(*f(s));print(*f(s[::-1]))
```

Second pass reverses the base order and reruns the exact same pipeline.
