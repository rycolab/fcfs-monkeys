# fcfs-monkeys

This code is used to generate random text with First-come First-serve Monkeys.

## Dependencies

To install dependencies run:
```bash
$ conda env create -f scripts/environment.yml
```

Then install the appropriate version of pytorch:
```bash
$ conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch -c conda-forge
$ # conda install pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
```
Finally, download WordNet from nltk. For this, open python in the terminal and use commands:
```python
$ import nltk
$ nltk.download('omw-1.4')
```

## Getting the data

Use [this Wikipedia tokenizer repository](https://github.com/tpimentelms/wiki-tokenizer) to get the data and move it into `data/<wiki-code>/parsed-wiki40b.txt` file.
