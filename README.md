# cse392

This is mathew's final presentation project for course CSE 392 - Fa21 - SCIENTIF COMP MACH/DEEP LRN. 

It requires pytorch and FrEIA installed. pytorch can be install by conda, pip, or docker. FrEIA can be installed by pip.

## Build up (by conda)

```bash
cd <a directory you would like to run the project>
git clone https://github.com/mathewgaohu/cse392.git
cd cse392
conda create -n pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib tqdm
pip install git+https://github.com/VLL-HD/FrEIA.git
python ml.py
```

It takes about 15min for training. 

## Clean up

```bash
conda remove --name pytorch --all
```



