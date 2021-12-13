# cse392

This is mathew's final presentation project for course CSE 392 - Fa21 - SCIENTIF COMP MACH/DEEP LRN. 

It requires pytorch and FrEIA installed. pytorch can be install by conda, pip, or docker. FrEIA can be installed by pip.

## Build up (by docker)

```bash
cd <a directory you would like to run the project>
git pull https://github.com/mathewgaohu/cse392.git
cd cse392
docker pull pytorch/pytorch
docker run -it --name cse392-mathew -v $PWD:/workspace pytorch/pytorch
```

Then in the shell, run command

```
```

