# Installation guide

ManyWells can be installed as a package by running the following command:
```GIT_LFS_SKIP_SMUDGE=1 pip install git+ssh://git@github.com/solution-seeker-as/manywells.git```

This command requires that you have registered an SSH key in your GitHub account.

Note that the environment variable `GIT_LFS_SKIP_SMUDGE=1` is set to prevent git from cloning the large dataset files.

If you want to install from a specific branch, you must append the branch name:
```GIT_LFS_SKIP_SMUDGE=1 pip install git+ssh://git@github.com/solution-seeker-as/manywells.git@branch-name```

After having installed the environment, you can import the manywells package.
For example, you could import the simulator as `import manywells.simulator as sim`. 