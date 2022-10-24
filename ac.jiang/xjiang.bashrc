# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ac.jiang/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ac.jiang/anaconda/etc/profile.d/conda.sh" ]; then
        . "/home/ac.jiang/anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/home/ac.jiang/anaconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

module load gcc/7.1.0-bgguyp

export PETSC_DIR=/home/ac.jiang/petsc

export SLEPC_DIR=/home/ac.jiang/slepc

export PETSC_ARCH=gcc-7.1_complex_impi_mkl_int64_sprng_v3.13.4
