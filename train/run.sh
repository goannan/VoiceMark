#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# initial pyenv（关键）
# export PYENV_ROOT="$HOME/mrnas04home/.pyenv"
# [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init - bash)"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"

# pyenv activate venv_3.12

bash train_example.sh