# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

module unload gcc/6.5.0
module load julia/1.12.4
module load gcc/10.2.0
module load python/3.9.2
module load cmake/3.13.0
module load boost/1.68.0

export PYTHONPATH=/home/mmaso/Kratos/bin/Release:$PYTHONPATH
export LD_LIBRARY_PATH=/home/mmaso/Kratos/bin/Release/libs:$LD_LIBRARY_PATH
