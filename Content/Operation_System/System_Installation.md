<!-- toc -->

# System Installation

## 1. System Installation

### 1.1. Add Swap File

> Reference:     
> -- [Create a Linux Swap File](https://linuxize.com/post/create-a-linux-swap-file/)

{%ace edit=true, lang='python'%}
mkdir -p /magic
touch /magic/swapfile
fallocate -l 8G /magic/swapfile
chmod 600 /magic/swapfile
mkswap /magic/swapfile
swapon /magic/swapfile #enable the swap
cat /proc/sys/vm/swappiness
sysctl vm.swappiness=80
{%endace%}

## 2. Linux Command

> whereis httpd   
> man rpm








