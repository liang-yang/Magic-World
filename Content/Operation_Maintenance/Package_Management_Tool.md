<!-- toc -->

# Package Management Tool

原始的Linux程序为二进制的格式，包含：

- 二进制程序，位于 /bin, /sbin, /usr/bin, /usr/sbin, /usr/local/bin, /usr/local/sbin 等目录中；
- 库文件，位于 /lib, /usr/lib, /usr/local/lib 等目录中。Linux中库文件以 .so（动态链接库）或 .a（静态链接库）作为文件后缀名；
- 配置文件，位于 /etc 目录中；
- 帮助文件：手册, README, INSTALL (/usr/share/doc/)；

在没有软件包管理器之前，安装程序，卸载程序是非常繁杂的。 因此，大多数Linux操作系统都提供了 包管理系统，以一种中心化的机制来管理软件包。

虽然这些包管理系统的功能和优点大致相同，但打包格式和工具却因操作系统平台而异：

操作系统 | 软件包格式 | 工具
:-: | :-: | :-:
Debian/Ubuntu | .deb | dpkg, apt, apt-get, apt-cache
CentOS | .rpm | yum
Fedora | .rpm | dnf
FreeBSD | Ports, .txz | make, pkg

这里重点分析 RPM、YUM。

## 1. RPM

RPM早期被称为RedHat Package Manager，但由于目前RPM非常流行，且已经成为Linux工业标准。 但是，Linux中的程序大多是小程序，程序与程序之间存在非常复杂的依赖关系，**RPM无法解决软件包的依赖关系**。

### 1.1. RPM Package

用RPM工具可以将二进制程序进行打包，包被称为RPM包。  
RPM包并不是跨平台的，RedHat的RPM包与SUSE的RPM包不能混用。

- 二进制程序的命名规范：name-version.tar.{gz|bz2|xz}
    
    > eg. bash-4.3.1.tar.xz

- RPM包的命名规范：name-version-release.os.arch.rpm

    > eg. bash-4.3.2-5.el6.x86_64.rpm

### 1.2. RPM Command

Reference:  
-- [All you have to know about RPM](http://fedoranews.org/alex/tutorial/rpm/)

Scene | Command | Description
:-: | :-: | :-:
Install | rpm -ivh package.rpm | 安装软件包，-i(--install)表示安装操作，-v(--verbose)表示命令过程可视化，-h表示以"#"号显示安装进度
Upgrade | rpm -Uvh new-package.rpm | 升级软件包。若尚未安装，则安装。-U(--Upgrade)
Erase | rpm -e package | 卸载软件包，-e(--erase)
Erase | rpm -e --test package | 虚拟卸载过程，以测试是否存在问题
Erase | rpm -e --nodeps package | 卸载软件包，忽略依赖关系
Erase | rpm -e --force package | 卸载软件包，忽略软件包及文件的冲突
Query | rpm -qa | 所有已安装的软件包，-a(--all)
Query | rpm -q package | 查询某软件包是否安装
Query | rpm -qi package | 查询某软件包（已安装）的详细信息，-i(--info)
Query | rpm -q --whatrequires package | 查询依赖此软件包的其他软件包 
Query | rpm -qpi package.rpm | 查询某安装文件的详细信息，-p表示后续参数视作安装文件，而非软件包，故可处理未安装软件包
Query | rpm -qR package | 软件包（已安装）需要的软件包和应用程序 
Query | rpm -qpR package.rpm | 安装文件（可未安装）需要的软件包和应用程序
Query | rpm -qf file | 查询某文件是哪个软件包生成的，-f(--file)
Query | rpm -ql package | 列出某软件包中所包含的文件，-l(--list)

### 1.3. RPM 数据库

RPM软件包管理器内部有一个数据库，其中记载着程序的基本信息，校验信息，程序路径信息等。

RPM数据库文件位于：/var/lib/rpm  

若库损坏，很多RPM的查询将无法使用。对损坏的数据库，可以进行数据库重建：

> rpm --initdb #新建数据库   
> rpm --rebuilddb #重建数据库

## 2. YUM

YUM 被称为 Yellow dog Updater, Modified，是一个在Fedora和RedHat以及SUSE中的Shell前端软件包管理器。 YUM 使用Python开发，可以通过HTTP服务器下载、FTP服务器下载、本地软件池的等方式获得软件包，并自动安装。   
**YUM 可以自动处理依赖性关系**。

### 2.1. YUM Repo

YUM 仓库又称为YUM源，主要用于存储软件包，其配置主要在：

> /etc/yum.conf  #主配置     
> /etc/yum.repos.d/*.repo  #片段配置

/etc/yum.conf 中配置了一个特殊的main仓库，为其他仓库提供默认的全局配置。

仓库的关键配置项有：

> **[ ... ]**：仓库的标识，不能重复。    
> **name**：仓库的名称，该项必须有。    
> **baseurl**：配置仓库的路径，用于指定一个url。    
> **mirrorlist**：指向一个镜像列表，里面有多个url。    

其中，baseurl 支持 ftp协议(ftp://)，http协议(http: //)，文件协议(file://)。例如，对于RHEL系列的Linux，其安装盘就是一个yum仓库，因此可通过文件协议将yum仓库指向光盘路径。

另外，配置中存在一些内置变量，可用于动态的配置yum路径。

> $releasever：当前操作系统的主版本号。若CentOS6.4 该值为6。    
> $arch：当前平台版本架构。x86_64 或 i386/i586/i686。     
> $basearch：当前平台的基本架构。x86_64 或 i386。     


### 2.2. YUM Command

yum的命令形式一般是如下：_yum [options] [subcommand] [package ...]_，其中 _package_ 可通过通配符指定包名。

- **yum list** [subcommand] [package ...]

    查询软件包信息，subcommand 可取值：
    
    > all: 所有的包   
    > installed: 已安装的包     
    > available: 没有安装，但可安装的包     
    > updates: 可更新的包     
    > extras: 不属于任何仓库，额外的包     
    > obsoletes: 废弃的包     
    > recent: 新添加进yum仓库的包     

    例如，可通过 yum list installed \*mysql\* 查询已安装的包含mysql的软件包。   

    查询结果说明：
    
    1. 第一列：软件包名称.平台名称；
    2. 第二列：软件版本号-release号；
    3. 第三列：安装情况：若显示@则表示该软件已通过该仓库安装；install，则表示系统已安装，但未通过仓库安装；若无@或不是install，则表示尚未安装，此列显示仓库名。


- **yum repolist** [subcommand]

    查询yum仓库，subcommand 可取值：
    
    > all: 所有的仓库     
    > enabled: 启用的仓库     
    > disabled: 禁用的仓库     

- **yum info** [package ...]：显示软件包的信息，类似于rpm -qi；
- **yum install** [-y] [package ...]：安装软件包；
- **yum update** [package ...]：更新已安装的包； 
- **yum check-update**：检测可升级的包；
- **yum remove** [package ...]：卸载软件包；

## 3. 三方仓库

为了稳定，官方的rpm repository提供的rpm包往往是很滞后的。同时，官方的rpm repository提供的rpm包也不够丰富，因此我们需要引入三方库。

### 3.1. MySQL

以安装mysql为例导入三方仓库。在操作系统自带的仓库中没有mysql软件包，因此以安装软件包的方式导入仓库：

> rpm -ivh https://dev.mysql.com/get/mysql80-community-release-el7-3.noarch.rpm

安装后查询：

> yum list all \*mysql80\*

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8859g6hucj30j8020glm.jpg)

可以看出，我们已安装了软件包 mysql80-community-release.noarch。再查询 mysql 相关的仓库：

> yum repolist all \*mysql\*

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g885gqz5sgj30ja09wta8.jpg)

可以看出，新增了不少的 mysql 相关的仓库。这里，我们想安装 MySQL 5.7 版本，因此，禁用 8.0版本仓库，启用5.7版本仓库。

> yum-config-manager --disable mysql80-community    
> yum-config-manager --enable mysql57-community

这样，我们就可以安装 5.7 版本的 MySQL 数据库了。

> yum install -y mysql-community-server

### 3.2. EPEL

Reference：   
-- [EPEL](https://fedoraproject.org/wiki/EPEL/zh-cn)

EPEL（Extra Packages for Enterprise Linux）是由 Fedora 社区打造，为“红帽系”的操作系统提供额外的软件包，适用于RHEL、CentOS和Scientific Linux。

在 CentOS 中，可直接通过安装 epel-release 软件包的形式安装：

> yum install epel-release

或者通过网络安装：

> rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

安装前，没有EPEL相关的仓库：

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g885x6xzh3j30fk01owef.jpg)

安装后，存在EPEL相关的仓库：

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g885y07zu5j30j904g3z3.jpg)


## Reference

- [Linux 包管理基础：apt、yum、dnf 和 pkg](https://zhuanlan.zhihu.com/p/28562152)
- [Linux软件安装中RPM与YUM 区别和联系](https://www.cnblogs.com/LiuChunfu/p/8052890.html)
