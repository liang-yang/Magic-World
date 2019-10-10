<!-- toc -->

# Package Management Tool

大多数类 Unix 操作系统都提供了一种 **中心化** 的机制用来搜索和安装软件。软件通常都是存放在存储库中，并通过软件包的形式进行管理。

大多数包管理系统都是围绕软件包的集合构建的。软件包通常是一个存档文件，它包含已编译的二进制文件和软件的其他资源，以及安装脚本。软件包同时也包含有价值的元数据，包括它们的依赖项，以及安装和运行它们所需的其他软件包的列表。

虽然这些包管理系统的功能和优点大致相同，但打包格式和工具却因操作系统平台而异：

操作系统 | 软件包格式 | 工具
:-: | :-: | :-:
Debian/Ubuntu | .deb | dpkg, apt, apt-get, apt-cache
CentOS | .rpm | yum
Fedora | .rpm | dnf
FreeBSD | Ports, .txz | make, pkg




## RPM

Reference:  
-- [All you have to know about RPM](http://fedoranews.org/alex/tutorial/rpm/)


> // 查询软件包所安装的相关文件
> rpm -ql httpd
> 

parameter: 

- [-i|--install]: 安装软件包。
- [-v|--verbose]: 安装过程可视化。
- [-h|--hash]: 显示安装进度。
- [-U|--upgrade]: 升级软件包。
- [-e|--erase]: 删除软件包。可通过添加 [--test] 模拟删除操作，并通过 [--repackage]回滚到删除前。
-
-
-
- 
-
-
-
-


## YUM

> // 更新包列表（大多数系统在本地都会有一个和远程存储库对应的包数据库，在安装或升级包之前最好更新一下这个数据库。）   
> yum check-update   
> // 更新已安装的包   
> yum update   
> // 搜索某个包   
> yum search httpd   
> // 查看某个软件包的信息   
> yum info httpd   
> //删除一个或多个已安装的包   
> yum remove httpd   






EPEL


## Reference

- [Linux 包管理基础：apt、yum、dnf 和 pkg](https://zhuanlan.zhihu.com/p/28562152)


