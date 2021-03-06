<!-- toc -->

# Software Environment

## 1. MySQL

> Reference:    
> -- [A Quick Guide to Using the MySQL Yum Repository](https://dev.mysql.com/doc/mysql-yum-repo-quick-guide/en/)

安装 5.7 版本 MySQL，并使用临时密码登录

{%ace edit=true, lang='python'%}
rpm -ivh https://dev.mysql.com/get/mysql80-community-release-el7-3.noarch.rpm #安装MySQL软件包仓库
yum-config-manager --disable mysql80-community #关闭mysql80仓库
yum-config-manager --enable mysql57-community #启用mysql57仓库
yum install -y mysql-community-server
echo 'event_scheduler=1'>>/etc/my.cnf #开启event_scheduler功能
systemctl start mysqld #启动MySQL
systemctl enable mysqld #开机自启
mysql -uroot -p$(awk '/temporary password/{print $NF}' /var/log/mysqld.log) #使用临时密码登录数据库
{%endace%}

修改root用户密码及新增远程登录用户
{%ace edit=true, lang='python'%}
mysql> ALTER USER 'root'@'localhost' IDENTIFIED BY 'Y!a4364sdfsdfsdf'; #修改root用户密码
mysql> CREATE USER 'admin'@'%' IDENTIFIED BY 'W!b342f6dsfds3'; #创建可远程登录的admin用户
mysql> GRANT ALL ON *.* TO 'admin'@'%' WITH GRANT OPTION; #对admin用户对所有库和表赋权
mysql> exit
{%endace%}

## 2. Git

> Reference:    
> -- [Installing Git](https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-%E5%AE%89%E8%A3%85-Git)

在 CentOS 上可以直接使用 yum 来安装 Git 二进制程序：

{%ace edit=true, lang='python'%}
yum install -y git
{%endace%}

但 yum 安装的版本一般都比较旧。如果想获得最新版本，可以通过源码安装：

{%ace edit=true, lang='python'%}
yum install -y epel-release libtool gcc curl-devel expat-devel gettext-devel openssl-devel zlib-devel asciidoc xmlto docbook2X #安装依赖包
ln -s /usr/bin/db2x_docbook2texi /usr/bin/docbook2x-texi #此命令名称有更改，需要设置软连接

mkdir -p /magic/download
cd /magic/download
wget -c https://github.com/git/git/archive/v2.23.0.tar.gz -O git-2.23.0.tar.gz -e use_proxy=yes -e http_proxy=127.0.0.1:1087
tar -zxf git-2.23.0.tar.gz
cd git-2.23.0
make configure
./configure --prefix=/usr
make all doc info
make install install-doc install-html install-info
{%endace%}

## 3. Node.js & NPM & GitBook

在 CentOS 上，如果已安装 epel-release，则可以通过 yum 直接安装 Node.js 和 NPM（安装Node.js会默认安装NPM），然后再通过 NPM 安装 GitBook：

{%ace edit=true, lang='python'%}
yum install -y nodejs
npm install -g gitbook-cli #安装gitbook
gitbook -V
{%endace%}

当然，此版本一般比较低，我们也可以从 http://nodejs.cn/download/ 查询最新版本的 Node.js 和 NPM 安装包：

{%ace edit=true, lang='python'%}
mkdir -p /magic/download
cd /magic/download
wget -c https://npm.taobao.org/mirrors/node/v12.13.0/node-v12.13.0-linux-x64.tar.xz
tar -xf node-v12.13.0-linux-x64.tar.xz
ln -s /magic/download/node-v12.13.0-linux-x64/bin/node /usr/local/bin/
ln -s /magic/download/node-v12.13.0-linux-x64/bin/npm /usr/local/bin/
npm install -g gitbook-cli #安装gitbook
ln -s /magic/download/node-v12.13.0-linux-x64/bin/gitbook /usr/local/bin/
gitbook -V
{%endace%}

不过，通过解压安装包的方式安装，需要设置软连接使 node、npm、gitbook 成为全局指令。

## 4. Miniconda & Python

> Reference:    
> -- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

{%ace edit=true, lang='python'%}
mkdir -p /magic/download
cd /magic/download
wget -c https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh
bash Miniconda3-4.6.14-Linux-x86_64.sh
source ~/.bashrc
conda -V
conda update -y python
conda install -y numpy pandas scikit-learn matplotlib pymysql statsmodels lxml
{%endace%}

## 5. FTP

> Reference:    
> -- [Centos搭建FTP服务](https://cloud.baidu.com/doc/BCC/s/ljxlpwkwv/)    
> -- [FTP的主动模式和被动模式](https://cn.bluehost.com/blog/zsk/hosting/6459.html)    

我们通过 vsftpd 提供FTP服务。

FTP 一般使用 命令（默认21） 和 数据（默认20） 两个端口。
在主动模式下，FTP客户端需要提前提供一个随机（大于1023）的端口作为数据接收端口，并告知FTP服务端。当需要传输数据时，由FTP服务端主动发起连接。  
在被动模式下，FTP服务端需要提前提供一个随机（大于1023）的端口作为数据接收端口，并告知FTP客户端。当需要传输数据时，由FTP客户端主动发起连接。  
FTP的被动模式是不安全的，因为在服务器上打开了一个随机端口，这是一个潜在的安全问题，因此不建议使用FTP的被动模式。

{%ace edit=true, lang='python'%}
useradd ftp_user
passwd ftp_user # bjfhsd421gf!disf
mkdir -p /magic/ftp
chown -R ftp_user:ftp_user /magic/ftp #新建登录用户及FTP使用的目录

yum install -y vsftpd
mv /etc/vsftpd/vsftpd.conf /etc/vsftpd/vsftpd.conf.default
touch /etc/vsftpd/vsftpd.conf
vim /etc/vsftpd/vsftpd.conf #添加下面的配置
touch /etc/vsftpd/chroot_list
systemctl start vsftpd.service
systemctl enable vsftpd.service
{%endace%}

    anonymous_enable=NO #禁止匿名登录FTP服务器    
    local_enable=YES #允许本地用户登录FTP服务器    
    write_enable=YES #允许登录FTP服务器的用户写权限    
    local_root=/magic/ftp #设置本地用户登录后所在的目录    
    chroot_local_user=YES #全部用户被限制在主目录    
    chroot_list_enable=YES #启用例外用户名单    
    chroot_list_file=/etc/vsftpd/chroot_list #指定例外用户列表，这些用户不被锁定在主目录        
        
    allow_writeable_chroot=YES    
    local_umask=022    
    dirmessage_enable=YES    
    xferlog_enable=YES    
    connect_from_port_20=YES    
    xferlog_std_format=YES    
    listen=YES    
    pam_service_name=vsftpd    
    userlist_enable=YES    
    tcp_wrappers=YES    

## 6. Apache & PHP & phpMyAdmin

> Reference:     
> -- [LAMP安装](https://linux.cn/article-1567-1.html)    
> -- [httpd.apache.org](http://httpd.apache.org/docs/current/)   

### 6.1. Apache

通过 yum 安装 Apache Httpd：

{%ace edit=true, lang='python'%}
yum install -y httpd
systemctl enable httpd 
systemctl start httpd
{%endace%}

安装后直接通过浏览器访问当前IP，例如本IP为 106.13.206.49，直接访问 http://106.13.206.49 就会呈现 Apache Http Server 的页面。

### 6.2. PHP

通过 yum 安装 PHP：

{%ace edit=true, lang='python'%}
yum install -y php
{%endace%}

安装后，我们可在 Apache 目录下创建文件：

{%ace edit=true, lang='python'%}
vi /var/www/html/testphp.php
{%endace%}

添加如下内容：

    <?php phpinfo(); ?>

重启httpd服务后访问 http://106.13.206.49/testphp.php ，可以看到我们安装的 PHP 信息。

### 6.3. phpMyAdmin

通过 yum 安装 phpMyAdmin（需要epel-release）：

{%ace edit=true, lang='python'%}
yum install -y phpMyAdmin
{%endace%}


phpMyAdmin 的访问权限等相关配置在 /etc/httpd/conf.d/phpMyAdmin.conf 中。 为使得 phpMyAdmin 可以远程访问，需要修改 /etc/httpd/conf.d/phpMyAdmin.conf 中的 RequireAny 节点为 Require all granted，如下图所示：

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8amgfjgjaj30830em0t8.jpg)

修改配置后重启 httpd 服务，即可访问 http://106.13.206.49/phpmyadmin 。

phpMyAdmin 访问的 MySQL 服务器的登录信息等相关配置在 /etc/phpMyAdmin/config.inc.php 中。 phpMyAdmin 在 http://106.13.206.49/phpmyadmin/setup 提供了web版的配置页面，但访问前需要先删除默认的 /etc/phpMyAdmin/config.inc.php 文件，否则会[报错](https://stackoverflow.com/questions/42014742/logging-into-phpmyadmin-error)。web版相关配置可[参考](https://blog.csdn.net/qq_37608398/article/details/80716458)，但需要注意，web版配置后是得到配置的内容，并没有保存到服务器的配置文件中，因此，需要人工拷贝配置内容保存到 /etc/phpMyAdmin/config.inc.php 中。

以上全部配置完成后，重启httpd服务，即可正常访问 phpMyAdmin 了。

## 7. HTTPS

### 7.1 Virtual Host

> Reference:     
> -- [apache.vhosts.name-based](http://httpd.apache.org/docs/2.4/vhosts/name-based.html)    

安装 HTTPS 证书前，首先需要为网站配置虚拟主机，如为 lmay.net 配置虚拟主机：

{%ace edit=true, lang='python'%}
mkdir -p /var/www/lmay.net
vim /etc/httpd/conf/httpd.conf #新增如下内容
{%endace%}

      ServerName localhost:80    
      <VirtualHost *:80>     
         ServerName www.lmay.net      
         ServerAlias lmay.net *.lmay.net    
         DocumentRoot "/var/www/lmay.net"    
      </VirtualHost>    


### 7.2 certbot

> Reference:     
> -- [certbot](https://certbot.eff.org/)     

使用 certbot 安装 HTTPS 证书：

{%ace edit=true, lang='python'%}
yum install -y yum-utils
yum-config-manager --enable rhui-REGION-rhel-server-extras rhui-REGION-rhel-server-optional
yum install -y certbot python2-certbot-apache
certbot --apache #select both choices like 1,2 to support both website
httpd -S #check config
{%endace%}








