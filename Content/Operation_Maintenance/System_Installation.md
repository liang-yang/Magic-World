<!-- toc -->

# System Installation

## LAMP

Reference:  
[LAMP安装](https://linux.cn/article-1567-1.html)

### Apache

{%ace edit=true, lang='java'%}
yum install -y httpd
{%endace%}

### MySql

Reference:  
[A Quick Guide to Using the MySQL Yum Repository](https://dev.mysql.com/doc/mysql-yum-repo-quick-guide/en/)

{%ace edit=true, lang='java'%}
// 配置Mysql扩展源
rpm -ivh https://dev.mysql.com/get/mysql80-community-release-el7-3.noarch.rpm
// yum安装mysql 5.7
yum repolist all | grep mysql
yum-config-manager --disable mysql80-community
yum-config-manager --enable mysql57-community
yum install -y mysql-community-server
yum list installed | grep mysql
// 启动Mysql，并加入开机自启
systemctl start mysqld
systemctl enable mysqld
// 使用临时密码登录数据库
mysql -uroot -p$(awk '/temporary password/{print $NF}' /var/log/mysqld.log)

// 修改用户密码、可远程登录
mysql> ALTER USER 'root'@'localhost' IDENTIFIED BY 'Y!a4364565464';
mysql> SHOW VARIABLES LIKE 'validate_password%';
mysql> SET GLOBAL validate_password_policy=LOW;
mysql> SET GLOBAL validate_password_length=6;
mysql> CREATE USER 'yangliang'@'%' IDENTIFIED BY 'y12345678';
mysql> GRANT ALL ON *.* TO 'yangliang'@'%';
{%endace%}

### PHP











