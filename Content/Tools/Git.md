<!-- toc -->

# Git
---

> see [【Git Doc】](https://git-scm.com/docs) [【Git教程】](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)

## 1. Glossary

### 1.1. Git

> **_Git is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency._**

### 1.2. Repository

> **_Repository is a directory, which all the files under it will be controlled by Git._**
**_Usually, a repository means a project._**

### 1.3. Working Tree

> **_Working tree is the local directory of repository, only this place could be directly edited by users._**

### 1.4. Index(Stage)

> **_After "add" command, the change will be transfered from Working Tree to Index(Stage)._**
**_Finally, after "commit" command, the change will be transfered from Index(Stage) to repository._**

### 1.5. Head

> **_Every commit will generate a version._**
**_Head is a pointer to the newest version of current branch repository._**

### 1.6. Branch

> **_Branch is the way to work on different versions of a repository at one time.
By default your repository has one branch named master which is considered to be the definitive branch. We use branches to experiment and make edits before committing them to master.  
When you create a branch off the master branch, you’re making a copy, or snapshot, of master as it was at that point in time._**

## 2. Git

### 2.1. config

> **_Get and set repository or global options._**

### 2.2. help

> **_Display help information about Git._**

## 3. Repository

### 3.1. init

> **_Create an empty Git repository or reinitialize an existing one._**

### 3.2. clone

> **_Clone a repository into a new directory._**

## 4. Working Tree & Index(Stage)

### 4.1. add

> **_Add file contents to the index._**

### 4.2. status

> **_Show the working tree status._**

### 4.3. diff

> **_Show changes between commits, commit and working tree, etc._**

### 4.4. commit

> **_Record changes to the repository._**

### 4.5. rm

> **_Remove files from the working tree and from the index._**

### 4.6. log

> **_Show commit logs._**

### 4.7. reset

> **_1. Reset current version to specified version. It means the HEAD pointer change to specified version._**
**_2. Reset files in Index(Stage), it means the change in Index(Stage) will turn back to working tree._**

### 4.8. checkout

> **_Restore working tree files from Index(Stage) or repository. It means discard the changes in working tree._**

> **_Tips:
Command "checkout" can also be used to change branch, so in this scene, the comand must use with "--"._**

## 5. Branching & Merging

### 5.1. branch

> **_List, create, or delete branches._**

### 5.2 checkout

> **_Switch branches._**

### 5.3. merge

> **_Join two or more development histories together._**

## 6. Sharing and Updating Projects

### 6.1. pull

> **_Fetch from and integrate with another repository or a local branch._**

### 6.2. push

> **_Update remote refs along with associated objects._**

### 6.3. remote

> **_Manage set of tracked repositories._**

## 7. FAQ

### 7.1. Set Chinese Log

> **_git config --global core.quotepath false_**
