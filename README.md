学习练习pytorch
====

# word2vec
## 一.十问word2vec
### 1.介绍一下word2vec:
> a.两个模型是CBOW和Skip-gram，两个加速训练的技巧是HS(Hierarchical Softmax)和负采样</br>
> b.假设一个训练样本是核心词w和其上下文context（w）组成，CBOW就是用去预测w;而SKip-gram则反过来，是用w去预测context(w)里的所有词</br>
> c.HS是试图用词频建立一棵哈夫曼树，那么经常出现的词路径会比较短。树的叶子节点表示词，共词典大小多个，而非叶子结点是模型的参数，比词典个数少一个。要预测的词，转化成预测从根节点到该词所在叶子节点的路径，是多个二分类问题。</br>
> d.对于负采样，则是把原来的 Softmax 多分类问题，直接转化成一个正例和多个负例的二分类问题。让正例预测 1，负例预测 0，这样子更新局部的参数。</br>

### 2.对比 Skip-gram 和 CBOW
> a. 训练速度上 CBOW 应该会更快一点。因为每次会更新 context(w) 的词向量，而 Skip-gram 只更新核心词的词向量。两者的预测时间复杂度分别是 O(V)，O(KV)</br>
> b.Skip-gram 对低频词效果比 CBOW好。因为是尝试用当前词去预测上下文，当前词是低频词还是高频词没有区别。但是 CBOW相当于是完形填空，会选择最常见或者说概率最大的词来补全，因此不太会选择低频词。（想想老师学生的那个例子）Skip-gram 在大一点的数据集可以提取更多的信息。总体比 CBOW 要好一些。</br>

### 3.对比 HS 和 负采样
>a.优化目标：
>>HS让每个非叶子节点去预测要选择的路径（每个节点是个二分类问题），目标函数是最大化路径上的二分类概率。</br>
>>负采样是最大化正样例概率同时最小化负样例概率。</br>

>b.负采样更快一些，特别是词表很大的时候。与HS相比，负采样不再使用霍夫曼树，而是使用随机负采样，能大幅度提高性能。</br>

### 4.负采样为什么要用词频来做采样概率
>因为这样可以让频率高的词先学习，然后带动其他词的学习

### 5.为什么训练完有两套词向量，为什么一般只用前一套
>a.对于 Hierarchical Softmax 来说，哈夫曼树中的参数是不能拿来做词向量的，因为没办法和词典里的词对应。</br>
>b.负采样中的参数其实可以考虑做词向量，因为中间是和前一套词向量做内积，应该也是有意义的。但是考虑负样本采样是根据词频来的，可能有些词会采不到，也就学的不好</br>

### 6.对比字向量和词向量
>a.字向量其实可以解决一些问题，比如未登陆词，还有做一些任务的时候还可以避免分词带来的误差。</br>
>b.词向量它的语义空间更大，更加丰富，语料足够的情况下，词向量是能够学到更多的语义的。</br>

### 7.为什么负采样/分层softmax能加快训练
>a. 负采样 </br>
>> 1在优化参数的时候，只更新涉及到的向量参数；</br>
>> 2 放弃用softmax而是用sigmoid，原来的方法中softmax需要遍历所有单词的概率得分。</br>

>b.分层softmax：上面的 Softmax 每次和全部的词向量做内积，复杂度是 O(V)，V 是词典大小。如果考虑把每个词都放到哈夫曼树的叶节点上，用sigmoid做二分类，那么复杂度就可以降为 O(logV)，即树的高度，因为只需要预测从根节点到相应叶节点的路径即可。

### 8.word2vec的缺点
>a.忽略了词序</br>
>b.一词多义无法识别</br>

### 9.hs为什么用霍夫曼树而不用其他二叉树
>这是因为Huffman树对于高频词会赋予更短的编码，使得高频词离根节点距离更近，从而使得训练速度加快。</br>

### 10.为什么用的是线性激活函数？
>word2vec不是为了做语言模型，它不需要预测得更准

## 二.论文复现
> 代码位于word2vec目录中

# Tranmsfor和Bert
## 1.理论知识
### 1.
### 2.
## 2.注意问题
### 1.
### 2.

# 目前的问题
## 1.GPU计算的使用，***.cuda(),熟悉语法。
> 看资料然后多试试</br>

## 2.word2vec论文的复现。
> 看懂pytorch实现版本的代码</br>
> 实现tensorboard和实现训练中止并重启

## 3.transform和BERT理论和实践。
## 4.各类文本分类地址(刷题):https://github.com/649453932
## 5.基于Pytorch的Bert应用项目,地址:https://mp.weixin.qq.com/s/GLG1NJQFboC-YyOT-sjF2Q
## 6.各类算法比赛的代码:https://github.com/datawhalechina/competition-baseline