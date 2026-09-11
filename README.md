# plagiarism_checker.py
# 论文查重工具 (Plagiarism Checker)

一个基于 Python 的本地论文查重脚本，用 TF-IDF + 余弦相似度批量比对一篇论文与本地文件夹中所有文本的相似度。

## 功能

- 批量比对：自动遍历目标文件夹，逐一计算相似度
- 排除自身：不会把待查文件和自己比对
- 极值报告：列出相似度最高和最低的比对文件
- 重复关键词：提取两篇文本共有的高频词，辅助定位改写重点
- 空文件保护：自动跳过读取失败或空内容的文件

## 技术栈

- Python 3
- jieba（中文分词）
- scikit-learn（TF-IDF 向量化 + 余弦相似度）

## 安装

```bash
pip install jieba scikit-learn
