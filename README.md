## 🌟 核心亮点：传统字面查重 vs AI 语义查重

传统的查重工具（如 TF-IDF）只能查“字面抄袭”，一旦把“公司”改成“企业”，把“融资”改成“资金注入”，传统算法就会失效。

本工具引入了 **Sentence-BERT (深度学习模型)**，能够真正理解句子的含义。

**实测数据对比（针对深度改写）**：
*   传统字面相似度：**1.68%**
*   AI 语义相似度：**68.04%**

📸 **效果截图**：
<img width="1100" height="622" alt="image" src="https://github.com/user-attachments/assets/15166fc4-8ede-4861-9524-5adc3142eb5b" />

🚀 **在线体验地址**：[点击这里立即使用](https://plagiarismchecker-bwqzeo37z4yhlz948r5e9s.streamlit.app/)
# 论文查重工具 (Plagiarism Checker)

一个基于 Python 的本地论文查重脚本，用 TF-IDF + 余弦相似度批量比对一篇论文与本地文件夹中所有文本的相似度。

## 两种使用方式

1. **命令行版**：适合批量处理，运行 `python plagiarism_checker.py`
2. **Web 界面版**：可视化操作，支持上传文件或读取本地文件夹，运行 `streamlit run app.py`

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
