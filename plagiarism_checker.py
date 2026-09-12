"""
论文查重工具 (Plagiarism Checker)
--------------------------------
基于 TF-IDF + 余弦相似度，批量比对一篇论文与本地文件夹中所有文本的相似度。
支持极值报告与重复关键词提取。

用法：
    python plagiarism_checker.py
    按提示输入待查文件路径和目标文件夹路径
"""
import docx
import os
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_text(file_path):
    """读取文件内容，支持 .txt 和 .docx"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    elif ext == '.docx':
        try:
            doc = docx.Document(file_path)
            # 提取所有段落文字，过滤掉空段落
            return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            print(f"⚠️ 读取 Word 失败 {file_path}: {e}")
            return ''

    else:
        print(f"⚠️ 跳过不支持的文件格式：{file_path}")
        return ''


def cut_words(text):
    """中文分词，用空格连接，供 TfidfVectorizer 使用。"""
    return ' '.join(jieba.lcut(text))


def find_common_keywords(text1, text2, top_n=10):
    """提取两篇文本共有的高频关键词（重复风险较高的词）。"""
    kw1 = set(jieba.analyse.extract_tags(text1, topK=top_n))
    kw2 = set(jieba.analyse.extract_tags(text2, topK=top_n))
    return list(kw1 & kw2)


def collect_files(folder_path):
    """收集文件夹下所有文件的完整路径，返回列表。"""
    files = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files


def calc_similarity(text_a, text_b):
    """计算两段文本的余弦相似度，返回百分比（0-100）。"""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([text_a, text_b])
    sim = cosine_similarity(matrix[0:1], matrix[1:2])
    return sim[0][0] * 100


def main():
    target = input('请输入你想要查重的文件: ').strip()
    folder = input('请输入文件夹位置: ').strip()

    target_text = read_text(target)
    files = collect_files(folder)

    results = {}

    for path in files:
        # 跳过自身
        if os.path.abspath(path) == os.path.abspath(target):
            continue

        other_text = read_text(path)
        if not other_text.strip():
            print(f"⚠️ 跳过空文件：{path}")
            continue

        cut_a = cut_words(target_text)
        cut_b = cut_words(other_text)
        score = calc_similarity(cut_a, cut_b)

        file_name = os.path.basename(path)
        results[file_name] = score

        print("=" * 30)
        print(f"📄 文件：{file_name}")
        print(f"⚠️ 整体相似度：{score:.2f}%")
        print("=" * 30)

        common = find_common_keywords(target_text, other_text, top_n=8)
        print("📌 疑似高度重复的核心词汇（建议改写）：")
        for word in common:
            print(f"   - {word}")
        print()

    # 极值报告
    if results:
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        print("=" * 40)
        print("📊 查重率极值报告")
        print("=" * 40)
        print(f"🔴 最高相似度：{ranked[0][0]} —— {ranked[0][1]:.2f}%")
        print(f"🟢 最低相似度：{ranked[-1][0]} —— {ranked[-1][1]:.2f}%")
        print("=" * 40)
    else:
        print("❌ 文件夹里除了待查文件，没有其他文件可比较。")


if __name__ == "__main__":
    main()
