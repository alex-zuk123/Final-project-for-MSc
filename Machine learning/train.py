# 导入需要的包
import re
import pandas as pd
from nltk.corpus import stopwords
stop_words=stopwords.words("english")
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# 文本预处理
# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def english_word_cut(mytext):
    #英文需要词形还原，小写
    mytext = str(mytext).lower() #小写
    mytext = re.compile("[^a-z^A-Z^,^.^'^\-^_^ ]").sub("", mytext)#文本过滤
#     分词
    mytext = word_tokenize(mytext)
    word_list = []
    tagged_sent = pos_tag(mytext)  # 获取单词词性
    for seg_word in tagged_sent:
        word = seg_word[0]
        if word not in stop_words and len(word)>2 and seg_word[1]:
            word_list.append(word)
    # 词形还原
    tagged_sent = pos_tag(word_list)  # 获取单词词性
    wnl = WordNetLemmatizer() #词形还原
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
    return ' '.join(lemmas_sent)
if __name__ == '__main__':
    #加载所有数据
    df = pd.read_csv('./data/review (1).csv')
    df.dropna(inplace=True)
    # 转化为标签字典保存
    labels = df['rating'].tolist()
    # 标签转化
    label_dict = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    # 标签转为化对应索引
    labels_ = [label_dict[i] for i in labels]
    # 数据平衡
    df['labels_'] = labels_
    df_ = pd.DataFrame()
    # 各个类别数据量差别大，取最少的一个类数量，其实也非常多数据
    length = []
    for i in [0, 1, 2]:
        length.append(len(df[df['labels_'] == i]))
    length = min(length)
    for i in [0,1,2]:
        temp = df[df['labels_']==i][:length]
        df_ = pd.concat([df_,temp])
    # 数据预处理
    texts = df_['review_text'].tolist()
    texts = [english_word_cut(str(item)) for item in texts]
    labels_ = df_['labels_'].tolist()
    # 随机打乱顺序
    texts, labels = shuffle(texts, labels_)
    # 向量化
    print('文本向量化')
    # tfidf向量化方法
    vectorizer = TfidfVectorizer()
    # tfidf模型训练
    _tfidf = vectorizer.fit_transform(texts)
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3)
    # 训练文本转化为向量
    train_tfidf = vectorizer.transform(x_train)
    test_tfidf = vectorizer.transform(x_test)
    # 模型
    clf = MultinomialNB()
    # 模型在训练集上进行训练
    clf.fit(train_tfidf, y_train)  # 模型训练
    # 在测试集上用模型预测结果
    y_pre = clf.predict(test_tfidf)
    # 评价模型训练结果
    print('模型评价：')
    # 标签名称
    target_names = ['负面','中性','赞扬']
    # 输出准确率 准确率 召回率 精准率 f1 等评价指标
    print(classification_report(y_test, y_pre, target_names=target_names, digits=3))
    # 用所有数据训练一遍模型用于predict
    clf = MultinomialNB()
    clf.fit(_tfidf, labels)
    print('保存模型')
    # 保存训练好的朴素贝叶斯模型 使用pickle保存
    with open('./model/NB.pickle', 'wb') as fw:
        pickle.dump(clf, fw)
    # 保存训练好的tfidf向量化模型 使用pickle保存
    with open('./model/vectorizer.pickle', 'wb') as fw:
        pickle.dump(vectorizer, fw)