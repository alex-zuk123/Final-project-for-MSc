# 需要使用的工具包
import pickle
import re
from nltk.corpus import stopwords
stop_words=stopwords.words("english")
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# 模型预测类
class NB_predict():
    def __init__(self):
        # 加载训练好的模型
        # 标签定义
        self.label_dict = ['负面','中性','赞扬']
        # 朴素贝叶斯模型
        with open('./model/NB.pickle', 'rb') as NB:
            self.NB_model = pickle.load(NB)
        # tfidf向量化
        with open('./model/vectorizer.pickle', 'rb') as vectorize:
            self.vectorizer = pickle.load(vectorize)


    def get_wordnet_pos(self,tag):
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

    def english_word_cut(self,mytext):
        # 英文需要词形还原，小写
        mytext = str(mytext).lower()  # 小写
        mytext = re.compile("[^a-z^A-Z^,^.^'^\-^_^ ]").sub("", mytext)  # 文本过滤
        #     分词
        mytext = word_tokenize(mytext)
        word_list = []
        tagged_sent = pos_tag(mytext)  # 获取单词词性
        for seg_word in tagged_sent:
            word = seg_word[0]
            if word not in stop_words and len(word) > 2 and seg_word[1]:
                word_list.append(word)
        # 词形还原
        tagged_sent = pos_tag(word_list)  # 获取单词词性
        wnl = WordNetLemmatizer()  # 词形还原
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        return ' '.join(lemmas_sent)

    def process(self,line):
        # 预处理
        word_list = self.english_word_cut(line)
        # 返回结果
        return word_list
    #得到预测结果
    def get_result(self,text):
        # 对输入文本进行预处理
        test_data = [self.process(text)]
        # 预处理之后进行向量化
        test_tfidf = self.vectorizer.transform(test_data)
        #预测
        test_pre = self.NB_model.predict(test_tfidf)
        # 返回预测结果
        return self.label_dict[test_pre[0]]

if __name__ == '__main__':
    # 需要预测的数据
    nb_predict = NB_predict()
    while 1:
        text = input('请输入需要分析的数据：')
        # NB_predict类实例化
        print('预测结果：')
        # 进行预测,并输出结果
        print(nb_predict.get_result(text))
        print()