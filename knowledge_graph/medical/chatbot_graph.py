from question_classifier import QuestionClassifier
from question_parser import QuestionPaser
from answer_search import AnswerSearcher


class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = "您好，我是医药智能助理，希望可以帮到您。"
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        return '\n'.join(final_answers) if final_answers else answer


if __name__ == '__main__':
    handler = ChatBotGraph()
    while 1:
        question = input("用户: ")
        answer = handler.chat_main(question)
        print("chatbot: ", answer)
