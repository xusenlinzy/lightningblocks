import clueai
import gradio as gr


client = clueai.Client("", check_api_key=False)


def classify(task, text, labels):
    labels = labels.split(" ")
    res = client.classify(model_name="clueai-base", task_name=task, inputs=[text], labels=labels)
    return task, res.classifications[0]


def generate(task, prompt):
    res = client.generate(model_name="clueai-base", prompt=prompt)
    return task, res.generations[0].text


def text2image(prompt, style):
    client.text2image(model_name="clueai-base", prompt=prompt, style=style)
    return "test.png"


classify_demo = gr.Interface(
    classify,
    [
        gr.Textbox(lines=2, label="task"),
        gr.Textbox(lines=5, label="text"),
        gr.Textbox(lines=2, label="labels"),
    ],
    [gr.Text(label="task"), gr.Json(label="result")],
    examples=[
        ["产品分类", "强大图片处理器，展现自然美丽的你,,修复部分小错误，提升整体稳定性。",
         " ".join(["美颜", "二手", "外卖", "办公", "求职"])],
        ["产品分类", "求闲置买卖，精品购物，上畅易无忧闲置商城，安全可信，优质商品有保障",
         " ".join(["美颜", "二手", "外卖", "办公", "求职"])],
    ],
)


generate_demo = gr.Interface(
    generate,
    [
        gr.Textbox(lines=2, label="task"),
        gr.Textbox(lines=5, label="prompt"),
    ],
    [gr.Text(label="task"), gr.Text(label="result")],
    examples=[
        ["文本摘要",
        """摘要：
本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代人
答案："""],
        ["意图分类",
        """意图分类：
帮我定一个周日上海浦东的房间
选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
答案："""],
        ["推理关系判断",
        """推理关系判断：
前提：小明今天在北京
假设：小明在深圳旅游
选项：矛盾，蕴含，中立
答案："""],
        ["信息抽取",
        """信息抽取：
据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
问题：机构名，人名，职位
答案："""],
        ["翻译",
        """翻译成中文：
This is a dialogue robot that can talk to people.
答案："""],
        ["问题生成",
        """问题生成：
中新网2022年9月22日电 22日，商务部召开例行新闻发布会，商务部新闻发言人束珏婷表示，今年1-8月，中国实际使用外资1384亿美元，增长20.2%；其中，欧盟对华投资增长123.7%(含通过自由港投资数据)。这充分表明，包括欧盟在内的外国投资者持续看好中国市场，希望继续深化对华投资合作。
答案："""],
        ["关键词抽取",
        """抽取关键词：
当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
关键词："""],
        ["指代消解",
        """代词指向哪个名词短语：
段落：
当地时间9月21日，英国首相特拉斯在纽约会见了美国总统拜登。随后她便在推特上发文强调，英美是坚定盟友。推文下方还配上了她（代词）与拜登会面的视频。
问题：代词“她”指代的是？
答案："""],
        ["电商客户需求分析",
        """电商客户诉求分类：
收到但不太合身，可以退换吗
选项：买家咨询商品是否支持花呗付款，买家表示收藏关注店铺，买家咨询退换货规则，买家需要商品推荐
答案："""],
        ["新闻分类",
        """新闻分类：
今天（3日）稍早，中时新闻网、联合新闻网等台媒消息称，佩洛西3日上午抵台“立法院”，台湾新党一早8时就到台“立法院”外抗议，高喊：“佩洛西，滚蛋！”台媒报道称，新党主席吴成典表示，佩洛西来台一点道理都没有，“平常都说来者是客，但这次来的是祸！是来祸害台湾的。”他说，佩洛西给台湾带来祸害，“到底还要欢迎什么”。
选项：财经，法律，国际，军事
答案："""],
        ["情感分析",
        """情感分析：
这个看上去还可以，但其实我不喜欢
选项：积极，消极
答案："""],
        ["阅读理解",
        """阅读以下对话并回答问题。
男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
问题：女的怎么样？
选项：正在工作，感冒了，在打电话，要出差。
答案："""],
        ["语义相似度",
        """下面句子是否表示了相同的语义：
文本1：糖尿病腿麻木怎么办？
文本2：糖尿病怎样控制生活方式
选项：相似，不相似
答案："""],
    ],
)


text2image_demo = gr.Interface(
    text2image,
    [
        gr.Textbox(lines=2, label="prompt"),
        gr.Textbox(lines=2, label="style"),
    ],
    "image",
    examples=[
        ["秋日的晚霞", "毕加索"],
        ["远处有雪山的蓝色湖泊，蓝天白云，很多鸟", "达芬奇"],
        ["一只猫坐在椅子上，戴着一副墨镜", "梵高"],
    ],
)


demo = gr.TabbedInterface([classify_demo, generate_demo, text2image_demo],
                          ["Classification", "Generation", "Text2Image"])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
