from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# 検索対象のテキストを準備
texts = [
    "私の趣味は読書です。",
    "私の好きな食べ物はカレーです。",
    "私の嫌いな食べ物は饅頭です。",
]

# 近傍探索ライブラリ「Faiss」でベクトル検索するためのインデックスを作成
print("Indexing...")
vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
print("Indexed.")

# プロンプトのテンプレートを準備
prompt = ChatPromptTemplate.from_template(
    """以下のcontextだけに基づいて回答してください。

{context}

質問: {question}
"""
)

# LLMを準備
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# 「検索 => プロンプトの穴埋め => LLMで回答を生成」という連鎖（chain）を作成
chain: Runnable = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# chainを実行
question = "私の好きな食べ物はなんでしょう？"
print("Invoking chain...")
print(f"Question: '{question}'")
answer = chain.invoke(question)
print(f"Answer: '{answer}'")
