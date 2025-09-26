# smartelva.py (或 generate_blog.py)

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. 初始化模型 (LLamaCpp 部分保持不变)
# *** 关键修改在这里：使用相对于当前运行目录的文件名 ***
MODEL_PATH = r"F:\DL\LLM\Complete-Langchain-Tutorials\Blog_Generation\llama-2-7b-chat.Q2_K.gguf" 

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_tokens=2048,
    n_ctx=4096, 
    verbose=True, 
)

# 2. 定义提示模板 (Prompt Template)
# *** 修改 1: 模板只需一个占位符 {topic} ***
template = """
{topic}
"""
# input_variables 保持不变，用于接收输入
prompt = PromptTemplate(template=template, input_variables=["topic"])

# 3. 创建 LLMChain (保持不变)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 4. 运行应用
# *** 修改 2: 将问题赋值给输入变量 ***
blog_topic = "where is the capital of china"

# 调整打印信息，使其更贴合实际任务
print(f"--- Asking simple question: {blog_topic} ---")

# *** 修改 3: 使用命名参数 (topic=...) 避免 StopIteration 错误 ***
response = llm_chain.run(topic=blog_topic)

print("\n--- GENERATED CONTENT ---\n")
print(response)
print("\n-------------------------\n")