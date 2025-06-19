from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question(question, vectorstore):
    """Get answer to question using RAG"""
    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
        )
        
        prompt_rag = PromptTemplate.from_template(
            """Ответьте на вопрос: {question} на основе следующего контекста: {context}
            
            Инструкции:
            - Отвечайте на русском языке
            - Предоставляйте ссылки на исходный материал, когда это возможно
            - Если ответ не может быть найден в контексте, четко скажите об этом
            - Форматируйте ответ для лучшей читаемости
            - Используйте списки и подзаголовки, если это уместно
            
            Вопрос: {question}
            """
        )
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0,
            max_tokens=1000
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_rag
            | llm
        )
        
        result = rag_chain.invoke(question)
        return result.content
    
    except Exception as e:
        return f"❌ Ошибка при обработке вопроса: {str(e)}"