from llama_cpp import Llama

llm = Llama(model_path="models/mistral-7b.gguf")

def explain_gap(gap, papers):
    prompt = f"""
    Gap: {gap}
    Evidence papers: {[p.title for p in papers[:3]]}

    Explain why this gap matters.
    """
    return llm(prompt)["choices"][0]["text"]
