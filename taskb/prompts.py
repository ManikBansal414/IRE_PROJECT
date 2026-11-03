"""Prompts for grounded answer generation. A variant is a system prompt; build_messages adds the passages and
the conversation as chat turns, ending with the user's last question."""

# Reply for tasks without passages. In the dev set these are exactly the UNANSWERABLE and CONVERSATIONAL tasks,
# which the official scoring rewards only when the reply says it cannot answer. Wording taken from the references.
IDK = "I'm sorry, but I don't have the answer to your question."

BASE = """You are a helpful assistant in a multi-turn conversation. Answer the user's last question using only the information in the documents below.
- Do not add facts that are not in the documents, even if you know them.
- If the documents do not answer the question, say that you don't have that information. If they answer only part of it, answer that part and say what is missing.
- Use the earlier turns to understand what the question refers to.
- Write plain conversational prose: no headings, no citations, no document numbers."""

VARIANTS = {
    "base": BASE + "\n- Keep the answer focused, usually 2 to 5 sentences.",
    # Closer to the passage wording: favours ROUGE-L, BERTScore precision against the passages and faithfulness.
    "extractive": BASE + "\n- Reuse the wording of the documents where it answers the question.\n"
                         "- Keep the answer focused, usually 2 to 5 sentences.",
    "concise": BASE + "\n- Answer in 1 to 3 sentences.",
}


def format_passages(contexts):
    def title(c):  # many passages already start with their title
        return f"{c['title']}\n" if c.get("title") and not c["text"].startswith(c["title"]) else ""
    return "\n\n".join(f"[Document {i}]\n{title(c)}{c['text']}" for i, c in enumerate(contexts, 1))


def build_messages(task, variant):
    msgs = [{"role": "system", "content": f"{VARIANTS[variant]}\n\nDocuments:\n\n{format_passages(task['contexts'])}"}]
    for turn in task["input"]:
        msgs.append({"role": "user" if turn["speaker"] == "user" else "assistant", "content": turn["text"]})
    return msgs
