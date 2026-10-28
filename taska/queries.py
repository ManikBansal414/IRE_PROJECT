"""Turn a conversation (list of user/agent turns) into a retrieval query.

Every strategy here uses only the conversation itself, which is what is available
at evaluation time. `gold_rewrite` is the exception and exists only as an upper bound.
"""


def user_turns(task):
    return [t["text"].strip() for t in task["input"] if t["speaker"] == "user"]


def last_turn(task):
    return user_turns(task)[-1]


def all_questions(task):
    """All user questions, most recent last."""
    return "\n".join(user_turns(task))


def last_k_questions(task, k=2):
    return "\n".join(user_turns(task)[-k:])


def weighted_history(task, repeat_last=2, history=2):
    """Last question repeated (heavier BM25 weight) plus the previous `history` questions."""
    qs = user_turns(task)
    return "\n".join(qs[-1 - history:-1] + [qs[-1]] * repeat_last)


STRATEGIES = {
    "last": last_turn,
    "all": all_questions,
    "last2": last_k_questions,
    "weighted": weighted_history,
}
