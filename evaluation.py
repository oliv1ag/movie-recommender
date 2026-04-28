def precision_at_k(recommended, relevant, k):
    if len(recommended) == 0:
        return 0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = sum([1 for item in recommended_k if item in relevant_set])
    return hits / k


def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0

    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = sum([1 for item in recommended_k if item in relevant_set])
    return hits / len(relevant)


def evaluate_models(content_recs, collab_recs, hybrid_recs, ground_truth):
    k = 10

    print("\nEvaluation Results:")

    for name, recs in {
        "Content": content_recs,
        "Collaborative": collab_recs,
        "Hybrid": hybrid_recs
    }.items():

        prec = precision_at_k(recs, ground_truth, k)
        rec = recall_at_k(recs, ground_truth, k)

        print(f"{name} -> Precision@10: {prec:.3f}, Recall@10: {rec:.3f}")