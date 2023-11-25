
# The c@1 score is adopted from Mike Kestemont's implementation, which is offered in
#  https://github.com/pan-webis-de/pan-code/blob/master/clef22/authorship-verification/pan22_verif_evaluator.py
# In my implementation, the punctual threshold is translated to a range.

def c_at_1(true_y, pred_y, threshold_range=0.1):
    """
    Calculates the c@1 score, an evaluation method specific to the
    PAN competition. This method rewards predictions which leave
    some problems unanswered (score = 0.5). See:
        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will always be `0` or `1`.
    Returns
    ----------
    c@1 = the c@1 measure (which accounts for unanswered
        problems.)
    References
    ----------
        - E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
        - A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    """
    lower_level = 0.5 - (threshold_range / 2)
    upper_level = 0.5 + (threshold_range / 2)

    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score <= lower_level and (gt_score == 0):
            nc += 1
        elif pred_score >= upper_level and (gt_score == 1):
            nc += 1

        elif (lower_level < pred_score < upper_level):
            nu += 1.0

    return (1 / n) * (nc + (nu * nc / n))


def calculate_share(text_as_list, ne_list, normalize='l1', standardize=False, case_sensitive=False):
    """
    proceeds the calculation operation and stores the result wordlist_shares attribute.
    returns the abs/relative/normalized number of hits of the tokens of an input list relative to a list of searched types (ne_list)
    """
    # text_as_list = text.split(" ")

    print(text_as_list)
    print(len(text_as_list))
    hits = 0
    if case_sensitive == True:
        for token in text_as_list:
            if token in ne_list:
                hits += 1
    elif case_sensitive == False:
        wordlist_lower = [token.lower() for token in ne_list]
        for token in text_as_list:
            if token.lower() in wordlist_lower:
                hits += 1

    if normalize ==  "abs" and standardize == False:
        share = hits
    elif normalize == "l1" and standardize == False:
        share = hits / len(text_as_list)
    elif normalize == "l1" and standardize == True:
        share = hits / (len(text_as_list) * len(ne_list))


    return share
