def rp_cost_func(wts_vec, ret_mat, min_type="mad"):
    risk_contribution_vec = calc_risk_contribution(wts_vec, ret_mat)
    if min_type == "mad":
        return sum(abs(risk_contribution_vec - 1 / len(risk_contribution_vec)))
    elif min_type == "mse":
        return (risk_contribution_vec - 1 / len(risk_contribution_vec)).std()


def calc_risk_contribution(wts_vec, ret_mat):
    Sigma = ret_mat.cov()
    var_p = wts_vec.dot(Sigma).dot(wts_vec)
    risk_contribution_vec = (wts_vec * (Sigma.dot(wts_vec))) / var_p
    return risk_contribution_vec
