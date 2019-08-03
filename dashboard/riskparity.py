# ====== RISK PARITY COMP =====#
import numpy as np
from scipy.optimize import minimize

def allocation_risk(weights, covariances):

   # We calculate the risk of the weights distribution
   portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

   # It returns the risk of the weights distribution
   return portfolio_risk

def assets_risk_contribution_to_allocation_risk(weights, covariances):

   # We calculate the risk of the weights distribution
   portfolio_risk = allocation_risk(weights, covariances)

   # We calculate the contribution of each asset to the risk of the weights
   # distribution
   assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
       / portfolio_risk

   # It returns the contribution of each asset to the risk of the weights
   # distribution
   return assets_risk_contribution

def risk_budget_objective_error(weights, args):

   # The covariance matrix occupies the first position in the variable
   covariances = args[0]

   # The desired contribution of each asset to the portfolio risk occupies the
   # second position
   assets_risk_budget = args[1]

   # We convert the weights to a matrix
   weights = np.matrix(weights)

   # We calculate the risk of the weights distribution
   portfolio_risk = allocation_risk(weights, covariances)

   # We calculate the contribution of each asset to the risk of the weights
   # distribution
   assets_risk_contribution = \
       assets_risk_contribution_to_allocation_risk(weights, covariances)

   # We calculate the desired contribution of each asset to the risk of the
   # weights distribution
   assets_risk_target = \
       np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

   # Error between the desired contribution and the calculated contribution of
   # each asset
   error = \
       sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

   # It returns the calculated error
   return error

def get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):
   TOLERANCE = 1e-10
   # Restrictions to consider in the optimisation: only long positions whose
   # sum equals 100%
   constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                  {'type': 'ineq', 'fun': lambda x: x})

   # Optimisation process in scipy
   optimize_result = minimize(fun=risk_budget_objective_error,
                              x0=initial_weights,
                              args=[covariances, assets_risk_budget],
                              method='SLSQP',
                              constraints=constraints,
                              tol=TOLERANCE,
                              options={'disp': True})

   # Recover the weights from the optimised object
   weights = optimize_result.x

   # It returns the optimised weights
   return weights

def spinu(weights,args):
    cov = args[0]
    assets_risk_budget = args[1]
    obj = 0.5 * np.matmul(np.matmul(weights,cov),weights) - np.sum(np.multiply(assets_risk_budget,np.log(weights)))
    return obj

def design_pf(cov, b, x0):
    optimize_result = minimize(fun=spinu,
                              x0=x0,
                              args=[cov, b],
                              method='SLSQP',
                              tol=1e-10,
                              options={'disp': False})
    res = optimize_result.x
    return res/sum(res)