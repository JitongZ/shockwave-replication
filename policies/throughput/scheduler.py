import cvxpy as cp
import numpy as np


class Problem:
    def __init__(self, a):
        """
        a: 2D matrix where a[i][j] is the progress rate of user i on resource j
        """
        assert len(a.shape) == 2
        self.a = a

    @property
    def n(self):
        """Number of users in the problem."""
        return self.a.shape[0]

    @property
    def m(self):
        """Number of resources in the problem."""
        return self.a.shape[1]

    @property
    def normalized(self):
        """
        Return a normalized version of the problem where each user's progress rate would
        be 1.0 if it could use all the resources.
        """
        scale = 1.0 / self.a.sum(axis=1)
        new_a = self.a * scale.reshape(self.n, 1)
        return Problem(new_a)

    def user_rates(self, solution):
        """Return the progress rate achieved by each user in a solution."""
        return (self.a * solution.x).sum(axis=1)

    def normalized_user_rates(self, solution):
        """Return the normalized progress rate achieved by each user in a solution."""
        return (self.normalized.a * solution.x).sum(axis=1)

    def __repr__(self):
        return "Problem(a=%s)" % self.a


class Solution:
    def __init__(self, x):
        """
        x: 2D matrix where x[i][j] is the fraction of resource j allocated to user i
        """
        assert len(x.shape) == 2
        self.x = x

    def __repr__(self):
        return "Solution(x=%s)" % self.x


def solve_isolated(problem):
    """
    This algorithm splits all resources equally between all users, giving the isolation baseline.
    """
    x = np.full(problem.a.shape, 1.0 / problem.n)
    return Solution(x)


def solve_max_throughput(problem, normalize=True):
    """
    This algorithm tries to maximize total throughput by assigning each resource to the user who
    will receive the highest rate from it. If multiple users are tied in their rate from a
    resource, we split it equally between them.
    """
    a = problem.a
    if normalize:
        a = problem.normalized.a

    col_maxes = a.max(axis=0)
    is_max = a == col_maxes
    num_equal_to_max = is_max.sum(axis=0)
    scale = 1.0 / num_equal_to_max
    x = is_max * scale
    return Solution(x)


def solve_isolated_max_throughput(problem):
    """
    This algorithm tries to maximize total normalized throughput subject to giving each user
    at least as much as they'd get in the isolated solution.
    """
    a = problem.normalized.a
    x = cp.Variable(a.shape)
    objective = cp.Maximize(cp.sum(cp.multiply(a, x)))
    constraints = [
        x >= 0,
        cp.sum(x, axis=0) == 1,
        cp.sum(cp.multiply(a, x), axis=1) >= 1.0 / problem.n
    ]
    cvxprob = cp.Problem(objective, constraints)
    result = cvxprob.solve()
    assert cvxprob.status == "optimal"
    return Solution(x.value)


def solve_nash(problem):
    """
    This algorithm tries to maximize the product of the throughputs of each user (i.e. the
    Nash bargaining solution, also called CEEI).
    """
    a = problem.normalized.a   # Technically we don't need to normalize for this one
    x = cp.Variable(a.shape)
    objective = cp.Maximize(cp.geo_mean(cp.sum(cp.multiply(a, x), axis=1)))
    constraints = [
        x >= 0,
        cp.sum(x, axis=0) == 1,
    ]
    cvxprob = cp.Problem(objective, constraints)
    result = cvxprob.solve()
    assert cvxprob.status == "optimal"
    return Solution(x.value)


def solve_ks(problem):
    """
    This algorithm tries to equalize the users' normalized throughputs (i.e. the
    Kalai-Smorodinsky bargaining solution, similar to DRF).
    """
    a = problem.normalized.a
    x = cp.Variable(a.shape)
    objective = cp.Maximize(cp.min(cp.sum(cp.multiply(a, x), axis=1)))
    constraints = [
        x >= 0,
        cp.sum(x, axis=0) == 1,
    ]
    cvxprob = cp.Problem(objective, constraints)
    result = cvxprob.solve()
    assert cvxprob.status == "optimal"
    return Solution(x.value)


def find_strategy_proofness_counterexample(solve_fn):
    """
    Perturbs a single entry in random throughput matrix to attempt to generate
    examples that show lack of strategy-proofness of passed-in policy.
    """
    for i in range(50):
        a = np.random.random((4, 3))
        a_fake = a.copy()
        import random
        # Mis-advertise P4's advertised throughput for resource B.
        if i < 25:
            a_fake[3][1] += random.uniform(1, 10)
        else:
            a_fake[3][1] -= random.uniform(0, a_fake[3][1])
        problem = Problem(a)
        rates = problem.normalized_user_rates(solve_fn(problem))
        problem_fake = Problem(a_fake)
        rates_fake = problem.normalized_user_rates(solve_fn(problem_fake))
        if rates_fake[3] > (1.01 * rates[3]):
            explore_problem(a)
            explore_problem(a_fake, a_test=a)
            break


def explore_problem(a, a_test=None):
    """
    Explore a scheduling problem, printing the various solutions.
    """
    a = np.array(a, dtype=np.float64)
    problem = Problem(a)
    problem_test = Problem(a)
    if a_test is not None:
        a_test = np.array(a_test, dtype=np.float64)
        problem_test = Problem(a_test)
    print("For problem %s:" % str(a).replace('\n', ''))
    print("  (normalized: %s)" % str(problem.normalized.a).replace('\n', ''))
    print_solution("Isolated", problem_test, solve_isolated(problem))
    print_solution("Unnormalized max throughput", problem_test,
                   solve_max_throughput(problem, False))
    print_solution("Max throughput", problem_test, solve_max_throughput(problem))
    print_solution("Isolated max throughput", problem_test,
                   solve_isolated_max_throughput(problem))
    print_solution("Nash bargaining", problem_test, solve_nash(problem))
    print_solution("Kalai-Smorodinsky", problem_test, solve_ks(problem))
    print()


def print_solution(name, problem, solution):
    """
    Pretty-print and analyze a problem solution.
    """
    print("%s solution:" % name)
    print("  assignments: %s" % str(solution.x).replace('\n', ''))
    rates = problem.user_rates(solution)
    print("  user rates: %s (total: %.3g)" % (rates, rates.sum()))
    norm_rates = problem.normalized_user_rates(solution)
    print("  normalized rates: %s (total: %.3g)" % (norm_rates, norm_rates.sum()))


def main():
    np.set_printoptions(precision=4, suppress=True)
    explore_problem([[1., 2.], [2., 1.]])
    explore_problem([[1., 2.], [1., 1.]])
    explore_problem([[1., 2.], [10., 10.]])
    explore_problem([[1., 2.], [2., 1.], [1., 1.]])
    explore_problem([[3., 1.], [2.5, 1.], [2., 1.]])

    print("=====================================================\n")
    print("Example showing lack of strategy-proofness of isolated max throughput ",
          "(P3 gets more by faking their demand):\n")
    a = [[2., 1.], [1., 1.], [1., 1.]]
    a_fake = [[2., 1.], [1., 1.], [1., 3.]]
    explore_problem(a)
    explore_problem(a_fake, a_test=a)

    print("=====================================================\n")
    print("Example showing lack of strategy-proofness of Nash bargaining ",
          "(P4 gets more by faking their demand):\n")
    a = [[1., 1., 8.], [2., 4., 1.], [8., 1., 1.], [1., 2., 1.]]
    a_fake = [[1., 1., 8.], [2., 4., 1.], [8., 1., 1.], [1., 1.5, 1.]]
    explore_problem(a)
    explore_problem(a_fake, a_test=a)

    print("=====================================================\n")
    print("Example showing lack of strategy-proofness of Kalai-Smorodinsky ",
          "(P2 gets more by faking their demand):\n")
    a = [[1., 1.], [1., 3.]]
    a_fake = [[1., 1.], [1., 2.]]
    explore_problem(a)
    explore_problem(a_fake, a_test=a)


if __name__ == "__main__":
    main()
