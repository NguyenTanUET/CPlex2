import os
from docplex.mp.model import Model
from docplex.mp.environment import Environment

# 1) Kiểm tra biến môi trường
if "CPLEX_STUDIO_DIR" in os.environ:
    print("CPLEX_STUDIO_DIR =", os.environ["CPLEX_STUDIO_DIR"])
else:
    print("CPLEX_STUDIO_DIR chưa được set; docplex sẽ dùng Community Edition")

# 2) Kiểm tra thông tin ngay trong Environment của docplex
env = Environment()
print("  • CPLEX available local?  ", env.cplex_location is not None)
print("  • CPLEX module importable?  ", env.get_cplex_module() is not None)

# 3) Thử giải một mô hình nhỏ để xem engine nào được dùng
m = Model(name="check_full_vs_ce")
x = m.continuous_var(name="x", ub=10)
m.maximize(x)
sol = m.solve(log_output=False)
if sol:
    details = m.get_solve_details()
    print("  • Solver used:        ", details.solver_name)
    print("  • Objective value:    ", sol.get_objective_value())
else:
    print("Không tìm được giải pháp (lỗi setup CPLEX)")
