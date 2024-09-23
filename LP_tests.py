import pulp
import pandas as pd

 

def read_graph_from_csv(file_name):

    df = pd.read_csv(file_name)
    edges = []

    for index, row in df.iterrows():
        edges.append((row['node 1'], row['node 2'], row['distance'], row['capacity'], row['saturation']))

    return edges

def solve_linear_program(edges):
    
    prob = pulp.LpProblem("Minimize_traffic_cost", pulp.LpMinimize)
    x = {}
    for u, v, dist, cap, sat in edges:
        x[(u, v)] = pulp.LpVariable(f"x_{u}_{v}", cat='Binary')

    sat_weight = 0.5
    dist_weight = 0.5

    prob += pulp.lpSum((dist * dist_weight + sat_weight * (sat / cap)) * x[(u, v)] for u, v, dist, cap, sat in edges), "Total Cost"

    nodes = set()


    for u, v, _, _, _ in edges:
        nodes.add(u)
        nodes.add(v)



    for node in nodes:
        in_flow = pulp.lpSum(x[(u, v)] for u, v, _, _, _ in edges if v == node)

        out_flow = pulp.lpSum(x[(u, v)] for u, v, _, _, _ in edges if u == node)

        if node == 0:  
            prob += out_flow == 1, f"Flow_Node_{node}"

        elif node == max(nodes):  
            prob += in_flow == 1, f"Flow_Node_{node}"

            
        else: 
            prob += in_flow == out_flow, f"Flow_Node_{node}"

    prob.solve()

    solution_status = pulp.LpStatus[prob.status]
    variable_values = {f"x_{u}_{v}": pulp.value(x[(u, v)]) for u, v in x}
    objective_value = pulp.value(prob.objective)

    return solution_status, variable_values, objective_value

def main():


    file_name = input("Veuillez entrer le nom du fichier: ") 



    edges = read_graph_from_csv(file_name)
    status, values, obj_value = solve_linear_program(edges)
    
    print("Status:", status)
    
    print("\nVariable Values:")

    for var, value in values.items():
        print(f"{var}: {value}\n")

    
    print("Objective Value:", obj_value)




if __name__ == "__main__":
    main()

