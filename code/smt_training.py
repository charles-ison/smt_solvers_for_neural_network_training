from pysmt.shortcuts import Symbol, And, Real, LT, GT, Max, Plus, get_model
from pysmt.typing import REAL

def train_smt(model, all_data, labels):
    
    layers_weights = []
    num_layers = len(model.linear_layers)
    for i in range(num_layers):
        num_rows = len(model.linear_layers[i].weight.data)
        num_cols = len(model.linear_layers[i].weight.data[0])
        layer_weights = []
        for j in range(num_rows):
            weights = []
            for k in range(num_cols):
                weights.append(Symbol("weight_" + str(i) + "_" + str(j) + "_" + str(k), REAL))
            weights.append(Symbol(str("bias_" + str(i) + "_" + str(j)), REAL))
            layer_weights.append(weights)
        layers_weights.append(layer_weights)

    equations = []
    for (data, label) in zip(all_data, labels):
        x = []
        for value in data:
            x.append(Real(value.item()))
            
        for i in range(num_layers):
            num_rows = len(model.linear_layers[i].weight.data)
            num_cols = len(model.linear_layers[i].weight.data[0])
            new_x = []
            for j in range(num_rows):
                calculations = []
                for k in range(num_cols):
                    calculation = layers_weights[i][j][k] * x[k]
                    calculations.append(calculation)
                calculations.append(layers_weights[i][j][num_cols])
                output = Plus(calculations)
                if i < num_layers-1:
                    output = Max(output, Real(0.0))
                new_x.append(output)
            x = new_x
    
        if label.item() == 0:
            equation = GT(x[0], x[1])
        else:
            equation = LT(x[0], x[1])
            
        equations.append(equation)

    formula = And(equations)
    print("formula.size(): ", formula.size())
    solution = get_model(formula)
    
    if solution:
        for i in range(num_layers):
            num_rows = len(model.linear_layers[i].weight.data)
            num_cols = len(model.linear_layers[i].weight.data[0])
            for j in range(num_rows):
                for k in range(num_cols):
                    weight = layers_weights[i][j][k]
                    weight_solution = solution[weight].constant_value()
                    model.linear_layers[i].weight.data[j][k] = float(weight_solution)
    
                bias = layers_weights[i][j][num_cols]
                bias_solution = solution[bias].constant_value()
                model.linear_layers[i].bias.data[j] = float(bias_solution)

    else:
        print("No solution found")
