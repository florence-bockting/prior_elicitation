import pandas as pd

def write_results(global_dict):
    f = open(global_dict["output_path"]["data"]+"\overview.txt", "w")
    
    df = pd.DataFrame()
    df["target quantities"] = global_dict["target_quantities"]["name"]
    df["elicitation technique"] = global_dict["target_quantities"]["elicitation_method"]
    df["combine-loss"] = global_dict["target_quantities"]["loss_components"]
    
    loss_comp = pd.read_pickle(global_dict["output_path"]["data"]+"/loss_components.pkl")
    
    df2 = pd.DataFrame()
    df2["loss components"] = list(loss_comp.keys())
    df2["shape"] = [list(loss_comp[key].shape) for key in list(loss_comp.keys())]
    
    f.write("Target quantities and elicitation techniques"+
            "\n--------------------- \n"+
            f"\n{df}\n"+"\nLoss components"+
            "\n--------------------- \n"+
            f"\n{df2}")
    f.close()

