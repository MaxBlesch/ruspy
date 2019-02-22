import yaml
import pickle
from simulation.simulation import simulate


with open('init.yml') as y:
    init_dict = yaml.load(y)
pickle.dump(simulate(init_dict), open('sim_file.pkl', 'wb'))