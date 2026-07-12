

import os
from circle_dataset import get_circle_dataset
from utils.face_torch  import FNNModule
from utils.face_torch  import count_configurations
from utils.set_random  import set_random_seed
from utils.graph       import ImplicitEquationPlotter
from utils.gface_torch import get_rules
from utils.print_texts import print_bold
from utils.ineq import get_face_contrib_accelerated

print("Hola")
