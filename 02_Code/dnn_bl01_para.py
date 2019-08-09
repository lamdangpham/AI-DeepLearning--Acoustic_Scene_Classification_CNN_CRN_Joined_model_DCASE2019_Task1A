import numpy as np
import os

class dnn_bl01_para(object):

    def __init__(self):

        #=======Layer 05: full connection
        self.l01_fc             = 512 # node number of first full-connected layer 
        self.l01_is_act         = True
        self.l01_act_func       = 'RELU'
        self.l01_is_drop        = True
        self.l01_drop_prob      = 0.2

        #=======Layer 06: full connection
        self.l02_fc             = 1024  # node number of first full-connected layer 
        self.l02_is_act         = True
        self.l02_act_func       = 'RELU'
        self.l02_is_drop        = True
        self.l02_drop_prob      = 0.2

        #=======Layer 07: Final layer
        self.l03_fc             = 256   # output node number = class numbe
        self.l03_is_act         = True
        self.l03_act_func       = 'RELU'
        self.l03_is_drop        = True
        self.l03_drop_prob      = 0.2

