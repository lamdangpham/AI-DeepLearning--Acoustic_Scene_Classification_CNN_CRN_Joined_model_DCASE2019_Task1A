class dnn_bl02_para(object):

    def __init__(self):

        #=======Layer 01: full connection
        self.l01_fc             = 1024 # node number of first full-connected layer 
        self.l01_is_act         = True
        self.l01_act_func       = 'RELU'
        self.l01_is_drop        = True
        self.l01_drop_prob      = 0.2

        #=======Layer 02: Final layer
        self.l02_fc             = 10   # output node number = class numbe
        self.l02_is_act         = False
        self.l02_act_func       = 'RELU'
        self.l02_is_drop        = False
        self.l02_drop_prob      = 1

