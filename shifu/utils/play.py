import pybullet as p


class ControlPanel:
    def __init__(self, num_parameters):
        p.connect(p.GUI, options=f"--width=500 --height={200 + num_parameters * 60}")
        self._parm_dict = {}

    def add_parameter(self, name, x_min, x_max, x_start=None):
        assert name not in self._parm_dict.keys()
        if not x_start:
            x_start = (x_max - x_min) / 2 + x_min
        self._parm_dict[name] = p.addUserDebugParameter(name, x_min, x_max, x_start)

    def get_value(self, name):
        return p.readUserDebugParameter(self._parm_dict[name])
