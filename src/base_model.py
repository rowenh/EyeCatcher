# Copyright 2023 Rowen Horbach, Eline R. de Groot, Jeroen Dudink, Ronald Poppe.

# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with 
# this program. If not, see <https://www.gnu.org/licenses/>.

class BaseModel():
    def __init__(self, dir_):
        self.dir_ = dir_
        self.write_dir = dir_ + "models/last/"
        self.read_dir = dir_ + "models/active/"

    # Abstract
    def load_model(self):
        pass

    # Abstract
    def get_samples(self):
        pass

    # Abstract
    def train(self, train_set, val_set, test_set):
        pass
