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

import argparse

import src.model_helper
import src.base_model
import visibility_model.model
import open_model.model
import rem_model.model
import sleep_model.model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_set", help="Identify samples for train set with name constraints. Use '+' to separate constraints.")
parser.add_argument("--val_set", help="Identify samples for val set with name constraints. Use '+' to separate constraints.")
parser.add_argument("--test_set", help="Identify samples for test set with name constraints. Use '+' to separate constraints.")
parser.add_argument("--folds", help="Identify folds for cross validation with name constraints. Use '+' to separate constraints; use '=' to separate folds.")
args, unknown = parser.parse_known_args()
config = vars(args)

if len(unknown) > 1:
    print("Unexpected arguments!")
else:
    target_model = unknown[0]

    m = None
    if target_model == "visibility":
        m = visibility_model.model.VisibilityModel()
    elif target_model == "open":
        m = open_model.model.OpenModel()
    elif target_model == "rem":
        m = rem_model.model.RemModel()
    elif target_model == "sleep":
        m = sleep_model.model.SleepModel()
    src.model_helper.run(m, config)
