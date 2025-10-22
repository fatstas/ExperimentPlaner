import sqlite3
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


class ExperimentModel:
    def __init__(self):
        self.current_model = None
        self.scaler = None
        self.param_names = []

    def generate_box_behnken(self, num_factors):
        """Генерация матрицы плана Бокса-Бенкина"""
        if num_factors == 3:
            return [
                [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
                [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 0, 1],
                [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],
                [0, 0, 0], [0, 0, 0], [0, 0, 0]
            ]
        else:
            return self._generate_simple_design(num_factors)

    def _generate_simple_design(self, num_factors):
        """Упрощенная генерация плана для большего числа факторов"""
        design = []
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                # Все комбинации -1/-1, 1/-1, -1/1, 1/1
                for x in [-1, 1]:
                    for y in [-1, 1]:
                        point = [0] * num_factors
                        point[i] = x
                        point[j] = y
                        design.append(point)

        # Центральные точки
        for _ in range(max(3, num_factors)):
            design.append([0] * num_factors)

        return design

    def scale_design(self, design_matrix, parameters):
        """Масштабирование значений согласно границам"""
        scaled_design = []
        param_names = list(parameters.keys())

        for run in design_matrix:
            scaled_run = {}
            for i, param_name in enumerate(param_names):
                coded_value = run[i]
                low = parameters[param_name]['low']
                high = parameters[param_name]['high']

                if coded_value == -1:
                    real_value = low
                elif coded_value == 1:
                    real_value = high
                else:  # 0
                    real_value = (low + high) / 2

                scaled_run[param_name] = real_value
            scaled_design.append(scaled_run)

        return scaled_design, param_names