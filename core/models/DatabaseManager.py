import sqlite3
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


class DatabaseManager:
    def __init__(self, db_path='DATA/adaptive_experiments.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Инициализация базы данных SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_date TEXT NOT NULL,
                parameters TEXT NOT NULL,
                description TEXT,
                adaptive_mode BOOLEAN DEFAULT 0,
                target_type TEXT DEFAULT 'maximize'
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                run_order INTEGER,
                parameters_values TEXT NOT NULL,
                results TEXT,
                notes TEXT,
                is_suggested BOOLEAN DEFAULT 0,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')

        conn.commit()
        conn.close()

    def get_connection(self):
        return sqlite3.connect(self.db_path)