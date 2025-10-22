import tkinter as tk
from tkinter import ttk, messagebox
from core.models import DatabaseManager, ExperimentModel
from core.gui.gui import MainWindow
from core.services import ExperimentService, PredictionService


class AdaptiveExperimentDesignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Box-Behnken Experiment Designer")
        self.root.geometry("1200x800")

        # Инициализация компонентов
        self.db_manager = DatabaseManager()
        self.experiment_model = ExperimentModel()
        self.experiment_service = ExperimentService(self.db_manager)
        self.prediction_service = PredictionService()

        # Создание интерфейса
        self.gui = MainWindow(
            root,
            self.experiment_service,
            self.prediction_service,
            self.db_manager
        )
