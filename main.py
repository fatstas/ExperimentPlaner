import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class AdaptiveExperimentDesignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Box-Behnken Experiment Designer")
        self.root.geometry("1200x800")

        # Инициализация БД
        self.init_database()

        # Текущая модель для предсказаний
        self.current_model = None
        self.scaler = None
        self.param_names = []

        # Создание интерфейса
        self.create_widgets()

        # Загрузка списка экспериментов
        self.load_experiments_list()

    def init_database(self):
        """Инициализация базы данных SQLite"""
        self.conn = sqlite3.connect('adaptive_experiments.db')
        self.cursor = self.conn.cursor()

        # Таблица для хранения метаданных экспериментов
        self.cursor.execute('''
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

        # Таблица для хранения результатов экспериментов
        self.cursor.execute('''
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

        self.conn.commit()

    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Конфигурация весов для растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Левая панель - управление экспериментами
        left_frame = ttk.LabelFrame(main_frame, text="Управление экспериментами", padding="10")
        left_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Кнопки управления
        ttk.Button(left_frame, text="Новый эксперимент",
                   command=self.create_new_experiment).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Удалить эксперимент",
                   command=self.delete_experiment).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Загрузить эксперимент",
                   command=self.load_experiment).pack(fill=tk.X, pady=5)

        # Настройки адаптивного режима
        adaptive_frame = ttk.LabelFrame(left_frame, text="Адаптивный режим", padding="5")
        adaptive_frame.pack(fill=tk.X, pady=10)

        self.adaptive_var = tk.BooleanVar()
        ttk.Checkbutton(adaptive_frame, text="Адаптивный режим",
                        variable=self.adaptive_var).pack(anchor=tk.W)

        ttk.Label(adaptive_frame, text="Цель оптимизации:").pack(anchor=tk.W, pady=(5, 0))
        self.target_var = tk.StringVar(value="maximize")
        ttk.Radiobutton(adaptive_frame, text="Максимизация",
                        variable=self.target_var, value="maximize").pack(anchor=tk.W)
        ttk.Radiobutton(adaptive_frame, text="Минимизация",
                        variable=self.target_var, value="minimize").pack(anchor=tk.W)
        ttk.Radiobutton(adaptive_frame, text="Целевое значение",
                        variable=self.target_var, value="target").pack(anchor=tk.W)

        self.target_value_var = tk.StringVar(value="0")
        ttk.Entry(adaptive_frame, textvariable=self.target_value_var, width=10).pack(anchor=tk.W, pady=5)

        # Список экспериментов
        ttk.Label(left_frame, text="Список экспериментов:").pack(anchor=tk.W, pady=(10, 5))
        self.experiments_listbox = tk.Listbox(left_frame, height=12)
        self.experiments_listbox.pack(fill=tk.BOTH, expand=True)
        self.experiments_listbox.bind('<<ListboxSelect>>', self.on_experiment_select)

        # Правая панель - параметры эксперимента
        right_frame = ttk.LabelFrame(main_frame, text="Параметры эксперимента", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        right_frame.columnconfigure(1, weight=1)

        # Название эксперимента
        ttk.Label(right_frame, text="Название:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.exp_name_var = tk.StringVar()
        self.exp_name_entry = ttk.Entry(right_frame, textvariable=self.exp_name_var)
        self.exp_name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))

        # Описание
        ttk.Label(right_frame, text="Описание:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.exp_desc_text = tk.Text(right_frame, height=3, width=40)
        self.exp_desc_text.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))

        # Фрейм для параметров
        params_frame = ttk.Frame(right_frame)
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        params_frame.columnconfigure(1, weight=1)

        ttk.Label(params_frame, text="Параметры эксперимента:").grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Таблица параметров
        self.params_tree = ttk.Treeview(params_frame, columns=('name', 'low', 'high'),
                                        show='headings', height=6)
        self.params_tree.heading('name', text='Название')
        self.params_tree.heading('low', text='Нижняя граница')
        self.params_tree.heading('high', text='Верхняя граница')
        self.params_tree.column('name', width=150)
        self.params_tree.column('low', width=100)
        self.params_tree.column('high', width=100)
        self.params_tree.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Кнопки управления параметрами
        ttk.Button(params_frame, text="Добавить параметр",
                   command=self.add_parameter).grid(row=2, column=0, pady=5)
        ttk.Button(params_frame, text="Удалить параметр",
                   command=self.delete_parameter).grid(row=2, column=1, pady=5)
        ttk.Button(params_frame, text="Редактировать параметр",
                   command=self.edit_parameter).grid(row=2, column=2, pady=5)

        # Кнопки генерации
        button_frame = ttk.Frame(right_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Сгенерировать базовый план",
                   command=self.generate_initial_design).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Предложить следующий опыт",
                   command=self.suggest_next_experiment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Обновить модель",
                   command=self.update_prediction_model).pack(side=tk.LEFT, padx=5)

        # Нижняя панель - результаты
        results_frame = ttk.LabelFrame(main_frame, text="План эксперимента и результаты", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Таблица результатов
        self.results_tree = ttk.Treeview(results_frame, height=15)
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar для таблицы результатов
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Кнопки для работы с результатами
        results_buttons_frame = ttk.Frame(results_frame)
        results_buttons_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(results_buttons_frame, text="Добавить результат",
                   command=self.add_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_buttons_frame, text="Редактировать результат",
                   command=self.edit_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_buttons_frame, text="Экспорт в CSV",
                   command=self.export_to_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_buttons_frame, text="Анализ зависимости",
                   command=self.show_dependency_analysis).pack(side=tk.LEFT, padx=5)

    def update_prediction_model(self):
        """Обновление модели машинного обучения на основе имеющихся данных"""
        if not self.has_completed_experiments():
            messagebox.showwarning("Внимание", "Недостаточно данных для построения модели. Проведите хотя бы 3 опыта.")
            return

        try:
            # Получаем данные из БД
            X, y = self.get_training_data()

            if len(X) < 3:
                messagebox.showwarning("Внимание", "Недостаточно данных для построения модели.")
                return

            # Масштабируем данные
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Обучаем модель (используем Random Forest как более robust метод)
            self.current_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.current_model.fit(X_scaled, y)

            # Рассчитываем точность модели на имеющихся данных
            score = self.current_model.score(X_scaled, y)

            messagebox.showinfo("Модель обновлена",
                                f"Модель успешно обучена!\nR² score: {score:.3f}\n"
                                f"Обучено на {len(X)} опытах.")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обучении модели: {str(e)}")

    def get_training_data(self):
        """Получение данных для обучения модели"""
        experiment_name = self.exp_name_var.get()

        self.cursor.execute('''
            SELECT er.parameters_values, er.results
            FROM experiment_results er
            JOIN experiments e ON e.id = er.experiment_id
            WHERE e.name = ? AND er.results IS NOT NULL AND er.results != ''
        ''', (experiment_name,))

        results = self.cursor.fetchall()

        X = []
        y = []

        for params_json, result in results:
            try:
                params = json.loads(params_json)
                # Сохраняем порядок параметров
                if not self.param_names:
                    self.param_names = list(params.keys())

                x_vector = [params[param] for param in self.param_names]
                X.append(x_vector)
                y.append(float(result))
            except (json.JSONDecodeError, ValueError) as e:
                continue

        return np.array(X), np.array(y)

    def suggest_next_experiment(self):
        """Предложение следующего опыта на основе модели"""
        if self.current_model is None:
            messagebox.showwarning("Внимание", "Сначала обновите модель на основе имеющихся данных!")
            return

        if not self.adaptive_var.get():
            messagebox.showwarning("Внимание", "Включите адаптивный режим для использования этой функции!")
            return

        try:
            # Получаем границы параметров
            bounds = self.get_parameter_bounds()

            # Целевая функция для оптимизации
            def objective(x):
                x_scaled = self.scaler.transform([x])
                prediction = self.current_model.predict(x_scaled)[0]

                target_type = self.target_var.get()
                if target_type == 'maximize':
                    return -prediction  # Минимизируем отрицательное значение
                elif target_type == 'minimize':
                    return prediction
                else:  # target
                    target_val = float(self.target_value_var.get())
                    return abs(prediction - target_val)

            # Начальная точка - центр области
            x0 = [(b[0] + b[1]) / 2 for b in bounds]

            # Ограничения
            constraints = []
            for i, (low, high) in enumerate(bounds):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, l=low: x[i] - l})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, h=high: h - x[i]})

            # Оптимизация
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                suggested_params = result.x

                # Создаем словарь параметров
                param_dict = {name: float(value) for name, value in zip(self.param_names, suggested_params)}

                # Показываем предложение
                self.show_suggestion_dialog(param_dict)
            else:
                messagebox.showerror("Ошибка", "Не удалось найти оптимальные параметры. Попробуйте другие настройки.")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при генерации предложения: {str(e)}")

    def show_suggestion_dialog(self, suggested_params):
        """Диалог показа предложенных параметров"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Предложенный следующий опыт")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Система предлагает следующий набор параметров:",
                  font=('Arial', 10, 'bold')).pack(pady=10)

        # Таблица параметров
        tree = ttk.Treeview(dialog, columns=('parameter', 'value'), show='headings', height=8)
        tree.heading('parameter', text='Параметр')
        tree.heading('value', text='Значение')
        tree.column('parameter', width=200)
        tree.column('value', width=150)
        tree.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        for param, value in suggested_params.items():
            tree.insert('', tk.END, values=(param, f"{value:.4f}"))

        # Предсказание результата
        try:
            X = np.array([list(suggested_params.values())])
            X_scaled = self.scaler.transform(X)
            predicted_result = self.current_model.predict(X_scaled)[0]

            ttk.Label(dialog, text=f"Предсказанный результат: {predicted_result:.4f}",
                      font=('Arial', 9)).pack(pady=5)
        except:
            pass

        def add_to_plan():
            self.add_suggested_experiment(suggested_params)
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Добавить в план",
                   command=add_to_plan).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Отмена",
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def add_suggested_experiment(self, suggested_params):
        """Добавление предложенного эксперимента в план"""
        experiment_name = self.exp_name_var.get()

        # Получаем следующий номер опыта
        self.cursor.execute('''
            SELECT MAX(run_order) FROM experiment_results 
            WHERE experiment_id = (SELECT id FROM experiments WHERE name = ?)
        ''', (experiment_name,))

        max_order = self.cursor.fetchone()[0] or 0
        new_order = max_order + 1

        # Сохраняем в БД
        try:
            self.cursor.execute('''
                INSERT INTO experiment_results (experiment_id, run_order, parameters_values, is_suggested)
                VALUES (
                    (SELECT id FROM experiments WHERE name = ?),
                    ?, ?, 1
                )
            ''', (experiment_name, new_order, json.dumps(suggested_params)))

            self.conn.commit()

            # Обновляем отображение
            self.load_experiment_results(experiment_name)

            messagebox.showinfo("Успех", "Предложенный опыт добавлен в план эксперимента!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при добавлении опыта: {str(e)}")

    def get_parameter_bounds(self):
        """Получение границ параметров"""
        bounds = []
        for item in self.params_tree.get_children():
            values = self.params_tree.item(item, 'values')
            if len(values) == 3:
                bounds.append((float(values[1]), float(values[2])))
        return bounds

    def has_completed_experiments(self):
        """Проверка наличия завершенных опытов с результатами"""
        experiment_name = self.exp_name_var.get()

        self.cursor.execute('''
            SELECT COUNT(*) FROM experiment_results er
            JOIN experiments e ON e.id = er.experiment_id
            WHERE e.name = ? AND er.results IS NOT NULL AND er.results != ''
        ''', (experiment_name,))

        return self.cursor.fetchone()[0] > 0

    def show_dependency_analysis(self):
        """Показать анализ зависимостей"""
        if not self.has_completed_experiments():
            messagebox.showwarning("Внимание", "Недостаточно данных для анализа.")
            return

        try:
            X, y = self.get_training_data()

            if len(X) < 3:
                messagebox.showwarning("Внимание", "Недостаточно данных для анализа.")
                return

            # Создаем окно анализа
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Анализ зависимостей")
            analysis_window.geometry("600x400")

            # Анализ важности признаков
            if self.current_model and hasattr(self.current_model, 'feature_importances_'):
                importance_frame = ttk.LabelFrame(analysis_window, text="Важность параметров", padding="10")
                importance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                importances = self.current_model.feature_importances_
                features = self.param_names

                tree = ttk.Treeview(importance_frame, columns=('parameter', 'importance'), show='headings')
                tree.heading('parameter', text='Параметр')
                tree.heading('importance', text='Важность')
                tree.column('parameter', width=200)
                tree.column('importance', width=150)
                tree.pack(fill=tk.BOTH, expand=True)

                for feature, importance in zip(features, importances):
                    tree.insert('', tk.END, values=(feature, f"{importance:.4f}"))

                ttk.Label(importance_frame,
                          text="Чем выше значение, тем сильнее параметр влияет на результат",
                          font=('Arial', 8)).pack(pady=5)

            else:
                # Линейная регрессия для анализа
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)

                coef_frame = ttk.LabelFrame(analysis_window, text="Коэффициенты линейной модели", padding="10")
                coef_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                tree = ttk.Treeview(coef_frame, columns=('parameter', 'coefficient'), show='headings')
                tree.heading('parameter', text='Параметр')
                tree.heading('coefficient', text='Коэффициент')
                tree.column('parameter', width=200)
                tree.column('coefficient', width=150)
                tree.pack(fill=tk.BOTH, expand=True)

                for feature, coef in zip(self.param_names, model.coef_):
                    tree.insert('', tk.END, values=(feature, f"{coef:.4f}"))

                ttk.Label(coef_frame,
                          text="Положительный коэффициент: увеличение параметра увеличивает результат\n"
                               "Отрицательный коэффициент: увеличение параметра уменьшает результат",
                          font=('Arial', 8)).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе: {str(e)}")

    def generate_initial_design(self):
        """Генерация начального плана Бокса-Бенкина"""
        if not self.save_experiment_metadata():
            return

        # Получаем параметры
        parameters = {}
        param_names = []
        for item in self.params_tree.get_children():
            values = self.params_tree.item(item, 'values')
            if len(values) == 3:
                param_name = values[0]
                parameters[param_name] = {
                    'low': float(values[1]),
                    'high': float(values[2])
                }
                param_names.append(param_name)

        if len(param_names) < 3:
            messagebox.showerror("Ошибка", "Для плана Бокса-Бенкина нужно минимум 3 параметра!")
            return

        # Генерируем план
        try:
            design_matrix = self.generate_box_behnken(len(param_names))

            # Масштабируем значения согласно границам
            scaled_design = []
            for run in design_matrix:
                scaled_run = {}
                for i, param_name in enumerate(param_names):
                    # Преобразование из [-1, 0, 1] в реальные значения
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

            # Отображаем план
            self.display_design(scaled_design, param_names)

            # Сохраняем план в БД
            self.save_design_to_db(scaled_design)

            messagebox.showinfo("Успех", "Базовый план эксперимента успешно сгенерирован!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при генерации плана: {str(e)}")

    # Остальные методы (generate_box_behnken, save_experiment_metadata, display_design,
    # save_design_to_db, load_experiments_list, create_new_experiment, add_parameter,
    # edit_parameter, delete_parameter, load_experiment, add_result, edit_result,
    # export_to_csv, delete_experiment) остаются аналогичными предыдущей версии

    def generate_box_behnken(self, num_factors):
        """Генерация матрицы плана Бокса-Бенкина"""
        if num_factors == 3:
            base_design = [
                [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
                [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 0, 1],
                [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],
                [0, 0, 0], [0, 0, 0], [0, 0, 0]
            ]
        else:
            base_design = self.generate_simple_design(num_factors)
        return base_design

    def generate_simple_design(self, num_factors):
        """Упрощенная генерация плана для большего числа факторов"""
        design = []
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                point = [0] * num_factors
                point[i] = -1
                point[j] = -1
                design.append(point)

                point = [0] * num_factors
                point[i] = 1
                point[j] = -1
                design.append(point)

                point = [0] * num_factors
                point[i] = -1
                point[j] = 1
                design.append(point)

                point = [0] * num_factors
                point[i] = 1
                point[j] = 1
                design.append(point)

        for _ in range(max(3, num_factors)):
            design.append([0] * num_factors)

        return design

    def save_experiment_metadata(self):
        """Сохранение метаданных эксперимента в БД"""
        name = self.exp_name_var.get().strip()
        if not name:
            messagebox.showerror("Ошибка", "Введите название эксперимента!")
            return False

        parameters = {}
        for item in self.params_tree.get_children():
            values = self.params_tree.item(item, 'values')
            if len(values) == 3:
                parameters[values[0]] = {'low': float(values[1]), 'high': float(values[2])}

        if not parameters:
            messagebox.showerror("Ошибка", "Добавьте хотя бы один параметр!")
            return False

        description = self.exp_desc_text.get(1.0, tk.END).strip()
        adaptive_mode = self.adaptive_var.get()
        target_type = self.target_var.get()

        try:
            self.cursor.execute("SELECT id FROM experiments WHERE name = ?", (name,))
            existing = self.cursor.fetchone()

            if existing:
                self.cursor.execute('''
                    UPDATE experiments 
                    SET parameters = ?, description = ?, adaptive_mode = ?, target_type = ?
                    WHERE name = ?
                ''', (json.dumps(parameters), description, adaptive_mode, target_type, name))
            else:
                self.cursor.execute('''
                    INSERT INTO experiments (name, created_date, parameters, description, adaptive_mode, target_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                name, datetime.now().isoformat(), json.dumps(parameters), description, adaptive_mode, target_type))

            self.conn.commit()
            self.load_experiments_list()
            return True

        except Exception as e:
            messagebox.showerror("Ошибка БД", f"Ошибка при сохранении: {str(e)}")
            return False

    def load_experiments_list(self):
        """Загрузка списка экспериментов из БД"""
        self.experiments_listbox.delete(0, tk.END)
        self.cursor.execute("SELECT name FROM experiments ORDER BY created_date DESC")
        experiments = self.cursor.fetchall()
        for exp in experiments:
            self.experiments_listbox.insert(tk.END, exp[0])

    def create_new_experiment(self):
        """Создание нового эксперимента"""
        name = simpledialog.askstring("Новый эксперимент", "Введите название эксперимента:")
        if name:
            if self.check_experiment_exists(name):
                messagebox.showerror("Ошибка", "Эксперимент с таким названием уже существует!")
                return

            # Очистка полей
            self.exp_name_var.set(name)
            self.exp_desc_text.delete(1.0, tk.END)
            self.clear_parameters()
            self.clear_results()

            messagebox.showinfo("Успех", f"Создан новый эксперимент: {name}\nТеперь добавьте параметры.")

    def check_experiment_exists(self, name):
        """Проверка существования эксперимента"""
        self.cursor.execute("SELECT id FROM experiments WHERE name = ?", (name,))
        return self.cursor.fetchone() is not None

    def clear_parameters(self):
        """Очистка таблицы параметров"""
        for item in self.params_tree.get_children():
            self.params_tree.delete(item)

    def clear_results(self):
        """Очистка таблицы результатов"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def add_parameter(self):
        """Добавление нового параметра"""
        dialog = ParameterDialog(self.root, "Добавить параметр")
        if dialog.result:
            name, low, high = dialog.result
            self.params_tree.insert('', tk.END, values=(name, low, high))

    def edit_parameter(self):
        """Редактирование выбранного параметра"""
        selection = self.params_tree.selection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите параметр для редактирования!")
            return

        item = selection[0]
        current_values = self.params_tree.item(item, 'values')

        dialog = ParameterDialog(self.root, "Редактировать параметр", current_values)
        if dialog.result:
            name, low, high = dialog.result
            self.params_tree.item(item, values=(name, low, high))

    def delete_parameter(self):
        """Удаление выбранного параметра"""
        selection = self.params_tree.selection()
        if selection:
            self.params_tree.delete(selection)

    def save_experiment_metadata(self):
        """Сохранение метаданных эксперимента в БД"""
        name = self.exp_name_var.get().strip()
        if not name:
            messagebox.showerror("Ошибка", "Введите название эксперимента!")
            return False

        # Сбор параметров
        parameters = {}
        for item in self.params_tree.get_children():
            values = self.params_tree.item(item, 'values')
            if len(values) == 3:
                parameters[values[0]] = {'low': float(values[1]), 'high': float(values[2])}

        if not parameters:
            messagebox.showerror("Ошибка", "Добавьте хотя бы один параметр!")
            return False

        description = self.exp_desc_text.get(1.0, tk.END).strip()

        try:
            # Проверяем, существует ли уже эксперимент
            self.cursor.execute("SELECT id FROM experiments WHERE name = ?", (name,))
            existing = self.cursor.fetchone()

            if existing:
                # Обновляем существующий
                self.cursor.execute('''
                    UPDATE experiments 
                    SET parameters = ?, description = ?
                    WHERE name = ?
                ''', (json.dumps(parameters), description, name))
            else:
                # Создаем новый
                self.cursor.execute('''
                    INSERT INTO experiments (name, created_date, parameters, description)
                    VALUES (?, ?, ?, ?)
                ''', (name, datetime.now().isoformat(), json.dumps(parameters), description))

            self.conn.commit()
            self.load_experiments_list()
            return True

        except Exception as e:
            messagebox.showerror("Ошибка БД", f"Ошибка при сохранении: {str(e)}")
            return False

    def generate_design(self):
        """Генерация плана Бокса-Бенкина"""
        if not self.save_experiment_metadata():
            return

        # Получаем параметры
        parameters = {}
        param_names = []
        for item in self.params_tree.get_children():
            values = self.params_tree.item(item, 'values')
            if len(values) == 3:
                param_name = values[0]
                parameters[param_name] = {
                    'low': float(values[1]),
                    'high': float(values[2])
                }
                param_names.append(param_name)

        if len(param_names) < 3:
            messagebox.showerror("Ошибка", "Для плана Бокса-Бенкина нужно минимум 3 параметра!")
            return

        # Генерируем план
        try:
            design_matrix = self.generate_box_behnken(len(param_names))

            # Масштабируем значения согласно границам
            scaled_design = []
            for run in design_matrix:
                scaled_run = {}
                for i, param_name in enumerate(param_names):
                    # Преобразование из [-1, 0, 1] в реальные значения
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

            # Отображаем план
            self.display_design(scaled_design, param_names)

            # Сохраняем план в БД
            self.save_design_to_db(scaled_design)

            messagebox.showinfo("Успех", "План эксперимента успешно сгенерирован!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при генерации плана: {str(e)}")

    def generate_box_behnken(self, num_factors):
        """Генерация матрицы плана Бокса-Бенкина"""
        # Базовая реализация для 3-5 факторов
        if num_factors == 3:
            # Базовый план для 3 факторов
            base_design = [
                [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
                [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 0, 1],
                [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],
                [0, 0, 0], [0, 0, 0], [0, 0, 0]  # Центральные точки
            ]
        elif num_factors == 4:
            # Упрощенный план для 4 факторов (можно расширить)
            base_design = []
            # Комбинации для 4 факторов
            combinations = [
                (-1, -1, 0, 0), (1, -1, 0, 0), (-1, 1, 0, 0), (1, 1, 0, 0),
                (-1, 0, -1, 0), (1, 0, -1, 0), (-1, 0, 1, 0), (1, 0, 1, 0),
                (-1, 0, 0, -1), (1, 0, 0, -1), (-1, 0, 0, 1), (1, 0, 0, 1),
                (0, -1, -1, 0), (0, 1, -1, 0), (0, -1, 1, 0), (0, 1, 1, 0),
                (0, -1, 0, -1), (0, 1, 0, -1), (0, -1, 0, 1), (0, 1, 0, 1),
                (0, 0, -1, -1), (0, 0, 1, -1), (0, 0, -1, 1), (0, 0, 1, 1)
            ]
            base_design = list(combinations)
            # Добавляем центральные точки
            for _ in range(3):
                base_design.append((0, 0, 0, 0))
        else:
            # Для большего числа факторов используем упрощенный подход
            messagebox.showwarning("Внимание",
                                   "Для более чем 4 факторов используется упрощенный план.\n"
                                   "Рекомендуется использовать специализированные библиотеки.")
            base_design = self.generate_simple_design(num_factors)

        return base_design

    def generate_simple_design(self, num_factors):
        """Упрощенная генерация плана для большего числа факторов"""
        design = []
        # Добавляем угловые точки (частично)
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                point = [0] * num_factors
                point[i] = -1
                point[j] = -1
                design.append(point)

                point = [0] * num_factors
                point[i] = 1
                point[j] = -1
                design.append(point)

                point = [0] * num_factors
                point[i] = -1
                point[j] = 1
                design.append(point)

                point = [0] * num_factors
                point[i] = 1
                point[j] = 1
                design.append(point)

        # Добавляем центральные точки
        for _ in range(max(3, num_factors)):
            design.append([0] * num_factors)

        return design

    def display_design(self, design, param_names):
        """Отображение плана в таблице"""
        self.clear_results()

        # Создаем колонки
        self.results_tree['columns'] = ['run_order'] + param_names + ['results', 'notes']
        self.results_tree.heading('run_order', text='№ опыта')
        self.results_tree.column('run_order', width=80)

        for param in param_names:
            self.results_tree.heading(param, text=param)
            self.results_tree.column(param, width=100)

        self.results_tree.heading('results', text='Результаты')
        self.results_tree.column('results', width=150)

        self.results_tree.heading('notes', text='Примечания')
        self.results_tree.column('notes', width=200)

        # Заполняем данными
        for i, run in enumerate(design, 1):
            values = [i] + [f"{run[param]:.4f}" for param in param_names] + ['', '']
            self.results_tree.insert('', tk.END, values=values, iid=str(i))

    def save_design_to_db(self, design):
        """Сохранение плана в базу данных"""
        experiment_name = self.exp_name_var.get()

        # Получаем ID эксперимента
        self.cursor.execute("SELECT id FROM experiments WHERE name = ?", (experiment_name,))
        result = self.cursor.fetchone()
        if not result:
            messagebox.showerror("Ошибка", "Эксперимент не найден в БД!")
            return

        experiment_id = result[0]

        try:
            # Удаляем старые результаты
            self.cursor.execute("DELETE FROM experiment_results WHERE experiment_id = ?", (experiment_id,))

            # Сохраняем новый план
            for i, run in enumerate(design, 1):
                self.cursor.execute('''
                    INSERT INTO experiment_results (experiment_id, run_order, parameters_values)
                    VALUES (?, ?, ?)
                ''', (experiment_id, i, json.dumps(run)))

            self.conn.commit()

        except Exception as e:
            messagebox.showerror("Ошибка БД", f"Ошибка при сохранении плана: {str(e)}")

    def on_experiment_select(self, event):
        """Обработка выбора эксперимента из списка"""
        selection = self.experiments_listbox.curselection()
        if selection:
            self.load_experiment()

    def load_experiment(self):
        """Загрузка выбранного эксперимента"""
        selection = self.experiments_listbox.curselection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите эксперимент из списка!")
            return

        experiment_name = self.experiments_listbox.get(selection[0])

        try:
            # Загружаем метаданные
            self.cursor.execute(
                "SELECT name, description, parameters FROM experiments WHERE name = ?",
                (experiment_name,)
            )
            result = self.cursor.fetchone()

            if result:
                name, description, parameters_json = result
                parameters = json.loads(parameters_json)

                # Заполняем поля
                self.exp_name_var.set(name)
                self.exp_desc_text.delete(1.0, tk.END)
                self.exp_desc_text.insert(1.0, description or "")

                # Заполняем параметры
                self.clear_parameters()
                for param_name, bounds in parameters.items():
                    self.params_tree.insert('', tk.END,
                                            values=(param_name, bounds['low'], bounds['high']))

                # Загружаем результаты
                self.load_experiment_results(name)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке эксперимента: {str(e)}")

    def load_experiment_results(self, experiment_name):
        """Загрузка результатов эксперимента"""
        try:
            self.cursor.execute('''
                SELECT er.run_order, er.parameters_values, er.results, er.notes
                FROM experiment_results er
                JOIN experiments e ON e.id = er.experiment_id
                WHERE e.name = ?
                ORDER BY er.run_order
            ''', (experiment_name,))

            results = self.cursor.fetchall()

            if results:
                # Получаем имена параметров из первого результата
                first_result = json.loads(results[0][1])
                param_names = list(first_result.keys())

                # Создаем таблицу
                self.display_results_table(results, param_names)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке результатов: {str(e)}")

    def display_results_table(self, results, param_names):
        """Отображение таблицы с результатами"""
        self.clear_results()

        # Создаем колонки
        columns = ['run_order'] + param_names + ['results', 'notes']
        self.results_tree['columns'] = columns

        # Настраиваем заголовки
        self.results_tree.heading('run_order', text='№ опыта')
        self.results_tree.column('run_order', width=80)

        for param in param_names:
            self.results_tree.heading(param, text=param)
            self.results_tree.column(param, width=100)

        self.results_tree.heading('results', text='Результаты')
        self.results_tree.column('results', width=150)

        self.results_tree.heading('notes', text='Примечания')
        self.results_tree.column('notes', width=200)

        # Заполняем данными
        for row in results:
            run_order, params_json, results_val, notes = row
            params = json.loads(params_json)

            values = [run_order] + [f"{params[param]:.4f}" for param in param_names]
            values.extend([results_val or '', notes or ''])

            self.results_tree.insert('', tk.END, values=values, iid=str(run_order))

    def add_result(self):
        """Добавление нового опыта с пользовательскими параметрами"""
        if not self.exp_name_var.get():
            messagebox.showwarning("Внимание", "Сначала создайте или загрузите эксперимент!")
            return

        # Создаем диалог для ввода параметров
        param_dialog = CustomExperimentDialog(self.root, "Добавить новый опыт", self.get_parameter_names())
        if param_dialog.result:
            # Получаем введенные параметры
            custom_params = param_dialog.result

            # Получаем следующий номер опыта
            experiment_name = self.exp_name_var.get()
            self.cursor.execute('''
                SELECT MAX(run_order) FROM experiment_results 
                WHERE experiment_id = (SELECT id FROM experiments WHERE name = ?)
            ''', (experiment_name,))

            max_order = self.cursor.fetchone()[0] or 0
            new_order = max_order + 1

            # Сохраняем в БД
            try:
                self.cursor.execute('''
                    INSERT INTO experiment_results (experiment_id, run_order, parameters_values)
                    VALUES (
                        (SELECT id FROM experiments WHERE name = ?),
                        ?, ?
                    )
                ''', (experiment_name, new_order, json.dumps(custom_params)))

                self.conn.commit()

                # Обновляем отображение
                self.load_experiment_results(experiment_name)

                # Теперь запрашиваем результат для этого опыта
                selection = self.results_tree.selection()
                if selection:
                    run_id = selection[0]
                    dialog = ResultDialog(self.root, "Добавить результат")
                    if dialog.result:
                        result_value, notes = dialog.result
                        self.save_result_to_db(run_id, result_value, notes)
                        self.load_experiment_results(experiment_name)  # Обновляем еще раз

                messagebox.showinfo("Успех", "Новый опыт успешно добавлен!")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при добавлении опыта: {str(e)}")

    def get_parameter_names(self):
        """Получение списка имен параметров"""
        param_names = []
        for item in self.params_tree.get_children():
            values = self.params_tree.item(item, 'values')
            if len(values) >= 1:
                param_names.append(values[0])
        return param_names

    def edit_result(self):
        """Редактирование результата эксперимента"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите опыт для редактирования!")
            return

        run_id = selection[0]
        current_values = self.results_tree.item(run_id, 'values')

        dialog = ResultDialog(self.root, "Редактировать результат",
                              current_values[-2], current_values[-1])
        if dialog.result:
            result_value, notes = dialog.result

            # Обновляем таблицу
            new_values = list(current_values)
            new_values[-2] = result_value
            new_values[-1] = notes
            self.results_tree.item(run_id, values=new_values)

            # Сохраняем в БД
            self.save_result_to_db(run_id, result_value, notes)

    def save_result_to_db(self, run_order, result_value, notes):
        """Сохранение результата в БД"""
        experiment_name = self.exp_name_var.get()

        try:
            self.cursor.execute('''
                UPDATE experiment_results 
                SET results = ?, notes = ?
                WHERE experiment_id = (SELECT id FROM experiments WHERE name = ?) 
                AND run_order = ?
            ''', (result_value, notes, experiment_name, int(run_order)))

            self.conn.commit()

        except Exception as e:
            messagebox.showerror("Ошибка БД", f"Ошибка при сохранении результата: {str(e)}")

    def delete_experiment(self):
        """Удаление эксперимента"""
        selection = self.experiments_listbox.curselection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите эксперимент для удаления!")
            return

        experiment_name = self.experiments_listbox.get(selection[0])

        if messagebox.askyesno("Подтверждение",
                               f"Вы уверены, что хотите удалить эксперимент '{experiment_name}'?"):
            try:
                # Удаляем из БД
                self.cursor.execute("DELETE FROM experiments WHERE name = ?", (experiment_name,))
                self.conn.commit()

                # Обновляем интерфейс
                self.load_experiments_list()
                self.exp_name_var.set("")
                self.exp_desc_text.delete(1.0, tk.END)
                self.clear_parameters()
                self.clear_results()

                messagebox.showinfo("Успех", "Эксперимент удален!")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при удалении: {str(e)}")

    def export_to_csv(self):
        """Экспорт результатов в CSV"""
        experiment_name = self.exp_name_var.get()
        if not experiment_name:
            messagebox.showwarning("Внимание", "Сначала загрузите или создайте эксперимент!")
            return

        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{experiment_name}_results.csv"
        )

        if filename:
            try:
                # Собираем данные из таблицы
                data = []
                columns = self.results_tree['columns']

                # Заголовки
                headers = [self.results_tree.heading(col)['text'] for col in columns]
                data.append(headers)

                # Данные
                for item in self.results_tree.get_children():
                    values = self.results_tree.item(item, 'values')
                    data.append(values)

                # Создаем DataFrame и сохраняем
                df = pd.DataFrame(data[1:], columns=data[0])
                df.to_csv(filename, index=False, encoding='utf-8')

                messagebox.showinfo("Успех", f"Данные экспортированы в {filename}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при экспорте: {str(e)}")

    def __del__(self):
        """Закрытие соединения с БД при удалении объекта"""
        if hasattr(self, 'conn'):
            self.conn.close()


class CustomExperimentDialog(simpledialog.Dialog):
    """Диалог для ввода пользовательских параметров опыта"""

    def __init__(self, parent, title, param_names):
        self.param_names = param_names
        self.result = None
        super().__init__(parent, title)

    def body(self, frame):
        self.entries = {}

        for i, param_name in enumerate(self.param_names):
            ttk.Label(frame, text=f"{param_name}:").grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(frame, width=15)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[param_name] = entry

        return list(self.entries.values())[0] if self.entries else None

    def validate(self):
        try:
            self.result = {}
            for param_name, entry in self.entries.items():
                value = entry.get().strip()
                if not value:
                    messagebox.showerror("Ошибка", f"Введите значение для параметра '{param_name}'!")
                    return False
                self.result[param_name] = float(value)
            return True
        except ValueError:
            messagebox.showerror("Ошибка", "Все значения должны быть числами!")
            return False


class ParameterDialog(simpledialog.Dialog):
    """Диалог для ввода параметров"""

    def __init__(self, parent, title, initial_values=None):
        self.result = None
        self.initial_values = initial_values or ("", "", "")
        super().__init__(parent, title)

    def body(self, frame):
        ttk.Label(frame, text="Название параметра:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_entry = ttk.Entry(frame, width=20)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Нижняя граница:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.low_entry = ttk.Entry(frame, width=20)
        self.low_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Верхняя граница:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.high_entry = ttk.Entry(frame, width=20)
        self.high_entry.grid(row=2, column=1, padx=5, pady=5)

        # Заполняем начальные значения если есть
        if self.initial_values:
            self.name_entry.insert(0, self.initial_values[0])
            self.low_entry.insert(0, str(self.initial_values[1]))
            self.high_entry.insert(0, str(self.initial_values[2]))

        return self.name_entry  # initial focus

    def validate(self):
        try:
            name = self.name_entry.get().strip()
            low = float(self.low_entry.get())
            high = float(self.high_entry.get())

            if not name:
                messagebox.showerror("Ошибка", "Введите название параметра!")
                return False

            if low >= high:
                messagebox.showerror("Ошибка", "Нижняя граница должна быть меньше верхней!")
                return False

            self.result = (name, low, high)
            return True

        except ValueError:
            messagebox.showerror("Ошибка", "Границы должны быть числами!")
            return False


class ResultDialog(simpledialog.Dialog):
    """Диалог для ввода результатов"""

    def __init__(self, parent, title, initial_result="", initial_notes=""):
        self.result = None
        self.initial_result = initial_result
        self.initial_notes = initial_notes
        super().__init__(parent, title)

    def body(self, frame):
        ttk.Label(frame, text="Результат:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.result_entry = ttk.Entry(frame, width=20)
        self.result_entry.grid(row=0, column=1, padx=5, pady=5)
        self.result_entry.insert(0, self.initial_result)

        ttk.Label(frame, text="Примечания:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.notes_text = tk.Text(frame, width=30, height=4)
        self.notes_text.grid(row=1, column=1, padx=5, pady=5)
        self.notes_text.insert(1.0, self.initial_notes)

        return self.result_entry

    def validate(self):
        result = self.result_entry.get().strip()
        notes = self.notes_text.get(1.0, tk.END).strip()

        if not result:
            messagebox.showerror("Ошибка", "Введите значение результата!")
            return False

        self.result = (result, notes)
        return True


if __name__ == "__main__":
    root = tk.Tk()
    app = AdaptiveExperimentDesignApp(root)
    root.mainloop()