import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import json
import pandas as pd
from core.services import ExperimentService
from core.services import PredictionService
from core.models import ExperimentModel
from core.gui.dialogs import ParameterDialog
from core.gui.dialogs import ResultDialog
from core.gui.dialogs import CustomExperimentDialog


class MainWindow:
    def __init__(self, root, experiment_service, prediction_service, db_manager):
        self.root = root
        self.experiment_service = experiment_service
        self.prediction_service = prediction_service
        self.db_manager = db_manager
        self.experiment_model = ExperimentModel()

        self.current_experiment = None
        self.param_names = []

        self.create_widgets()
        self.load_experiments_list()

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
        self.create_left_panel(main_frame)

        # Правая панель - параметры эксперимента
        self.create_right_panel(main_frame)

        # Нижняя панель - результаты
        self.create_results_panel(main_frame)

    def create_left_panel(self, parent):
        """Создание левой панели управления"""
        left_frame = ttk.LabelFrame(parent, text="Управление экспериментами", padding="10")
        left_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Кнопки управления
        ttk.Button(left_frame, text="Новый эксперимент",
                   command=self.create_new_experiment).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Удалить эксперимент",
                   command=self.delete_experiment).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Загрузить эксперимент",
                   command=self.load_experiment).pack(fill=tk.X, pady=5)

        # Настройки адаптивного режима
        self.create_adaptive_settings(left_frame)

        # Список экспериментов
        self.create_experiments_list(left_frame)

    def create_adaptive_settings(self, parent):
        """Создание настроек адаптивного режима"""
        adaptive_frame = ttk.LabelFrame(parent, text="Адаптивный режим", padding="5")
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

    def create_experiments_list(self, parent):
        """Создание списка экспериментов"""
        ttk.Label(parent, text="Список экспериментов:").pack(anchor=tk.W, pady=(10, 5))
        self.experiments_listbox = tk.Listbox(parent, height=12)
        self.experiments_listbox.pack(fill=tk.BOTH, expand=True)
        self.experiments_listbox.bind('<<ListboxSelect>>', self.on_experiment_select)

    def create_right_panel(self, parent):
        """Создание правой панели параметров"""
        right_frame = ttk.LabelFrame(parent, text="Параметры эксперимента", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        right_frame.columnconfigure(1, weight=1)

        # Название и описание
        self.create_experiment_info(right_frame)

        # Параметры
        self.create_parameters_section(right_frame)

        # Кнопки генерации
        self.create_generation_buttons(right_frame)

    def create_experiment_info(self, parent):
        """Создание секции информации об эксперименте"""
        ttk.Label(parent, text="Название:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.exp_name_var = tk.StringVar()
        self.exp_name_entry = ttk.Entry(parent, textvariable=self.exp_name_var)
        self.exp_name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))

        ttk.Label(parent, text="Описание:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.exp_desc_text = tk.Text(parent, height=3, width=40)
        self.exp_desc_text.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))

    def create_parameters_section(self, parent):
        """Создание секции параметров"""
        params_frame = ttk.Frame(parent)
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

    def create_generation_buttons(self, parent):
        """Создание кнопок генерации"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Сгенерировать базовый план",
                   command=self.generate_initial_design).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Предложить следующий опыт",
                   command=self.suggest_next_experiment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Обновить модель",
                   command=self.update_prediction_model).pack(side=tk.LEFT, padx=5)

    def create_results_panel(self, parent):
        """Создание панели результатов"""
        results_frame = ttk.LabelFrame(parent, text="План эксперимента и результаты", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Таблица результатов
        self.results_tree = ttk.Treeview(results_frame, height=15)
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Кнопки для работы с результатами
        self.create_results_buttons(results_frame)

    def create_results_buttons(self, parent):
        """Создание кнопок для работы с результатами"""
        results_buttons_frame = ttk.Frame(parent)
        results_buttons_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(results_buttons_frame, text="Добавить опыт",
                   command=self.add_custom_experiment).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_buttons_frame, text="Добавить результат",
                   command=self.add_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_buttons_frame, text="Редактировать результат",
                   command=self.edit_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_buttons_frame, text="Экспорт в CSV",
                   command=self.export_to_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_buttons_frame, text="Анализ зависимости",
                   command=self.show_dependency_analysis).pack(side=tk.LEFT, padx=5)

    # Методы бизнес-логики
    def load_experiments_list(self):
        """Загрузка списка экспериментов"""
        self.experiments_listbox.delete(0, tk.END)
        experiments = self.experiment_service.get_experiments_list()
        for exp in experiments:
            self.experiments_listbox.insert(tk.END, exp)

    def create_new_experiment(self):
        """Создание нового эксперимента"""
        name = simpledialog.askstring("Новый эксперимент", "Введите название эксперимента:")
        if name:
            if self.experiment_service.load_experiment(name):
                messagebox.showerror("Ошибка", "Эксперимент с таким названием уже существует!")
                return

            self.exp_name_var.set(name)
            self.exp_desc_text.delete(1.0, tk.END)
            self.clear_parameters()
            self.clear_results()
            messagebox.showinfo("Успех", f"Создан новый эксперимент: {name}\nТеперь добавьте параметры.")

    def save_experiment_metadata(self):
        """Сохранение метаданных эксперимента"""
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

        success = self.experiment_service.save_experiment(
            name, parameters, description, adaptive_mode, target_type
        )

        if success:
            self.load_experiments_list()
            return True
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить эксперимент!")
            return False

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

    def generate_initial_design(self):
        """Генерация начального плана Бокса-Бенкина"""
        if not self.save_experiment_metadata():
            return

        try:
            # Получаем параметры
            parameters = {}
            for item in self.params_tree.get_children():
                values = self.params_tree.item(item, 'values')
                if len(values) == 3:
                    parameters[values[0]] = {
                        'low': float(values[1]),
                        'high': float(values[2])
                    }

            if len(parameters) < 3:
                messagebox.showerror("Ошибка", "Для плана Бокса-Бенкина нужно минимум 3 параметра!")
                return

            # Генерируем план
            design_matrix = self.experiment_model.generate_box_behnken(len(parameters))
            scaled_design, param_names = self.experiment_model.scale_design(design_matrix, parameters)

            # Сохраняем и отображаем
            self.experiment_service.save_experiment_results(self.exp_name_var.get(), scaled_design)
            self.display_design(scaled_design, param_names)

            messagebox.showinfo("Успех", "Базовый план эксперимента успешно сгенерирован!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при генерации плана: {str(e)}")

    def display_design(self, design, param_names):
        """Отображение плана в таблице"""
        self.clear_results()

        # Создаем колонки
        self.results_tree['columns'] = ['run_order'] + param_names + ['results', 'notes']
        self.results_tree.heading('run_order', text='№ опыта')
        self.results_tree.column('run_order', width=80)

        for param in param_names:
            self.results_tree.heading(param, text=param)
            self.results_tree.column(param, width=200)

        self.results_tree.heading('results', text='Результаты')
        self.results_tree.column('results', width=250)

        self.results_tree.heading('notes', text='Примечания')
        self.results_tree.column('notes', width=300)

        # Заполняем данными
        for i, run in enumerate(design, 1):
            values = [i] + [f"{run[param]:.4f}" for param in param_names] + ['', '']
            self.results_tree.insert('', tk.END, values=values, iid=str(i))

    def update_prediction_model(self):
        """Обновление модели предсказания"""
        try:
            X, y, param_names = self.experiment_service.get_training_data(self.exp_name_var.get())
            score = self.prediction_service.update_model(X, y)
            self.prediction_service.param_names = param_names

            messagebox.showinfo("Модель обновлена",
                                f"Модель успешно обучена!\nR² score: {score:.3f}\n"
                                f"Обучено на {len(X)} опытах.")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def suggest_next_experiment(self):
        """Предложение следующего опыта"""
        if not self.prediction_service.current_model:
            messagebox.showwarning("Внимание", "Сначала обновите модель на основе имеющихся данных!")
            return

        if not self.adaptive_var.get():
            messagebox.showwarning("Внимание", "Включите адаптивный режим для использования этой функции!")
            return

        try:
            bounds = self.experiment_service.get_parameter_bounds(self.params_tree)
            existing_points = self.experiment_service.get_existing_points(
                self.exp_name_var.get(),
                self.prediction_service.param_names
            )
            target_type = self.target_var.get()
            target_value = float(self.target_value_var.get())

            suggested_params = self.prediction_service.suggest_next_experiment(
                bounds, existing_points, target_type, target_value
            )

            # Преобразуем в словарь параметров
            param_dict = {
                name: float(value)
                for name, value in zip(self.prediction_service.param_names, suggested_params)
            }

            self.show_suggestion_dialog(param_dict)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

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
            predicted_result = self.prediction_service.predict_result(suggested_params)
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
        max_order = self.experiment_service.get_max_run_order(experiment_name)
        new_order = max_order + 1

        # Сохраняем в БД
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO experiment_results (experiment_id, run_order, parameters_values, is_suggested)
                VALUES (
                    (SELECT id FROM experiments WHERE name = ?),
                    ?, ?, 1
                )
            ''', (experiment_name, new_order, json.dumps(suggested_params)))

            conn.commit()
            conn.close()

            # Обновляем отображение
            self.load_experiment_results(experiment_name)

            messagebox.showinfo("Успех", "Предложенный опыт добавлен в план эксперимента!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при добавлении опыта: {str(e)}")

    def add_custom_experiment(self):
        """Добавление нового опыта с пользовательскими параметрами"""
        if not self.exp_name_var.get():
            messagebox.showwarning("Внимание", "Сначала создайте или загрузите эксперимент!")
            return

        param_names = self.experiment_service.get_parameter_names(self.params_tree)
        if not param_names:
            messagebox.showwarning("Внимание", "Сначала добавьте параметры эксперимента!")
            return

        dialog = CustomExperimentDialog(self.root, "Добавить новый опыт", param_names)
        if dialog.result:
            custom_params = dialog.result

            # Получаем следующий номер опыта
            max_order = self.experiment_service.get_max_run_order(self.exp_name_var.get())
            new_order = max_order + 1

            # Сохраняем в БД
            try:
                conn = self.db_manager.get_connection()
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO experiment_results (experiment_id, run_order, parameters_values)
                    VALUES (
                        (SELECT id FROM experiments WHERE name = ?),
                        ?, ?
                    )
                ''', (self.exp_name_var.get(), new_order, json.dumps(custom_params)))

                conn.commit()
                conn.close()

                # Обновляем отображение
                self.load_experiment_results(self.exp_name_var.get())

                messagebox.showinfo("Успех", "Новый опыт успешно добавлен!")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при добавлении опыта: {str(e)}")

    def add_result(self):
        """Добавление результата эксперимента"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите опыт для добавления результата!")
            return

        run_id = selection[0]

        dialog = ResultDialog(self.root, "Добавить результат")
        if dialog.result:
            result_value, notes = dialog.result

            # Обновляем таблицу
            current_values = list(self.results_tree.item(run_id, 'values'))
            current_values[-2] = result_value
            current_values[-1] = notes
            self.results_tree.item(run_id, values=current_values)

            # Сохраняем в БД
            self.save_result_to_db(run_id, result_value, notes)

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
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE experiment_results 
                SET results = ?, notes = ?
                WHERE experiment_id = (SELECT id FROM experiments WHERE name = ?) 
                AND run_order = ?
            ''', (result_value, notes, experiment_name, int(run_order)))

            conn.commit()
            conn.close()

        except Exception as e:
            messagebox.showerror("Ошибка БД", f"Ошибка при сохранении результата: {str(e)}")

    def show_dependency_analysis(self):
        """Показать анализ зависимостей"""
        if not self.experiment_service.has_completed_experiments(self.exp_name_var.get()):
            messagebox.showwarning("Внимание", "Недостаточно данных для анализа.")
            return

        try:
            X, y, param_names = self.experiment_service.get_training_data(self.exp_name_var.get())

            if len(X) < 3:
                messagebox.showwarning("Внимание", "Недостаточно данных для анализа.")
                return

            # Создаем окно анализа
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Анализ зависимостей")
            analysis_window.geometry("600x400")

            # Анализ важности признаков
            importances = self.prediction_service.get_feature_importance()
            if importances is not None:
                importance_frame = ttk.LabelFrame(analysis_window, text="Важность параметров", padding="10")
                importance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                tree = ttk.Treeview(importance_frame, columns=('parameter', 'importance'), show='headings')
                tree.heading('parameter', text='Параметр')
                tree.heading('importance', text='Важность')
                tree.column('parameter', width=200)
                tree.column('importance', width=150)
                tree.pack(fill=tk.BOTH, expand=True)

                for feature, importance in zip(param_names, importances):
                    tree.insert('', tk.END, values=(feature, f"{importance:.4f}"))

                ttk.Label(importance_frame,
                          text="Чем выше значение, тем сильнее параметр влияет на результат",
                          font=('Arial', 8)).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе: {str(e)}")

    def export_to_csv(self):
        """Экспорт результатов в CSV"""
        experiment_name = self.exp_name_var.get()
        if not experiment_name:
            messagebox.showwarning("Внимание", "Сначала загрузите или создайте эксперимент!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{experiment_name}_results.csv"
        )

        if filename:
            try:
                df = self.experiment_service.export_to_csv(experiment_name, self.results_tree)
                df.to_csv(filename, index=False, encoding='utf-8')
                messagebox.showinfo("Успех", f"Данные экспортированы в {filename}")

            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

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
            experiment_data = self.experiment_service.load_experiment(experiment_name)
            if experiment_data:
                # Заполняем поля
                self.exp_name_var.set(experiment_data['name'])
                self.exp_desc_text.delete(1.0, tk.END)
                self.exp_desc_text.insert(1.0, experiment_data['description'] or "")
                self.adaptive_var.set(experiment_data['adaptive_mode'])
                self.target_var.set(experiment_data['target_type'])

                # Заполняем параметры
                self.clear_parameters()
                for param_name, bounds in experiment_data['parameters'].items():
                    self.params_tree.insert('', tk.END,
                                            values=(param_name, bounds['low'], bounds['high']))

                # Загружаем результаты
                self.load_experiment_results(experiment_name)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке эксперимента: {str(e)}")

    def load_experiment_results(self, experiment_name):
        """Загрузка результатов эксперимента"""
        try:
            results = self.experiment_service.get_experiment_results(experiment_name)

            if results:
                # Получаем имена параметров из первого результата
                first_result = json.loads(results[0][1])
                param_names = list(first_result.keys())

                # Создаем таблицу
                self.display_results_table(results, param_names)
            else:
                self.clear_results()

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
                conn = self.db_manager.get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM experiments WHERE name = ?", (experiment_name,))
                conn.commit()
                conn.close()

                # Обновляем интерфейс
                self.load_experiments_list()
                if self.exp_name_var.get() == experiment_name:
                    self.exp_name_var.set("")
                    self.exp_desc_text.delete(1.0, tk.END)
                    self.clear_parameters()
                    self.clear_results()

                messagebox.showinfo("Успех", "Эксперимент удален!")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при удалении: {str(e)}")

    def clear_parameters(self):
        """Очистка таблицы параметров"""
        for item in self.params_tree.get_children():
            self.params_tree.delete(item)

    def clear_results(self):
        """Очистка таблицы результатов"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)